from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2_pytorch import AdamAtan2 as AdamATan2

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    name: str
    loss: LossConfig


class TrainingStageConfig(pydantic.BaseModel):
    """Configuration for a single training stage"""
    data_path: str
    epochs: int
    eval_interval: Optional[int] = None
    lr: Optional[float] = None  # If None, use parent config lr
    lr_min_ratio: Optional[float] = None
    lr_warmup_steps: Optional[int] = None


class SequentialPretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig

    # Training stages (list of datasets to train on sequentially)
    training_stages: List[TrainingStageConfig]

    # Hyperparams (defaults for all stages)
    global_batch_size: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    checkpoint_every_stage: bool = True  # Save checkpoint after each stage
    eval_save_outputs: List[str] = []


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader(data_path: str, split: str, rank: int, world_size: int, seed: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=seed,
        dataset_path=data_path,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def create_model(config: SequentialPretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore

        batch_size=config.global_batch_size // world_size,

        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model, dynamic=False)  # type: ignore

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    optimizers = []
    optimizer_lrs = []

    # Only add puzzle embedding optimizer if puzzle embeddings are enabled
    if model.model.puzzle_emb is not None:
        optimizers.append(
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore

                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,

                world_size=world_size
            )
        )
        optimizer_lrs.append(config.puzzle_emb_lr)

    # Main model optimizer
    optimizers.append(
        AdamATan2(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2)
        )
    )
    optimizer_lrs.append(config.lr)

    return model, optimizers, optimizer_lrs


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: SequentialPretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int, total_steps: int):
    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(checkpoint_path: Optional[str], train_state: TrainState, stage_name: str = ""):
    if checkpoint_path is None:
        return

    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_name = f"step_{train_state.step}"
    if stage_name:
        checkpoint_name += f"_{stage_name}"
    torch.save(train_state.model.state_dict(), os.path.join(checkpoint_path, checkpoint_name))


def load_checkpoint(checkpoint_path: str, train_state: TrainState, rank: int, world_size: int):
    """Load checkpoint weights into existing model"""
    if rank == 0:
        print(f"Loading checkpoint from {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cuda")

    # Remove "_orig_mod." prefix if model was compiled
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    train_state.model.load_state_dict(state_dict, strict=True)

    # Broadcast to all ranks
    if world_size > 1:
        with torch.no_grad():
            for param in list(train_state.model.parameters()) + list(train_state.model.buffers()):
                dist.broadcast(param, src=0)


def compute_lr(base_lr: float, lr_min_ratio: float, lr_warmup_steps: int, train_state: TrainState, stage_step: int, stage_total_steps: int):
    """Compute learning rate for current step within a stage"""
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=stage_step,
        base_lr=base_lr,
        num_warmup_steps=round(lr_warmup_steps),
        num_training_steps=stage_total_steps,
        min_ratio=lr_min_ratio
    )


def train_batch(config: SequentialPretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int,
                rank: int, world_size: int, stage_step: int, stage_total_steps: int, stage_config: TrainingStageConfig):
    train_state.step += 1

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

    # Apply optimizer with stage-specific learning rates
    lr = stage_config.lr if stage_config.lr is not None else config.lr
    lr_min_ratio = stage_config.lr_min_ratio if stage_config.lr_min_ratio is not None else config.lr_min_ratio
    lr_warmup_steps = stage_config.lr_warmup_steps if stage_config.lr_warmup_steps is not None else config.lr_warmup_steps

    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        # Scale base_lr if stage has custom lr
        adjusted_base_lr = base_lr
        if stage_config.lr is not None and base_lr == config.lr:
            adjusted_base_lr = stage_config.lr

        lr_this_step = compute_lr(adjusted_base_lr, lr_min_ratio, lr_warmup_steps, train_state, stage_step, stage_total_steps)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step

        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}

            # Postprocess
            count = max(reduced_metrics["count"], 1)
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def evaluate(config: SequentialPretrainConfig, train_state: TrainState, eval_loader: torch.utils.data.DataLoader,
             eval_metadata: PuzzleDatasetMetadata, rank: int, world_size: int, checkpoint_path: Optional[str] = None):
    with torch.inference_mode():
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        all_preds = {}

        metric_keys = []
        metric_values = None
        metric_global_batch_size = [0 for _ in range(len(set_ids))]

        # Calculate total evaluation batches for progress bar
        total_eval_batches = sum(eval_metadata.total_groups_per_set.values())
        eval_progress_bar = None
        if rank == 0:
            eval_progress_bar = tqdm.tqdm(total=total_eval_batches, desc="Evaluating", leave=False)

        carry = None
        for set_name, batch, global_batch_size in eval_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            while True:
                carry, _, metrics, preds, all_finish = train_state.model(carry=carry, batch=batch, return_keys=config.eval_save_outputs)

                if all_finish:
                    break

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        all_preds.setdefault(k, [])
                        all_preds[k].append(v.cpu())

            del carry, preds, batch, all_finish

            # Aggregate
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros((len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda")

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            metric_global_batch_size[set_id] += global_batch_size

            # Update progress bar
            if eval_progress_bar is not None:
                eval_progress_bar.update(1)

        # Close progress bar
        if eval_progress_bar is not None:
            eval_progress_bar.close()

        if len(all_preds) and checkpoint_path is not None:
            all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}

            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(all_preds, os.path.join(checkpoint_path, f"step_{train_state.step}_all_preds.{rank}"))

        # Logging
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {set_name: {metric_name: reduced_metrics[set_id, metric_id] for metric_id, metric_name in enumerate(metric_keys)}
                                   for set_id, set_name in enumerate(set_ids)}

                # Postprocess
                for set_name, metrics in reduced_metrics.items():
                    count = metrics.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in metrics.items()}

                return reduced_metrics


def save_code_and_config(config: SequentialPretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)
            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> SequentialPretrainConfig:
    objects = [None]
    if rank == 0:
        config = SequentialPretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            stage_names = [os.path.basename(stage.data_path) for stage in config.training_stages]
            config.project_name = f"{'-'.join(stage_names).capitalize()} Sequential ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain_sequential", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1

    # Initialize distributed training if in distributed environment
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # Load synced config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs
    torch.random.manual_seed(config.seed + RANK)

    # Calculate total steps across all stages
    total_steps_all_stages = 0
    stage_total_steps = []

    for stage in config.training_stages:
        _, stage_metadata = create_dataloader(
            data_path=stage.data_path,
            split="train",
            rank=RANK,
            world_size=WORLD_SIZE,
            seed=config.seed,
            test_set_mode=False,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size
        )
        steps = int(stage.epochs * stage_metadata.total_groups * stage_metadata.mean_puzzle_examples / config.global_batch_size)
        stage_total_steps.append(steps)
        total_steps_all_stages += steps

    # Initialize on first stage
    first_stage = config.training_stages[0]
    train_loader, train_metadata = create_dataloader(
        data_path=first_stage.data_path,
        split="train",
        rank=RANK,
        world_size=WORLD_SIZE,
        seed=config.seed,
        test_set_mode=False,
        epochs_per_iter=first_stage.eval_interval if first_stage.eval_interval is not None else first_stage.epochs,
        global_batch_size=config.global_batch_size
    )

    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE, total_steps=total_steps_all_stages)

    # Progress bar and logger
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)

    # Sequential training loop
    for stage_idx, stage_config in enumerate(config.training_stages):
        stage_name = os.path.basename(stage_config.data_path)
        if RANK == 0:
            print(f"\n{'='*80}")
            print(f"Starting Stage {stage_idx + 1}/{len(config.training_stages)}: {stage_name}")
            print(f"Epochs: {stage_config.epochs}, Eval Interval: {stage_config.eval_interval}")
            print(f"{'='*80}\n")

        # Load data for this stage (unless it's the first stage, already loaded)
        if stage_idx > 0:
            train_loader, train_metadata = create_dataloader(
                data_path=stage_config.data_path,
                split="train",
                rank=RANK,
                world_size=WORLD_SIZE,
                seed=config.seed,
                test_set_mode=False,
                epochs_per_iter=stage_config.eval_interval if stage_config.eval_interval is not None else stage_config.epochs,
                global_batch_size=config.global_batch_size
            )

        eval_loader, eval_metadata = create_dataloader(
            data_path=stage_config.data_path,
            split="test",
            rank=RANK,
            world_size=WORLD_SIZE,
            seed=config.seed,
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size
        )

        # Reset carry for new dataset
        train_state.carry = None

        # Training iterations for this stage
        train_epochs_per_iter = stage_config.eval_interval if stage_config.eval_interval is not None else stage_config.epochs
        total_iters = stage_config.epochs // train_epochs_per_iter
        assert stage_config.epochs % train_epochs_per_iter == 0, f"Eval interval must be a divisor of epochs for stage {stage_idx}"

        stage_step = 0
        for iter_id in range(total_iters):
            if RANK == 0:
                print(f"[Stage {stage_idx + 1}, Rank {RANK}]: Epoch {iter_id * train_epochs_per_iter}/{stage_config.epochs}")

            # Train
            train_state.model.train()
            for set_name, batch, global_batch_size in train_loader:
                metrics = train_batch(
                    config, train_state, batch, global_batch_size,
                    rank=RANK, world_size=WORLD_SIZE,
                    stage_step=stage_step,
                    stage_total_steps=stage_total_steps[stage_idx],
                    stage_config=stage_config
                )
                stage_step += 1

                if RANK == 0 and metrics is not None:
                    # Add stage info to metrics
                    metrics[f"stage"] = stage_idx
                    wandb.log(metrics, step=train_state.step)
                    progress_bar.update(1)  # type: ignore

            # Evaluate
            train_state.model.eval()
            metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE, checkpoint_path=config.checkpoint_path)

            if RANK == 0 and metrics is not None:
                # Prefix metrics with stage name
                prefixed_metrics = {}
                for set_name, set_metrics in metrics.items():
                    for metric_name, value in set_metrics.items():
                        prefixed_metrics[f"eval/{set_name}/{metric_name}"] = value
                prefixed_metrics["stage"] = stage_idx
                wandb.log(prefixed_metrics, step=train_state.step)

            # Checkpoint
            if RANK == 0 and (config.checkpoint_every_eval or (iter_id == total_iters - 1)):
                save_train_state(config.checkpoint_path, train_state, stage_name=f"stage{stage_idx}_{stage_name}")

        # Save checkpoint after completing stage
        if RANK == 0 and config.checkpoint_every_stage:
            save_train_state(config.checkpoint_path, train_state, stage_name=f"stage{stage_idx}_{stage_name}_complete")
            print(f"\nCompleted Stage {stage_idx + 1}: {stage_name}\n")

    # Finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()

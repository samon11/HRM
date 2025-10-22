"""
Generate Kaggle submission file for ARC-AGI challenge.

This script loads a trained model checkpoint and generates predictions
for test puzzles in the format required by Kaggle, using the same logic
as arc_eval.ipynb for consistency.

Usage:
    python generate_submission.py checkpoint=<path_to_checkpoint> \
        data_path=<path_to_test_dataset> \
        output=submission.json
"""

from typing import List, Dict
import yaml
import os
import json
from collections import defaultdict
from glob import glob

import torch
import torch.distributed as dist
import numpy as np
from numba import njit

import pydantic
from omegaconf import OmegaConf
from pretrain import PretrainConfig, init_train_state, create_dataloader
from dataset.common import inverse_dihedral_transform


class SubmissionConfig(pydantic.BaseModel):
    checkpoint: str
    data_path: str = "data/arc-2-aug-1000"  # Default to ARC-2 test set
    output: str = "submission.json"
    num_attempts: int = 2  # Number of attempts per test case (Kaggle format)
    use_checkpoint_preds: bool = False  # Use pre-generated predictions from checkpoint dir


@njit
def crop(grid: np.ndarray):
    """
    Crop grid to remove padding and EOS tokens.
    Find maximum-sized rectangle without any EOS token inside.
    """
    grid = grid.reshape(30, 30)

    max_area = 0
    max_size = (0, 0)
    nr, nc = grid.shape

    num_c = nc
    for num_r in range(1, nr + 1):
        # Scan for maximum c
        for c in range(1, num_c + 1):
            x = grid[num_r - 1, c - 1]
            if (x < 2) | (x > 11):
                num_c = c - 1
                break

        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)

    return grid[:max_size[0], :max_size[1]] - 2


def grid_hash(grid: np.ndarray):
    """Create hash for grid to use as key."""
    return hash((grid.tobytes(), grid.shape))


def inverse_aug(name: str, grid: np.ndarray):
    """
    Inverse the augmentation transformation.
    Name format: <puzzle_name>_t<transform_id>_<perm>
    """
    if "_" not in name:
        return grid

    parts = name.split("_")
    if len(parts) < 3:
        return grid

    trans_id = parts[-2]
    perm = parts[-1]

    if not trans_id.startswith("t"):
        return grid

    trans_id = int(trans_id[1:])  # Remove "t" prefix
    inv_perm = np.argsort(list(perm))

    return inv_perm[inverse_dihedral_transform(grid, trans_id)]


def grid_to_list(grid: np.ndarray) -> List[List[int]]:
    """Convert numpy grid to list format for JSON."""
    return grid.astype(int).tolist()


def load_predictions_from_checkpoint(checkpoint_path: str):
    """Load pre-generated predictions from checkpoint directory (faster)."""
    all_preds = {}
    pred_pattern = f"{checkpoint_path}_all_preds.*"
    pred_files = glob(pred_pattern)

    if not pred_files:
        return None

    for filename in pred_files:
        preds = torch.load(filename, map_location="cpu")
        for k, v in preds.items():
            all_preds.setdefault(k, [])
            all_preds[k].append(v)
        del preds

    all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}
    return all_preds


def generate_predictions_from_model(config: SubmissionConfig, train_config: PretrainConfig, RANK: int, WORLD_SIZE: int):
    """Generate predictions by running the model."""
    # Load test dataset
    test_loader, test_metadata = create_dataloader(
        train_config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=train_config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE
    )

    # Load model
    train_state = init_train_state(train_config, test_metadata, world_size=WORLD_SIZE)

    # Load checkpoint weights
    try:
        train_state.model.load_state_dict(
            torch.load(config.checkpoint, map_location="cuda"),
            assign=True
        )
    except:
        # Handle torch.compile case
        state_dict = {
            k.removeprefix("_orig_mod."): v
            for k, v in torch.load(config.checkpoint, map_location="cuda").items()
        }
        train_state.model.load_state_dict(state_dict, assign=True)

    train_state.model.eval()

    if RANK == 0:
        print("Model loaded successfully")
        print("Generating predictions...")

    # Collect predictions
    all_preds = defaultdict(list)

    with torch.inference_mode():
        carry = None
        batch_idx = 0

        for set_name, batch, global_batch_size in test_loader:
            batch = {k: v.cuda() for k, v in batch.items()}

            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)

            # Run model until halting
            while True:
                carry, _, metrics, preds, all_finish = train_state.model(
                    carry=carry,
                    batch=batch,
                    return_keys=["inputs", "labels", "puzzle_identifiers", "logits", "q_halt_logits"]
                )

                if all_finish:
                    break

            # Collect predictions
            for k in ["inputs", "labels", "puzzle_identifiers", "logits", "q_halt_logits"]:
                if k in batch:
                    all_preds[k].append(batch[k].cpu())
                elif k in preds:
                    all_preds[k].append(preds[k].cpu())

            batch_idx += 1
            if RANK == 0 and batch_idx % 10 == 0:
                print(f"Processed {batch_idx} batches...")

    # Concatenate all batches
    all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}

    return all_preds


def generate_submission(config: SubmissionConfig):
    """Generate submission file from model checkpoint."""

    RANK = 0
    WORLD_SIZE = 1

    # Initialize distributed if in distributed environment
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # Load training config from checkpoint directory
    checkpoint_dir = os.path.dirname(config.checkpoint)
    with open(os.path.join(checkpoint_dir, "all_config.yaml"), "r") as f:
        train_config = PretrainConfig(**yaml.safe_load(f))
        train_config.data_path = config.data_path
        train_config.eval_save_outputs = ["inputs", "labels", "puzzle_identifiers", "logits", "q_halt_logits"]

    if RANK == 0:
        print(f"Loading checkpoint: {config.checkpoint}")
        print(f"Test dataset: {config.data_path}")
        print(f"Output file: {config.output}")

    # Load identifier mapping
    identifier_path = os.path.join(config.data_path, "identifiers.json")
    if not os.path.exists(identifier_path):
        # Fallback to dataset.json
        with open(os.path.join(config.data_path, "dataset.json"), "r") as f:
            dataset_metadata = json.load(f)
            identifier_map = {
                v: k for k, v in dataset_metadata.get("puzzle_identifiers", {}).items()
            }
    else:
        with open(identifier_path, "r") as f:
            identifier_map = json.load(f)

    # Get predictions (either load from checkpoint or generate)
    if config.use_checkpoint_preds:
        if RANK == 0:
            print("Loading pre-generated predictions from checkpoint directory...")
        all_preds = load_predictions_from_checkpoint(config.checkpoint)
        if all_preds is None:
            if RANK == 0:
                print("No pre-generated predictions found, generating from model...")
            all_preds = generate_predictions_from_model(config, train_config, RANK, WORLD_SIZE)
    else:
        all_preds = generate_predictions_from_model(config, train_config, RANK, WORLD_SIZE)

    # Remove padding
    PAD_PUZZLE_IDENTIFIER = 0
    mask = all_preds["puzzle_identifiers"] != PAD_PUZZLE_IDENTIFIER
    all_preds = {k: v[mask] for k, v in all_preds.items()}

    if RANK == 0:
        print(f"Processing {len(all_preds['puzzle_identifiers'])} predictions...")

    # Get argmax predictions
    preds = all_preds["logits"].argmax(-1)

    # Build global hash map and puzzle structure (following arc_eval.ipynb)
    global_hmap = {}
    puzzle_test_cases = {}  # puzzle_name -> {input_hash -> label_hash}

    # First pass: identify test cases for each puzzle
    for identifier, input_seq, label_seq in zip(
        all_preds["puzzle_identifiers"],
        all_preds["inputs"],
        all_preds["labels"]
    ):
        name = identifier_map[int(identifier.item())]

        # Only use non-augmented versions to identify test cases
        if "_" not in name:
            puzzle_test_cases.setdefault(name, {})

            input_grid = crop(input_seq.numpy())
            label_grid = crop(label_seq.numpy())

            input_hash = grid_hash(input_grid)
            label_hash = grid_hash(label_grid)

            global_hmap[input_hash] = input_grid
            global_hmap[label_hash] = label_grid

            puzzle_test_cases[name][input_hash] = label_hash

    if RANK == 0:
        print(f"Found {len(puzzle_test_cases)} unique puzzles")

    # Second pass: collect all predictions for each test case
    pred_answers = defaultdict(lambda: defaultdict(list))

    for identifier, input_seq, pred_seq, q_halt in zip(
        all_preds["puzzle_identifiers"],
        all_preds["inputs"],
        preds,
        all_preds["q_halt_logits"].sigmoid()
    ):
        name = identifier_map[int(identifier.item())]
        orig_name = name.split("_")[0]

        # Get original input (inverse augmentation)
        input_grid = inverse_aug(name, crop(input_seq.numpy()))
        input_hash = grid_hash(input_grid)

        # Skip if this test case doesn't exist (shouldn't happen)
        if orig_name not in puzzle_test_cases or input_hash not in puzzle_test_cases[orig_name]:
            continue

        # Get prediction (inverse augmentation)
        pred_grid = inverse_aug(name, crop(pred_seq.numpy()))
        pred_hash = grid_hash(pred_grid)
        global_hmap[pred_hash] = pred_grid

        # Store prediction with quality score
        pred_answers[orig_name][input_hash].append((pred_hash, q_halt.item()))

    # Generate submission in Kaggle format
    submission = {}

    for puzzle_name, test_cases in puzzle_test_cases.items():
        puzzle_submissions = []

        for input_hash, label_hash in test_cases.items():
            # Get all predictions for this test case
            predictions = pred_answers[puzzle_name][input_hash]

            # Aggregate by prediction hash (count votes and average q-value)
            pred_map = {}
            for pred_hash, q_value in predictions:
                if pred_hash not in pred_map:
                    pred_map[pred_hash] = [0, 0.0]
                pred_map[pred_hash][0] += 1  # vote count
                pred_map[pred_hash][1] += q_value  # sum of q-values

            # Average q-values
            for pred_hash, stats in pred_map.items():
                stats[1] /= stats[0]

            # Sort by vote count (descending), then by q-value (descending)
            sorted_preds = sorted(
                pred_map.items(),
                key=lambda kv: (kv[1][0], kv[1][1]),
                reverse=True
            )

            # Select top N predictions
            attempts = {}
            for i in range(config.num_attempts):
                if i < len(sorted_preds):
                    pred_hash, _ = sorted_preds[i]
                    pred_grid = global_hmap[pred_hash]
                else:
                    # Use first prediction if we don't have enough unique ones
                    pred_hash, _ = sorted_preds[0] if sorted_preds else (None, None)
                    pred_grid = global_hmap.get(pred_hash, np.array([[0, 0], [0, 0]]))

                attempts[f"attempt_{i+1}"] = grid_to_list(pred_grid)

            puzzle_submissions.append(attempts)

        submission[puzzle_name] = puzzle_submissions

    # Save submission
    if RANK == 0:
        with open(config.output, "w") as f:
            json.dump(submission, f)

        print(f"\nSubmission saved to {config.output}")
        print(f"Total puzzles: {len(submission)}")
        total_test_cases = sum(len(cases) for cases in submission.values())
        print(f"Total test cases: {total_test_cases}")

        # Show sample
        sample_key = list(submission.keys())[0]
        print(f"\nSample entry for puzzle '{sample_key}':")
        print(json.dumps({sample_key: submission[sample_key]}, indent=2)[:500] + "...")

    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    config = SubmissionConfig(**OmegaConf.to_container(OmegaConf.from_cli()))  # type: ignore
    generate_submission(config)


if __name__ == "__main__":
    main()

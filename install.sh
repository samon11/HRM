unzip data.zip

pip install packaging ninja wheel setuptools setuptools-scm
pip install -v flash-attn
pip install -r requirements.txt

# python pretrain.py data_path=data/arc-2-aug-1000 global_batch_size=192 epochs=100 eval_interval=100 lr=1e-4 puzzle_emb_lr=1e-2 weight_decay=0.1 puzzle_emb_weight_decay=0.1 arch=hrm_v1_no_puzzle_emb
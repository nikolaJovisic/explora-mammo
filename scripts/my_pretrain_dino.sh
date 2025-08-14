#!/usr/bin/env bash
set -euo pipefail

ENV="/home/nikola.jovisic.ivi/.conda/envs/explora"

export WANDB__SERVICE_WAIT=300
export WANDB_MODE=online
export WANDB_ENTITY=ivi-cvrs
export WANDB_PROJECT=explora-mammo
export PYTHONPATH=.

num_gpus=6
cfg_file="dinov2/configs/train/fmow_vitb14.yaml"
out_dir="/lustre/nj/explora"

"$ENV/bin/torchrun" --nproc_per_node=$num_gpus  dinov2/train/train.py \
  --wandb=$WANDB_PROJECT \
  --config-file="$cfg_file" \
  --output_dir="$out_dir" \
  lora.unfreeze_blocks="[23]" lora.rank=64 lora.unfreeze_cls_token="true" \
  optim.base_lr=0.001 optim.freeze_last_layer_epochs=3 optim.accum_iter=32 \
  train.num_workers=8 train.batch_size_per_gpu=8


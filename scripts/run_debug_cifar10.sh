#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python SAGE.py \
  --dataset=CIFAR10 \
  --alpha=0.1 \
  --gpu_id=0 \
  --num_clients=4 \
  --num_online_clients=2 \
  --local_epochs=1 \
  --batch_size_local_labeled_fixmatch=32 \
  --batch_size_test=256 \
  --debug_rounds=3 \
  --debug_num_labeled=20 \
  --debug_train_limit=5000 \
  --debug_test_limit=1000 \
  --disable_dataset_replication

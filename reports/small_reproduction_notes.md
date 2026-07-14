# Small CIFAR-10 Sanity Reproduction Notes

## Goal

This setup is for a small-scale CIFAR-10 sanity reproduction of the SAGE training path. It is not a full paper reproduction and should not be used to compare final accuracy with the reported results.

The smoke test keeps the federated semi-supervised structure intact: labeled samples are still selected per class, the remaining training samples still form the unlabeled pool, and both pools are still partitioned across clients with the existing IID or Dirichlet non-IID logic controlled by `alpha`.

## Changed files

- `options.py`: added debug-only CLI arguments for small runs.
- `SAGE.py`: wired debug overrides for rounds, labeled samples per class, class-balanced training limits, and class-balanced test limits when labels are available.
- `Dataset/dataset.py`: added an option to disable only artificial per-client dataset list replication.
- `scripts/run_debug_cifar10.sh`: added a tiny CIFAR-10 CUDA smoke-test command.
- `reports/small_reproduction_notes.md`: added these notes.

## Run on a CUDA GPU machine

From the repository root:

```bash
bash scripts/run_debug_cifar10.sh
```

The script runs:

```bash
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
```

## Outputs

Results are saved under:

- `results/CIFAR10/SAGE α=0.1.csv`
- `results/CIFAR10/logs/SAGE α=0.1.log`

## Successful smoke test

A successful smoke test should download or load CIFAR-10, print client labeled and unlabeled class distributions, complete 3 server rounds, print an accuracy after each round, and write the CSV/log files above.

The accuracy is only a sanity signal for code execution. It is expected to be noisy and low because the run is intentionally tiny.

## CUDA requirement

Do not run this smoke test on a Mac without CUDA. The project still uses explicit `.cuda(args.gpu_id)` calls for models, batches, loss, and seeding. This debug setup intentionally does not add CPU or Apple Silicon support.

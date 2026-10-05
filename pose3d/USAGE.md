# pose3d usage

## Setup

Python `>=3.11,<3.15`, [Poetry](https://python-poetry.org/):

```bash
cd pose3d
poetry install
```

`clifford` and `image2sphere` are forks installed from git (see `pyproject.toml`). The Pascal3D+
dataset is read through `image2sphere.pascal_dataset.Pascal3D`.

Poetry is not required where torch is already installed (Kaggle): install only what is missing and
run `python -m pose3d` from `pose3d/`, no install of the package itself.

```bash
pip install e3nn==0.5.9 healpy==1.19.0 \
  git+https://github.com/chagrygoris/image2sphere.git \
  git+https://github.com/chagrygoris/clifford-group-equivariant-neural-networks.git
```

On Kaggle, pass a constraints file pinning the preinstalled packages (`-c constraints.txt`, as in the
runner notebook) so `pip` does not replace torch.

## Run

Run from the `pose3d/` folder. The defaults are the reference recipe:

```bash
poetry run python -m pose3d --path_to_datasets /path/to/data --run_name my-run
```

Every option in `pose3d/config.py` is a flag of the same name; `python -m pose3d --help` lists them.

```bash
poetry run python -m pose3d --path_to_datasets ... --use_warp          # feature on
poetry run python -m pose3d --path_to_datasets ... --no-medoid_eval    # feature off
poetry run python -m pose3d --path_to_datasets ... --model i2s_real --encoder resnet101 --lr 1e-3 --n_epochs 10
```

`--run_name` turns on W&B logging and uploads a checkpoint at the end. `--sanity_check` trains on one
batch. `--path_to_checkpoint` evaluates a checkpoint before training.

Re-score a checkpoint from W&B:

```bash
poetry run python -m pose3d.evaluate --artifact <entity/project/name.pth:vN> --path_to_datasets ...
```

## GATr denoiser

`--vector_field gatr` swaps the Clifford MLP vector field of `clifford_flow` for the Geometric
Algebra Transformer ([reference](https://github.com/Qualcomm-AI-research/geometric-algebra-transformer)).
The rotor, the time and the `n_cond_mv` condition multivectors are embedded in Cl(3,0,1) and become
the tokens of one sequence; the velocity is read from the rotor token's rotation bivector. The
condition head stays a Clifford MLP unless `--condition_head gatr`, which runs GATr over the 256
backbone tokens plus `n_cond_mv` learned query tokens (each token also gets a learned scalar
embedding, since GATr treats tokens as an unordered set) and reads the condition multivectors from
the queries. Both share the sizes below. Size them with `--gatr_blocks`, `--gatr_mv_channels`,
`--gatr_s_channels` and `--gatr_heads`. The default is still `--vector_field clifford`.

```bash
pip install --no-deps einops opt_einsum \
  git+https://github.com/Qualcomm-AI-research/geometric-algebra-transformer.git
poetry run python -m pose3d --path_to_datasets ... --vector_field gatr
```

Install GATr with `--no-deps`: its `setup.py` pins `numpy<1.25` and `xformers`, which would replace
the preinstalled torch. `xformers` is only needed for attention masks, which the flow never passes,
so a stub stands in when it is missing.

## Micro and macro metrics

Every reported number has two versions. Micro is pooled over all validation images, so the big
classes dominate (car, chair). Macro averages over the 12 Pascal3D+ classes, so each counts the
same: `class_mean_median_error` (the mean of the per-class medians, the number the IPDF /
Image2Sphere tables report) every epoch, and at the end also `final_class_mean_acc@15` /
`@30` plus each class's median and accuracies (`final_median_error_class<c>`,
`final_acc@15_class<c>`, ...), printed as a table.

The pre-built RAM cache holds no class labels, so they are read from the annotations of the
mounted Pascal3D+ (no image is decoded) and checked against the cache's ground-truth rotations. If
Pascal3D+ is not mounted, or the check fails, macro metrics are skipped with a message and the
run goes on. Other loaders pass the labels through as before.

## Multi-GPU

Training uses every visible GPU by default (torch DistributedDataParallel, `--ddp`). The run relaunches
itself under `torchrun` with one process per GPU, so the command is the same as for one GPU; with one GPU
or CPU it changes nothing. `--num_gpus N` limits the count and `--no-ddp` turns it off.

```bash
poetry run python -m pose3d --path_to_datasets ... --no-ddp
```

`--batch_size` is the **global** batch: each GPU gets `batch_size // n_gpus`, so the recipe is the
same on 2 T4 or 4 L4. Raising the global batch on purpose (`--batch_size 128`) is a different recipe;
`--lr_scaling linear|sqrt` rescales the learning rate by `batch_size / lr_reference_batch`. Other
options: `--sync_bn`, `--nccl_p2p` (off by default, Kaggle's T4 x2 can hang with it), `--seed`
(each rank adds its rank). With the RAM cache the tensors are built once before the ranks start.
W&B, the checkpoint and the printed log come from rank 0 only.

## Pascal3D tensor cache

`--ram_memory` (on by default) decodes every image once per run (~34 min on Kaggle). The tensors can be
saved and reloaded instead: `--ram_cache_save_dir DIR` writes `pascal_train.pt` / `pascal_val.pt` after a
normal build, `--ram_cache_dir DIR` loads them (Pascal3D is not even constructed). By default
(`--pre_cache`) the run looks for the Kaggle dataset `syfry5suvzovvakmuj/pascal3d-ram-cache` and uses it
when it is mounted; `--no-pre_cache` disables that. The cache holds one un-augmented pass, so it is
skipped with `--use_warp`, `--use_synth`, `--raw_cache` or `--fisher_prior`.

## Kaggle

`notebooks/clifford-runner.ipynb` is the runner. Cell 0 holds `BRANCH`, `RUN_NAME` and the flags that
differ from the defaults; the other cells clone the branch, install what Kaggle lacks (no Poetry) and run
`python -m pose3d`. Attach the datasets `syfry5suvzovvakmuj/pascal3d` and
`syfry5suvzovvakmuj/pascal3d-ram-cache`.

### Offline (RTX / L4, no internet)

The premium accelerators run without internet, so nothing can be cloned or pip-installed.
`notebooks/clifford-runner-offline.ipynb` is the runner for them: cell 0 holds `RUN_NAME`,
`BEST_SETUP` (the warp + synth-pack + EMA recipe of the best run) and `EXTRA`; the other cells
copy the code from a **repo snapshot dataset**, install the wheels of
`syfry5suvzovvakmuj/ga-research-offline-deps`, set `WANDB_MODE=offline` and run `python -m pose3d`.
Attach `pascal3d`, `pascal3d-ram-cache`, `pascal3d-synth-pack`, `ga-research-offline-deps` and
exactly one snapshot. Build the snapshot from the branch and push the notebook with the pool
launcher (`gpu_pool/README.md`):

```bash
git archive --format=tar.gz --prefix=GA-Research/ -o /tmp/snap/ga-research-<branch>.tar.gz HEAD
kaggle datasets init -p /tmp/snap    # then set title/id in dataset-metadata.json
kaggle datasets create -p /tmp/snap --dir-mode skip
python gpu_pool/launcher.py --tokens-dir ~/.kaggle-accounts --title <kernel> \
  --notebook pose3d/notebooks/clifford-runner-offline.ipynb --accelerator rtx6000 \
  --competition arc-prize-2026-arc-agi-3 --no-internet --dataset syfry5suvzovvakmuj/pascal3d \
  --dataset syfry5suvzovvakmuj/pascal3d-ram-cache --dataset syfry5suvzovvakmuj/pascal3d-synth-pack \
  --dataset syfry5suvzovvakmuj/ga-research-offline-deps --dataset <owner>/<snapshot>
```

Metrics come out of the kernel log (`kaggle kernels logs <owner/kernel>`, one `WANDB_SYNC` JSON
line per epoch and a `WANDB_SYNC_FINAL` line at the end); `gpu_pool/rtx_monitor.py` can replay
them into W&B.

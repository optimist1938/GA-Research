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

Run from the `pose3d/` folder. The defaults are the reference recipe (8.95° on Pascal3D+, see
`pose3d/config.py`), with the speed options of the GATr section on. It needs the GATr package (below) and trains on Image2Sphere's data, read from
the Kaggle dataset `syfry5suvzovvakmuj/pascal3d-synth-pack` when it is mounted:

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

The vector field of `clifford_flow` is the Geometric Algebra Transformer by default
(`--vector_field gatr`, [reference](https://github.com/Qualcomm-AI-research/geometric-algebra-transformer));
`--vector_field clifford` brings back the Clifford MLP.
The rotor, the time and the `n_cond_mv` condition multivectors are embedded in Cl(3,0,1) and become
the tokens of one sequence; the velocity is read from the rotor token's rotation bivector. The
condition head stays a Clifford MLP unless `--condition_head gatr`, which runs GATr over the 256
backbone tokens plus `n_cond_mv` learned query tokens (each token also gets a learned scalar
embedding, since GATr treats tokens as an unordered set) and reads the condition multivectors from
the queries. Both share the sizes below. Size them with `--gatr_blocks`, `--gatr_mv_channels`,
`--gatr_s_channels` and `--gatr_heads`.

```bash
pip install --no-deps einops opt_einsum \
  git+https://github.com/Qualcomm-AI-research/geometric-algebra-transformer.git
poetry run python -m pose3d --path_to_datasets ...                             # GATr (default)
poetry run python -m pose3d --path_to_datasets ... --vector_field clifford     # Clifford MLP
```

Install GATr with `--no-deps`: its `setup.py` pins `numpy<1.25` and `xformers`, which would replace
the preinstalled torch. `xformers` is only needed for attention masks, which the flow never passes,
so a stub stands in when it is missing.

Three speed options are on by default. The first two compute the same function with the same
weights, so checkpoints load either way:

- `--gatr_dense_linear` runs every GATr `EquiLinear` as one dense matmul. GATr's own einsum over the
  9 basis maps has a slow backward.
- `--gatr_compile` runs the GATr modules, the EMA copy included, through `torch.compile`. It uses
  compile-friendly versions of GATr's primitives from `models/gatr_fast_ops.py`: the products are
  unrolled from GATr's own tables, and attention is no longer excluded from compilation. The
  whole network traces as one graph. On a fresh machine, compilation takes about 5 minutes per
  run: the first training steps, validation and the 32-sample final evaluation. A warm inductor
  cache brings this down to well under a minute.

`--backbone_bf16` runs only the ResNet backbone under bf16 autocast with channels_last. GATr and the
CGENN head stay fp32. Unlike the two flags above, it changes the numerics; a full run of the recipe
with it scores the same as without (the second table). GPUs without native bf16 (before sm_80, e.g.
the T4) run the backbone in fp32 anyway.

`--no-gatr_dense_linear`, `--no-gatr_compile` and `--no-backbone_bf16` turn them off;
`--no-gatr_compile` skips the compilation for quick debugging.

Measured on one RTX Pro 6000 with the trainer's own epoch and evaluation functions, batch 64,
synthetic images. The evaluation columns are for 360 images.

| options on | train step | 1-sample evaluation | 32-sample evaluation |
|---|---|---|---|
| none | 176 ms | 1.11 s | 6.54 s |
| `gatr_dense_linear` | 160 ms | 1.14 s | 6.72 s |
| `gatr_dense_linear`, `gatr_compile` | 120 ms | 0.51 s | 3.48 s |
| all three (default) | 80 ms | 0.44 s | 3.41 s |

Full 100-epoch runs of the default recipe on the same GPU, final 32-sample evaluation of the EMA
weights:

| options on | epoch | run | median | class-mean |
|---|---|---|---|---|
| none (reference, unseeded) | 215 s | 6.1 h | 7.85 deg | 8.95 deg |
| `gatr_dense_linear`, `gatr_compile`; `--seed 1` | 148 s | 4.5 h | 7.83 deg | 9.20 deg |
| all three (default); `--seed 1` | 103 s | 3.2 h | 7.88 deg | 9.28 deg |

The two seeded runs agree within 0.1 deg. The class-mean gap to 8.95 is the same with and without
bf16, and the first of them computes the reference's function, so the gap is most likely the seed:
the class-mean is noisy (diningtable has 7 validation images).

Details are in `reports/flash_clifford/`.

```bash
poetry run python -m pose3d --path_to_datasets ... --no-backbone_bf16   # exact fp32, still compiled
poetry run python -m pose3d --path_to_datasets ... --no-gatr_dense_linear --no-gatr_compile --no-backbone_bf16  # as before
```

## Flow-trajectory visualizations

`pose3d/visualize_flow.py` replays the flow's Euler loop from several random rotations and draws every step: each rotation relative
to the ground truth as an axis-angle vector in the ball of radius pi (true pose at the origin, distance = error), the angle to the true
pose against flow time, a rotating-frame film strip, and an animation. Several checkpoints can be compared on the same images and noise.

```bash
poetry run python -m pose3d.visualize_flow --checkpoint a.pth --label GATr --checkpoint b.pth --label MLP \
    --val_cache pascal_val.pt --out viz/
```

During training, `--log_viz` (on in the runner notebook) sends the same to W&B for `--viz_images` fixed validation images: every
`--viz_every` epochs a predictions table (image, class, true and predicted rotation, medoid error, share of the noise samples that end
within 15 deg) and the paths figure, and after the last epoch a film strip and an animation. Flow models only, and it needs `--run_name`.
A run launched with `WANDB_MODE=offline` stores the media in its run folder; `wandb sync` uploads it.

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

## SYMSOL and the Image2Sphere baseline

`--dataset symsol` reads `symsol/{train,test}/rotations.npz + images/` under `--path_to_datasets`
(datasets/symsol.py). Every test image carries all of its equivalent poses, so the rotation error is to the
**nearest equivalent** (a cube has 24 right answers), and the final evaluation also reports the SYMSOL
log-likelihood. `--symsol_train_views 4500` is the 10 % protocol of the `symsol1_10pct_*` runs (4,500 of
45,000 views per shape); `--symsol_test_views` caps the test views per shape (default: all 5,000).

The published Image2Sphere model is `--model i2s_real` (models/i2s_real.py, upstream's code under this
harness). On SYMSOL it reports the same median rotation error as the flows plus upstream's grid
log-likelihood (engine/grid_model_likelihood.py: `log(softmax(grid)[nearest cell] * N / pi^2)`, no cap). Its
evaluation grid is built lazily on the GPU, so the paper's level-5 grid (2.36M rotations, a 4.3 GB Wigner
matrix) is usable: `--i2s_eval_rec_level 5`. The paper's recipe on SYMSOL I with 10 % of the data:

```bash
poetry run python -m pose3d --path_to_datasets ... --run_name i2s_symsol1_10pct_sgd \
  --model i2s_real --encoder resnet50 --dataset symsol --symsol_set 1 --symsol_train_views 4500 \
  --no-use_warp --no-use_synth --no-ema --no-fixed_val_noise --batch_size 64 --n_epochs 40 \
  --optimizer sgd --lr 1e-3 --lr_scaling none --lr_schedule step --lr_step_epochs 15 --warmup_epochs 0 --grad_clip 0 \
  --i2s_eval_rec_level 5
```

`notebooks/i2s-symsol-runner.ipynb` runs it on Kaggle (one process, `--no-ddp`: two DDP ranks each holding the 7 GB of decoded images exceed a T4 x2 kernel's RAM) (and the same model under the AdamW / cosine schedule
of the `symsol1_10pct_*_i2s_adamw` flow runs), online from a clone of the branch or offline from a repo
snapshot; attach `syfry5suvzovvakmuj/symsol1-10pct-fulltest`. For reference, Image2Sphere's Table 3 gives
log-likelihoods only: 2.98 on SYMSOL I with 10k views per shape, 3.41 with the full public set (50k); it
reports no rotation error on SYMSOL.

## Two-tower model: the STF tensor track

`--stf_track` (off by default) adds a second feature track of symmetric traceless Cartesian tensors of ranks
2..`--stf_lmax` (default 3, up to 6), `--stf_channels` (8) per rank, to `clifford_flow` with the GATr vector
field. They carry the SO(3) degrees l >= 2 that multivectors cannot, are built from the grade-1 parts of the
condition multivectors and reach the vector field as rotor-dependent invariant scalars (`--stf_inject gram|frame`).
Design, tests, the SYMSOL protocol, run commands and the time estimates: `reports/symsol_stf/README.md`; runner:
`notebooks/symsol-stf-runner.ipynb` (offline mode for the RTX kernels; the repo snapshot comes from
`scripts/make_snapshot.py`).

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

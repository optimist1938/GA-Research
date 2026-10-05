# Fast, exact inference for Clifford Flow (`predict` fast-path)

Date: 2026-10-05. Branch: `flow-x1-prediction`. Status: approved design, to implement after the
x1 training run finishes.

## 1. Goal and non-goals

`CliffordFlow.predict()` integrates the vector field for `steps=20` Euler steps over
`n_samples=32` draws per image, then takes the geodesic medoid. On CPU (ResNet-101, heads 88,
`n_cond_mv=64`, batch 8 images) the 20 field steps take 52% of `predict()`, the backbone 42%.
Inside one field step (49 ms for 256 rows) the first `FullyConnectedSteerableGeometricProductLayer`
(FCGP, 66 -> 88 channels) costs 27.5 ms: 6 ms rebuilding its expanded weight tensor
`(88, 66, 8, 8, 8)` on every call and 11.5 ms for a dense einsum over the Cl(3,0) Cayley table,
of whose 512 entries only 64 are non-zero.

Goal: make `predict()` faster while producing the same rotations as today, up to float round-off.
The fast-path is a pure re-arrangement of the same arithmetic; it introduces no new
hyper-parameter and changes nothing in training.

Non-goals (separate work): fewer Euler steps or Heun (changes the result; evaluate once the x1
run is scored), bf16 / channels_last for the backbone, CUDA graphs / `torch.compile`, speeding
up training, changes to the `clifford` library (installed on Kaggle from an offline wheel that
would have to be rebuilt), the GATr heads (no exact per-image precompute exists for full
self-attention; see the idea board row "GATr with a condition mask and a KV cache").

## 2. Architecture

New module `pose3d/pose3d/models/fast_flow.py`:

```python
class FastCliffordField:
    """Evaluates a TralaleroTralala vector field with the condition hoisted out of the step loop."""
    def __init__(self, algebra, vector_field: TralaleroTralala): ...
    def prepare(self, cond_mv) -> Context:        # once per batch of images, (B, n_cond, 8)
    def step(self, ctx, rotor, t) -> Tensor:      # (B*K, 4), (B*K,) -> grade-2 output (B*K, 3)
```

* Built from the trained layers' parameters at the start of every `predict()` call. Construction
  gathers the path weights once per call (cheap: index operations on `(m, n, 20)` parameters);
  nothing is cached across calls, so EMA updates or further training can never leave a stale
  kernel behind.
* `CliffordFlow.predict(..., fast: bool = True)` uses it when `self.vector_field` is a
  `TralaleroTralala`; `mlp_heads` and GATr fields keep the generic `velocity()` path. The
  `flow_param="x1"` division by `(1 - t)` is applied to `step()`'s output exactly as `velocity()`
  does, so both parametrisations are covered.
* `velocity()`, `compute_loss()`, the trainer and the checkpoints are untouched.
* `pose3d.evaluate` gets `--no-fast` so one checkpoint can be timed both ways.

## 3. What is re-implemented and what is reused

Only two kinds of operations are re-implemented: the weighted geometric products of FCGP /
`SteerableGeometricProductLayer` and the channel mixing of `MVLinear`. The element-wise modules
`MVSiLU` and `NormalizationLayer` are called as they are (their `eps` values and the smooth
absolute value stay the library's), which keeps the fast-path's correctness argument confined to
linear algebra.

Notation: a layer input is `(b, n, 8)`; `W[m, n, i, j, k] = cayley[i, j, k] * w[m, n, g(i), g(j), g(k)]`
with `w` the `(m, n, 20)` path parameter expanded over the 64 grade triples (20 of them non-zero
for Cl(3,0)); `x_n ⊗_W r_n` denotes `Σ_{i,k} x[n,i] W[m,n,i,j,k] r[n,k]`.

FCGP forward, as in the library:

```
r   = Normalization(linear_right(x))                      # MVLinear without bias, then per-grade scaling
out = (linear_left(x) + Σ_n x_n ⊗_W r_n) / sqrt(2)        # linear_left: MVLinear with scalar bias
```

### 3.1 Sparse Cayley product (generic kernel)

`cayley != 0` at 64 index triples `(I, J, K)`; every pair `(i, k)` lands on exactly one `j`, and
every `j` collects exactly 8 pairs. For a layer with channel mixing:

```
Wp[m, n, p] = w[m, n, path(I_p, J_p, K_p)] * cayley[I_p, J_p, K_p]      # (m, n, 64), built once per predict
P[b, n, p]  = x[b, n, I_p] * r[b, n, K_p]                                 # (b, n, 64)
out[b, m, j] = Σ_{p : J_p = j} Σ_n P[b, n, p] Wp[m, n, p]
             = ( P[:, :, G_j].reshape(b, n*8) @ Wp[:, :, G_j].reshape(m, n*8).T )   for j in 0..7
```

Eight matmuls of `(b, 8n) x (8n, m)`: `8 * 8 * n * m` MACs per row instead of `512 * n * m`.
Measured on CPU for the 66 -> 88 layer at 256 rows: 11.2 ms -> 5.2 ms, max deviation from a
float64 reference 2e-6 (the dense einsum itself deviates 1e-6). The per-channel variant
(`SteerableGeometricProductLayer`, weight `(n, 20)`, no sum over `n`) is the same with
`Wp[n, p]` and an element-wise multiply plus a sum over `p` within each `G_j`.

Used for: the output FCGP (88 -> 1), the hidden `SteerableGeometricProductLayer`, the rotor and
time channels of the first FCGP, and every FCGP of any further hidden block.

### 3.2 Hoisting the condition out of the first FCGP

Input channels of the vector field: `n = 0` the rotor `R_t`, `n = 1` the time `t` (scalar blade),
`n >= 2` the `n_cond` condition multivectors `c`, constant across the 20 steps and the K samples
of one image. Per step only `R` and `t` change, but they reach every channel of the right operand
through `linear_right`, and the normalization is non-linear, so `r_n` is not separable. The left
operand of the condition channels is, and the product is bilinear:

```
Σ_{i,k} c[n,i] W[m,n,i,j,k] r[n,k]  =  Σ_k r[n,k] · G[m,n,j,k],      G[m,n,j,k] = Σ_i c[n,i] W[m,n,i,j,k]
```

`G` depends on the image only. Per image: `G` is `(m, n_cond, 8, 8)` (88 x 64 x 64 = 360k floats,
1.4 MB); since each `(j, k)` has exactly one `i`, it is filled from the 64 sparse entries, not by a
dense contraction.

`prepare(cond_mv)` computes, per image:

```
Lc[m, :]   = Σ_{n>=2} Wl[m, n, g] c[n, :] + bias_m         # linear_left's condition part, (B, m, 8)
rc[n, :]   = Σ_{k>=2} Wr[n, k, g] c[k, :]                  # linear_right's condition part, (B, n_in, 8)
Gmat       = G.permute(n, k, m, j).reshape(B, n_cond*8, m*8)
```

`step(ctx, rotor, t)`, with rows `(B*K)` laid out as `i*K + j -> image i` (the same interleaving
`predict` already uses), viewed as `(B, K, ...)`:

```
s      = rc[image] + Wr[:, 0, g] ⊙ R + Wr[:, 1, g] ⊙ T           # (B*K, n_in, 8): two outer products
r      = Normalization(s)                                        # the library module, unchanged
second = bmm(r[:, 2:, :].view(B, K, n_cond*8), Gmat).view(B*K, m, 8)
       + R ⊗_W[:,0] r_0 + T ⊗_W[:,1] r_1                          # sparse kernel, 2 channels
out    = (Lc[image] + Wl[:, 0, g] ⊙ R + Wl[:, 1, g] ⊙ T + second) / sqrt(2)
```

Per row and step the second-order term costs `n_cond*8 * m*8 = 360k` MACs instead of
`n_in * 512 * m = 3M`, there is no weight rebuild, and `cond_mv` is never `repeat_interleave`d.

### 3.3 Remaining layers

`act1`, `act2`: the model's own `MVSiLU` modules. `gp`: 3.1 per-channel kernel, with its own
`linear_left` / `linear_right` / normalization reproduced as in 3.2 but without hoisting (nothing
is constant after the first block). `out`: 3.1 kernel, 88 -> 1. Several hidden blocks
(`hidden_dim=[a, b, ...]`) are supported: hoisting applies to the first FCGP only, later FCGPs use
3.1 with channel mixing.

Expected cost per row and step: about 0.5M MACs against about 3.1M today, with no
`(m, n, 8, 8, 8)` tensors materialised.

## 4. Data flow in `predict`

```
cond_mv, fisher_a = self._features(x, cls)                 # (B, n_cond, 8), once
field = FastCliffordField(self.algebra, self.vector_field) # gathers weights, once per call
ctx   = field.prepare(cond_mv)                             # G, Lc, rc per image
rotor = r0 for B*K rows (uniform or Fisher, as today; no change to the random draws)
for i in range(steps):
    t   = full((B*K,), i*dt)
    out = field.step(ctx, rotor, t)                        # (B*K, 3)
    v   = out / (1 - t).clamp(min=_X1_EPS)[:, None] if flow_param == "x1" else out
    rotor = rotor_multiply(rotor, exp_map(dt * v))
medoid / rotor_to_matrix as today
```

Constraints checked at construction (`ValueError`): the algebra is Cl(3,0) (`dim == 3`,
`n_subspaces == 4`); every block of the field is the `fc / act1 / gp / act2` layout of
`TralaleroTralala`; `include_first_order` is on (the library default) in every product layer.
Device and dtype follow the model's parameters. Memory: `G` is 1.4 MB per image; the 64-image
evaluation batch holds about 92 MB.

## 5. Tests (`pose3d/tests/test_fast_flow.py`, CPU, seconds)

1. Sparse Cayley product equals the dense einsum of the library layer on random inputs
   (channel-mixing and per-channel variants), `atol=1e-5` on inputs of unit scale.
2. `FastCliffordField.step(prepare(cond), rotor, t)` equals `model._field(rotor, t, cond)` for a
   model with random (non-zero) weights, for `hidden_dim=[88]` and `hidden_dim=[8, 8]`, with
   `K > 1` samples per image, relative tolerance 1e-4.
3. `predict(img, n_samples=4, steps=5, fast=True)` equals `predict(..., fast=False)` under the
   same `torch.manual_seed`, for `flow_param` in `("velocity", "x1")`; the medoid choice is
   included in the comparison.
4. `mlp_heads=True` and a non-Cl(3,0) algebra fall back to / reject the fast-path as specified.

`pose3d/tests/bench_predict.py` (not collected by pytest, run by hand) times the backbone, one
field step both ways and `predict()` both ways, the script behind the numbers in section 1.

## 6. Success criteria

* All tests in section 5 pass.
* CPU benchmark at B=8, K=32, 20 steps: a field step at least 4x faster than the 49 ms of today,
  `predict()` at least 2x faster than 1.88 s.
* RTX: `python -m pose3d.evaluate` on the `clifford_flow_x1_warp_synth_ema_b64` checkpoint with
  and without `--no-fast`, same metrics to 3 decimals, wall time reported in the results section
  below once measured.

## 7. Results

To be filled in after implementation: CPU and RTX timings, metric equality on the real checkpoint.

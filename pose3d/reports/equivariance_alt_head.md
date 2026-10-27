# An image2sphere-style equivariant head for the SO(3) flow: design, equivariance proof, prototype

Branch `so2-cond-head` @ dc62901, October 2026. Design + numerical prototype only; nothing trained, no repo
code changed. Scratch scripts in `/tmp/eq_research_4/` (`so3fourier_head.py`, `check_equivariance.py`,
`bench_cpu.py`; each < 10 s CPU). The prototype module is reproduced in Appendix A because `/tmp` is
not permanent. **[verified]** = run on this machine; **[reasoned]** = derivation / literature.

---

## 0. Summary

* **Chosen design ("SO(3)-Fourier velocity head", `--vector_field so3fourier --cond_tokens sphere`).**
  Keep I2S's image→sphere→SO(3) pipeline as the *condition* (computed once per image): 1×1 conv on the
  7×7 ResNet map → D4-steerable orthographic lift onto an 88-point HEALPix hemisphere → ReLU → S² Fourier
  transform (ℓ ≤ 6) → S²→SO(3) convolution with learned Fourier filters → an exactly equivariant norm gate.
  The result is a tensor `c ∈ R^{C_f × 455}` of Wigner coefficients of `C_f` signals on SO(3). The velocity
  at the pose hypothesis `R_t` is read from the signal *expressed around R_t*: `c̃_ℓ = D^ℓ(R_t)ᵀ c_ℓ`, then a
  small time-conditioned MLP → body-frame velocity bivector. No tokens, no attention, no grid.
* **Equivariance (exact, proved and checked to 1e-15):** the read-out is invariant under the *left* action
  `(R, c) → (G R, D(G) c)` for **every** `G ∈ SO(3)`; end to end on the backbone map the field satisfies
  `v(R_z(−90°) R, rot90(map)) = v(R, map)` and `v(F R F, flip(map)) = (−,−,+)·v(R, map)` exactly; continuous
  roll is exact in the continuum and approximate only through the 7×7 resampling and the backbone — the same
  split as I2S / Howell (in-plane SO(2) on the image, SO(3) on the sphere/SO(3) side).
* **Why it is the right equivariant family here.** `c̃ = D(R)ᵀ c` is a *complete* invariant of the pair
  `(R, c)`: every equivariant velocity field built from an SO(3) signal, including the gradient of the I2S
  density, an SO(3)-convolution evaluated at `R`, an e3nn tensor-product network over irreps, or an energy
  gradient, is a function of `c̃` (§3). So the design subsumes options (a)–(c) of the brief at the cost of
  one 455×455-block matmul per channel. Unlike the Cl(3,0) token heads it carries angular frequencies up to
  ℓ = 6 (the SO(2)/GATr heads are truncated at ℓ ≤ 1 per token, `geometric_redesign.md` §2.2).
* **Sizes (counted):** small 246,499 params (vs DiT-small 0.249M, P1-small 0.232M), large 1,116,835
  (vs GATr 1.13M, DiT 1.19M). **Compute:** the velocity read-out is ~6× cheaper than the reference GATr
  vector field and ~2.5× cheaper than the DiT vector field at B = 512 on CPU; the condition path is 16 ms
  per 64 images on CPU. Training is backbone-bound: expect epoch times at or below the fast end of the current
  175–230 s range. **[verified on CPU; GPU ratios reasoned]**
* **First experiment:** small (0.25M) `so3fourier` vs its *non-equivariant twin* (same condition path,
  read-out on `[R, t, c]` without canonicalisation, equal params) vs the existing DiT-small, same recipe
  as P1-small, then 3 seeds for the winner. Decision rule in §6.

---

## 1. image2sphere end to end (Klee et al., ICLR 2023)

Installed package `image2sphere` (`.../site-packages/image2sphere/{models,predictor,so3_utils}.py`),
default Pascal3D+ configuration (`I2SReal` in `pose3d/models/i2s_real.py`: ResNet-101, `sphere_fdim=512`,
`lmax=6`, train grid rec 3, eval grid rec 5). The pipeline:

1. **Encoder.** ResNet-101 without pool/fc → `(B, 2048, 7, 7)`.
2. **Orthographic lift** (`SpatialS2Projector`). 1×1 conv 2048 → 512. A HEALPix grid on the hemisphere
   (`s2_healpix_grid(max_beta=90°, rec_level=2)`, nside 4; the pole is e3nn's `y` axis) is projected
   orthographically to the unit disc, `P(x) = (x, y)`, scaled by `coverage = 0.9`, and each sphere point
   takes a Gaussian-weighted (`σ = 0.2` in normalised coordinates) average of the 7×7 cells, tapered to zero
   between β = 75° and 90°. During training only 20 random points of the grid are used (dropout of the
   grid). ReLU on the grid. Then `Ψ_ℓm = Σ_i Y_ℓm(x_i) Ψ(x_i) / √P` with real spherical harmonics
   (`normalization='component'`), i.e. an S² signal with 512 *scalar* channels, ℓ ≤ 6 → `(B, 512, 49)`.
   **Note [verified by reading the code]:** the (7×7 → P) weights are a free `nn.Parameter` initialised to the
   Gaussian kernel, so I2S's projection is SO(2)-steerable only at initialisation; after training it is an
   arbitrary linear map from pixels to sphere points.
3. **S² → SO(3) convolution** (`o3.Linear(s2_irreps, so3_irreps, f_in=512, f_out=8, internal_weights=False)`
   with weights `HarmonicS2Features`, i.e. a learned filter `ψ` parametrised by its spherical-harmonic
   coefficients). By the convolution theorem the output per ℓ is the outer product
   `c_ℓ[y, m, n] = Σ_x Ψ_ℓ[x, m] ψ_ℓ[x, y, n]`: a `(2ℓ+1)×(2ℓ+1)` block per channel; the "left" index `m`
   comes from the image, the "right" index `n` from the template. Output `(B, 8, 455)`, 455 = Σ_{ℓ≤6}(2ℓ+1)².
   Template-matching reading: `f_y(g) = Σ_ℓ ⟨c_ℓ, D^ℓ(g)⟩` is large when the template rotated by `g`
   matches the hemisphere signal.
4. **Non-linearity**: `e3nn.nn.SO3Activation(6, 6, relu, 10)` — to an SO(3) grid, ReLU, back (approximately
   equivariant: **4e-4 relative error** under a random rotation in our test, §5).
5. **SO(3) convolution** with a *local* filter (`so3_near_identity_grid`, rotations ≤ 22.5°, 168 points) →
   `(B, 1, 455)`.
6. **Output / loss.** The scalar signal is evaluated on an SO(3) HEALPix grid (rec 3: 36,864 points, 7.5°)
   by one matmul with the precomputed Wigner-D matrix `flat_wigner` (455 × 36,864), softmax + cross-entropy to
   the nearest grid rotation of the label. Evaluation: rec 5 (2.36M points, 1.875°), argmax (optional gradient
   ascent). Pose convention: `nearest_rotmat(rot, rotmats)` with `rot` the loader's object→camera label — the
   same `R` our flow integrates.

**Numbers (Pascal3D+, class-mean of per-class median error, ResNet-101, 6 seeds):** I2S **9.8 ± 0.4°**;
per class (as tabulated by Howell et al.): plane 9.2, bike 12.7, boat 21.7, bottle 7.4, bus 3.3, car 4.9,
chair 9.5, table 9.3, mbike 11.5, sofa 10.5, train 7.2, tv 10.6. Ablations (ModelNet10-SO(3), limited views):
spatial projection ≫ Fourier projection (46.3 vs 57.0 median) and both ≫ "MLP → Fourier coefficients" (69.2);
L = 6 is best, L > 6 hurts; the extra SO(3) conv is worth little (Table 8).

## 2. What Howell et al. (NeurIPS 2023) change

* **Constraint.** An out-of-plane rotation of the camera changes the image non-geometrically (occlusion, new
  surfaces), so the only symmetry that can be imposed *exactly* on an image→3D map is the in-plane one:
  SO(2) rotations of the image plane acting on the output by the corresponding rotation about the optical
  axis. **Theorem 1** (their notation): the most general linear SO(2)-equivariant map from an image feature
  field `f: R² → V` (SO(2)-rep ρ) to a spherical signal is `[Φf](n̂) = ∫ κ(n̂, r) f(r) dr` with
  `κ(n̂, r) = Σ_ℓ F_ℓ(r)ᵀ Y_ℓ(n̂)`, where each `F_ℓ` is an ordinary SO(2)-steerable kernel from `ρ` to
  `ρ ⊗ Res^{SO(3)}_{SO(2)} D^ℓ`. Equivalently, by Frobenius reciprocity
  `Hom_{SO(2)}(ρ, Res σ) ≅ Hom_{SO(3)}(Ind ρ, σ)`: the output is a signal in the *induced* representation
  (a field on S² = SO(3)/SO(2)), and the layer is the "induction layer". The restricted representation
  `Res D^ℓ` contains the SO(2) frequencies `m = −ℓ..ℓ` exactly once each, so an image feature of in-plane
  frequency `|m|` can only feed harmonics with `ℓ ≥ |m|` (their Figure 2: not every SO(2) rep is a
  restriction of an SO(3) rep).
* **Which layers change.** Only the lift. I2S's orthographic projection is the special case
  `F_ℓ(r) ∝ Y_ℓ(P⁻¹ r)` (scalar ρ, fixed kernel); the icosahedral projections of Klee et al. (2022) and
  Esteves et al. (2019) are the other two special cases. Howell replace it by *learned* steerable kernels
  built with `e2cnn`, ℓ ≤ 6, producing a 64-channel S² signal with trivial fibres; the rest (Fourier S²-conv
  → 8-channel SO(3) signal, grid ReLU, local SO(3) conv, HEALPix softmax, rec 3/5 grids, batch 64, 40 epochs,
  SGD + StepLR) is I2S.
* **Numbers.** Pascal3D+ class-mean median **9.2°** (ResNet-101; per class plane 9.3, bike 12.6, boat 17.0,
  bottle 8.0, bus 3.0, car 4.5, chair 9.4, table 6.7, mbike 11.9, sofa 12.1, train 6.9, tv 9.9), 10.2° with
  ResNet-50; Yin et al. 9.4, I2S 9.8, IPDF 10.3. Appendix E: inducing to S² then convolving (9.2) beats
  inducing directly to SO(3) (9.7; ResNet-50: 10.2 vs 10.5). ModelNet10-SO(3): 17.8° vs I2S 16.3° (they
  slightly under-perform I2S there). Our current numbers (8.61–9.35) are already below both.

**Consequence for us.** Our velocity field must satisfy, for in-plane rotations `G = R_z(φ)` of the image,
`v(G R, t, G·image) = v(R, t, image)` (body velocity invariant; `_frame_velocity` docstring). The
Howell/I2S split applies verbatim: the image→S² part can only be SO(2)-exact (C4 on the 7×7 grid), the
S²→SO(3)→velocity part can and should be SO(3)-exact, so that the *structure* "rotating the hemisphere
signal rotates the velocity field" holds for every rotation, and the data only has to teach the backbone.

---

## 3. From a density on a grid to a velocity field

Setting: `c = {c_ℓ ∈ R^{C_f × (2ℓ+1) × (2ℓ+1)}}` from the condition path, transforming as `c_ℓ → D^ℓ(G) c_ℓ`
when the camera rolls by `G` (left action; **[verified]** for `G = R_z(−90°)` on `rot90(map)`, §5).
Pose `R_t` (rotor → matrix). Required: `v(G R, t, D(G) c) = v(R, t, c)`.

**The key identity.** Define the canonicalised coefficients `c̃_ℓ = D^ℓ(R)ᵀ c_ℓ`. Then
`c̃` is invariant: `D(GR)ᵀ D(G) c = D(R)ᵀ D(G)ᵀ D(G) c = c̃`. Conversely, any invariant `h(R, c)` equals
`h(I, D(R)ᵀ c) = h(I, c̃)` (take `G = R⁻¹`), so **the invariant functions of `(R, c)` are exactly the
functions of `c̃`**. Geometrically `c̃` is the Fourier transform of the signal seen from the hypothesis:
`f(R exp ξ) = Σ_ℓ ⟨c_ℓ, D^ℓ(R) D^ℓ(exp ξ)⟩ = Σ_ℓ ⟨c̃_ℓ, D^ℓ(exp ξ)⟩`, i.e. "the features of `f` in a
neighbourhood of `R`, mapped to the tangent space" of option (a), without any neighbourhood sampling.
**[reasoned; invariance verified numerically for random G ∈ SO(3)]**

| option | definition | equivariance | group | cost per Euler step (per sample, after `D(R)`) |
|---|---|---|---|---|
| (a) gradient of the I2S density | `v_i = ∂_ε f(R exp εX_i) = Σ_ℓ ⟨c̃_ℓ, X^ℓ_i⟩` | exact, body velocity invariant | all SO(3) on the Fourier side; SO(2)/C4 end to end | canonicalise `Σ_ℓ (2ℓ+1)³ ≈ 4.8k` MACs/channel + 3 dot products. Too rigid: a *linear* functional of `c̃`, no time dependence, and a conservative field — the flow-matching marginal `u_t(x) = E[log_x(x₁)/(1−t) \| x_t = x]` is not a gradient field in general |
| (a′) local patch of `f` around `R` | `f(R exp ξ_k)`, `k` on the 168-point local grid, then MLP | exact (same reason) | same | fixed linear map `455 → 168` on `c̃` then MLP: a special case of (d) with a frozen first layer |
| (b) e3nn tensor products of irreps with the pose | contract `c` (2ℓ+1 copies of ℓ) with `Y_ℓ(R e_i)` / `D(R)` rows via Clebsch–Gordan paths to scalars, MLP on the scalars | exact | same | ≥ one CG path per (ℓ₁, ℓ₂, ℓ_out) triple, O(L³) per channel pair; strictly contained in (d) since its scalars are functions of `c̃`, at higher cost |
| (c) energy / score | `v = −∇_R E(R, t, c)`, `E = MLP(c̃, t)`; `∂c̃/∂ξ_i = −X_i c̃`, so `v_i = ⟨∂E/∂c̃, −X_i c̃⟩` (3 JVPs) | exact | same | ≈ 2–3× (d). Conservative field by construction; flow-matching targets are not conservative; and the I2S density proper needs the 36k-grid CE loss. Useful only as an *auxiliary* loss (channel 0 trained as the I2S density with CE) or as a learned source distribution (as `fisher_prior`) |
| **(d) chosen: canonicalise + MLP** | `v = MLP_t(flatten c̃)`; first layer `Linear(C_f·455 → h)` | exact | same | canonicalise `C_f·4.8k` MACs + `C_f·455·h` MACs (+ 2× for the Z2 average) |

**Two facts that make (d) the natural I2S continuation.** (i) The first linear layer *is* an SO(3)
convolution evaluated at `R`: with a Fourier filter `ψ[x, k, ℓ]` (I2S's `SO3Convolution`), the convolution
theorem gives `(f_x ⋆ ψ_k)(R) = Σ_ℓ ⟨ψᵀ_{x,k,ℓ}, c̃_{x,ℓ}⟩`, a linear functional of `c̃`; so `h` output
units = `h` learned SO(3) correlation filters, sampled only at the current pose instead of on a 36k grid.
(ii) I2S's own prediction `argmax_g f(g)` is the fixed point of `v ∝ ∇f`, option (a) — the flow replaces
grid search + gradient ascent with 20 Euler steps of a *learned* ascent direction, trained on the geodesic
velocity target rather than on a cross-entropy over a 7.5° grid.

**Which group, honestly.** On the Fourier side everything in (a)–(d) is SO(3)-exact. The image only
supports camera roll, so end to end the exact symmetry is the D4 of the 7×7 grid (C4 roll + left-right
flip), identical to P1; continuous roll is exact in the continuum (Gaussian kernel, HEALPix quadrature on a
D4-symmetric grid) and approximate through resampling and the non-equivariant backbone. The flip needs a
Z2 Reynolds average in the read-out (§4.2), because the learned S² template `ψ` is not mirror-symmetric;
without it the roll stays exact and the flip is broken (negative control, §5).

---

## 4. The design

### 4.1 Modules and shapes (prototype: Appendix A; `pose3d/models/so3fourier_head.py` when implemented)

**Condition path, `SphereConditionHead` (once per image):**

| step | module | shape | params (small / large) |
|---|---|---|---|
| per-cell scalars | `Conv2d(2048, C_s, 1)` | `(B, C_s, 7, 7)` | 65,568 / 131,136 |
| D4-steerable lift | `K ∈ R^{49 × P}`, `K[p, n] = a(|p|)·exp(−|p − ρ q_n|²/2σ²)·taper(β_n)`, column-normalised; `q_n` = orthographic image position of hemisphere point `n`; learned `σ`, `ρ` (coverage, init 0.9), 10 per-radius gains `a` | `(B, C_s, P)`, P = 88 (HEALPix nside 4, z > 0) | 12 |
| grid ReLU | pointwise on the P points (exactly D4-equivariant: points permute) | | |
| S² Fourier transform | `Ψ = relu(·) @ Y / √P`, real `Y_ℓm` at the P points, 'component' normalisation, ℓ ≤ L = 6 | `(B, C_s, 49)` | |
| isotropic learned kernel (Howell, restricted to m = 0) | per-(channel, ℓ) gain | | `C_s·7` = 224 / 448 |
| S² → SO(3) conv | `c_ℓ[y,m,n] = Σ_x Ψ_ℓ[x,m] ψ_ℓ[x,y,n]` (I2S's `o3.Linear(s2_irreps → so3_irreps)` with external weights) | `(B, C_f, 455)` | `C_s·C_f·49` = 12,544 / 50,176 |
| exact non-linearity | norm gate per (channel, ℓ) block: `c_ℓ ← c_ℓ · sigmoid(a‖c_ℓ‖_F/(2ℓ+1) + b)`; `‖c_ℓ‖_F` is left- *and* right-invariant | | `2·C_f·7` = 112 / 224 |

Small: `C_s = 32, C_f = 8` → 78,460 params; large: `C_s = 64, C_f = 16` → 181,996. **[counted]**
Optional (not in the counts): I2S's local SO(3) conv (`C_f² · 168` params) after the gate; with the read-out's
full non-local first layer it is redundant unless followed by another gate (ablation only).

**Velocity read-out, `SO3FourierVelocity` (every flow step / time sample):**

| step | computation | shape | params (small / large) |
|---|---|---|---|
| pose | `R = rotor_to_matrix(rotor)`; `D^ℓ(R)` by `o3.matrix_to_angles` + `o3.wigner_D` (ℓ ≤ 6; 48 ms for 512 rotations on CPU, float64) | 7 blocks | |
| canonicalise | `c̃_ℓ = D^ℓ(R)ᵀ c_ℓ` (batched einsum per ℓ) | `(B, C_f·455)` | |
| SO(3)-correlation bank | `Linear(C_f·455 → h)` | `(B, h)` | 160,204 / 903,844 |
| time | sinusoidal `t` (64-d) → FiLM `(1+γ)·g + β` | | 5,720 / 16,120 |
| MLP | `SiLU → Linear(h, h) → SiLU → Linear(h, 3)`, last layer zero-initialised | `(B, 3)` = body bivector `(e12, e13, e23)` | 1,980+135 / 15,500+375 |
| flip (Z2 Reynolds) | `v = ½[m(c̃) + S·m(D(F) c̃)]`, `S = (−1, −1, +1)`, `D^ℓ(F) = (−1)^ℓ D^ℓ(R_x(π))` | | 0 (2× read-out compute) |

Small: `h = 44` → 168,039; large: `h = 124` → 934,839. **Totals 246,499 and 1,116,835.** **[counted]**
Output is already the body-frame velocity: no `_spatial_to_body`, and the Euler step
`r ← r exp(dt v)` is unchanged.

### 4.2 Why each piece is exact

* Lift: a 90° turn or a flip of the map permutes the 49 cells and, because the HEALPix hemisphere is
  D4-symmetric about `e3` (**[verified]**: every rotated/mirrored point lands on a grid point to 2e-8), permutes
  the 88 points; `K` depends only on `|p|`, `|p − ρq|`, `β`, which are preserved → `Ψ'(n̂) = Ψ(T⁻¹ n̂)` exactly;
  pointwise ReLU commutes with permutations; HEALPix is equal-area so the quadrature is a plain mean and
  `Ψ'_ℓ = D̂^ℓ(T) Ψ_ℓ` exactly, with `D̂` the O(3) representation of real harmonics (natural parity
  `(−1)^ℓ`; **[verified]** `Y_ℓ(F x) = (−1)^ℓ D^ℓ(R_x(π)) Y_ℓ(x)` to 1e-14).
* S²→SO(3) conv: `c'_ℓ = D̂(T) Ψ_ℓ ψ_ℓᵀ = D̂(T) c_ℓ` (left action, any `T`). Norm gate: `‖·‖_F` invariant.
* Read-out: `c̃' = D(GR)ᵀ D(G) c = c̃` for rotations. For the flip, `F R F = R_x(π) R R_x(π)` is proper, so
  `D(FRF) = D̂(F) D(R) D̂(F)` and `c̃' = D̂(F) D(R)ᵀ D̂(F) D̂(F) c = D̂(F) c̃` (since `D̂(F)² = I`). With
  `v(R,c) = ½[m(c̃) + S m(D̂(F)c̃)]`: `v(FRF, c') = ½[m(D̂(F)c̃) + S m(c̃)] = S v(R, c)`, and
  `S = (−,−,+)` on `(e12, e13, e23)` is the axial transformation `v' = −F v`, the same sign pattern
  `test_dircloud.py::test_velocity_flips_with_a_mirrored_image` asserts, so the integrated flow of the
  flipped image is `F R_t F`. The learned template `ψ` needs no symmetry constraint (an alternative with
  1× cost is to constrain `ψ_ℓ ∈ Fix(D̂^ℓ(F))`, mirror-symmetric templates, and build the MLP Z2-equivariant;
  the Reynolds average is simpler and the read-out is tiny).
* The time embedding is a scalar; FiLM preserves everything above.

### 4.3 Compute

CPU, float32, no grad, 4 threads, B = 512 (= 64 images × `n_time_samples` 8) **[verified, `bench_cpu.py`]**:

| vector field | params | forward per step |
|---|---|---|
| so3fourier small (with Z2) | 168k | 42 ms (≈ 40 ms of it is `wigner_D` for 512 rotations) |
| so3fourier large (with Z2 / without) | 935k | 48 / 44 ms |
| GATr reference `GATrVectorField(64, 8/32 × 4)` (66 tokens) | 98k | 282 ms |
| DiT `DiTVectorField(112 × 4)` (65 tokens) | 991k | 112 ms |
| condition path `SphereConditionHead(2048, 64, 16)` | 182k | 16 ms per 64-image batch |

Per (sample, step) the read-out is ~1.0 M MACs (large) with no token dimension, versus attention + geometric
products over 66 tokens × 4 blocks for GATr. On the RTX Pro 6000 the backbone (ResNet-101 fwd+bwd, ≈ 1.5 TFLOP
per 64-image step) dominates; the GATr heads' 175–230 s/epoch already include ~8 vector-field calls per
image. Expect the new head at the backbone floor (≈ the fast end of that range or below), and inference
(20 steps × 32 samples) several times faster than GATr. The one GPU caveat is `o3.matrix_to_angles` +
`wigner_D` (7 tiny kernels per ℓ, latency-bound, ~1–2 ms per call): if it shows up, replace by a closed-form
batched Wigner-D (e.g. the `_Jd`/z-rotation product done in one fused op, or caching `D(R₀)` and updating by
`D(exp dt v)` along the Euler path). **[reasoned]**

### 4.4 Plugging into `CliffordFlow` (`pose3d/pose3d/models/clifford_flow.py`)

* New file `pose3d/pose3d/models/so3fourier_head.py` with `SphereConditionHead` and `SO3FourierVelocity`
  (Appendix A, plus `c_in`/`grid` validation as in `DirectionCloudHead`).
* `ImageToMultivectors.__init__`: `cond_tokens="sphere"` branch → `self.conv_adapter = SphereConditionHead(
  backbone_channels, c_s, c_f, lmax)`, `self.n_mv = c_f` (informational); `forward` already returns the
  adapter output untouched for the non-pooled modes (`if self.cond_tokens in (...)`: add `"sphere"`).
* `CliffordFlow.__init__`: accept `vector_field="so3fourier"`; require `cond_tokens="sphere"`,
  `condition_head="none"`, `pose_tokens="rotor"`, no `mlp_heads` / `fisher_checkpoint` (mirror the existing
  checks; relax `pose_tokens != "rotor" needs gatr` is not needed). `self.condition_head = nn.Identity()`;
  `self.vector_field = SO3FourierVelocity(c_f, lmax, hidden, flip_symmetrize)`; the module zero-initialises
  its `out` itself, so skip the GATr zero-init branch (or point `self.vector_field.out` at it: it exists).
* `CliffordFlow.velocity`: new first branch
  `if self.vector_field_kind == "so3fourier": return self.vector_field(rotor_to_matrix(rotor, self.algebra), t, cond_mv)`
  — returns `(B, 3)` in the `(e12, e13, e23)` order of `relative_log`/`ROTOR_BLADES`, so `compute_loss`,
  `predict`, `_medoid`, `repeat_interleave` of the condition (dim 0 of a `(B, C_f, 455)` tensor) are unchanged.
* `pose3d/config.py` (`FlowConfig`): `vector_field: Literal["clifford", "gatr", "so3fourier"]`,
  `cond_tokens: Literal[..., "sphere"]`, new knobs `so3f_lmax=6`, `so3f_s2_channels=32`, `so3f_so3_channels=8`,
  `so3f_hidden=44`, `so3f_nside=4`, `so3f_flip_symmetrize=True`, `so3f_act: Literal["gate","grid"]="gate"`;
  `models/__init__.py::_clifford_flow` passes them through. Small/large presets: (32, 8, 44) / (64, 16, 124).
* Device: all grid/harmonic tensors are buffers; `wigner_D` moves its `_Jd` to the input device.

### 4.5 Unit tests (`pose3d/tests/test_so3fourier.py`, same style as `test_dircloud.py`, float64, random
weights with the zero-init undone)

1. `test_hemisphere_grid_is_d4_symmetric`: `R_z(−90°)` and `F` map the 88 points onto themselves.
2. `test_s2_coeffs_rotate_with_the_map`: `head.s2_coeffs(rot90(fmap)) == D(R_z(−90°)) · head.s2_coeffs(fmap)`
   with `o3.Irreps([(1,(ℓ,(−1)^ℓ))]).D_from_matrix`; same with `flip` and `D(F)`; assert the coefficients
   are not trivially zero.
3. `test_so3_coeffs_transform_by_the_left_action`: `head(rot90(fmap))` equals `D(G) c_ℓ` block-wise.
4. `test_readout_invariant_under_any_rotation`: random `G ∈ SO(3)`: `vf(G R, t, D(G) c) == vf(R, t, c)`
   (checks the Wigner-D convention `D(GR) = D(G)D(R)`, not just `R_z`).
5. `test_velocity_invariant_to_turning_image_and_pose_together` (through `CliffordFlow.velocity` with
   `model.condition_head(model.adapter.conv_adapter(fmap))`, `R → R_z(−90°) R`, atol 1e-8).
6. `test_velocity_flips_with_a_mirrored_image`: `v(F R F, flip(map)) == (−1, −1, 1)·v`.
7. `test_flip_breaks_without_symmetrisation` (negative control, `flip_symmetrize=False`; roll must still pass).
8. `test_norm_gate_is_exact_and_grid_relu_is_not` (documents the choice; grid ReLU error ≈ 4e-4 relative).
9. `test_initial_field_gets_a_gradient_through_the_zero_init_readout`.
10. `test_so3fourier_flow_trains_and_samples`: `compute_loss` finite, `predict(n_samples=2, steps=2)` shape
    `(2, 3, 3)`, `CliffordFlow(... vector_field="so3fourier", cond_tokens="pooled")` raises.
11. `test_param_counts`: small/large presets within ±3 % of 0.246M / 1.117M (guards the comparison).

---

## 5. Prototype results **[verified]**

`/tmp/eq_research_4/check_equivariance.py` (float64, random weights, B = 3, `c_in = 32`, `C_s = 8`,
`C_f = 4`, L = 6, h = 16; 5.8 s wall):

```
PASS S2 coeffs: rot90(map) -> D(Rz-90) psi: max abs err 8.44e-15
PASS S2 coeffs: flip(map) -> D(F) psi: max abs err 3.66e-15
PASS SO(3) coeffs: rot90(map) -> D(G) c  (left action): max abs err 3.25e-15
PASS read-out: v(G R, D(G) c) == v(R, c) for random G in SO(3): max abs err 1.94e-16
PASS end-to-end: v(Rz-90 R, rot90 map) == v(R, map): max abs err 1.67e-16
PASS end-to-end: v(F R F, flipped map) == (-, -, +) v(R, map): max abs err 2.78e-16
PASS no-symmetrisation: roll still exact: max abs err 2.78e-16
PASS negative control: flip breaks without symmetrisation, err 1.21e+00
INFO e3nn SO3Activation (grid ReLU, res 10): equivariance error 4.29e-04 relative 3.67e-04 (norm gate above: exact)
PASS zero-init read-out gets a gradient: max abs err -3.82e-01
PARAMS small (~0.25M): condition 78,460 + velocity 168,039 = 246,499
PARAMS large (~1.1M): condition 181,996 + velocity 934,839 = 1,116,835
PARAMS small, lmax=4: condition 117,964 + velocity 147,007 = 264,971
ALL CHECKS PASSED
```

Convention checks (`/tmp/eq_research_4`, first scratch run): `D^ℓ(G)D^ℓ(R) = D^ℓ(GR)` to 2e-14 for
ℓ = 1, 3, 6; `Y_ℓ(Fx) = (−1)^ℓ D^ℓ(R_x(π)) Y_ℓ(x)` to 1e-14 and `D̂^ℓ(F)` symmetric; 512 Wigner-D blocks
(ℓ ≤ 6) in 48 ms (float64, CPU).

---

## 6. Risks, and a fair comparison plan

**Risks.**
1. *Band-limit vs. precision.* L = 6 Wigner functions have ~30° half-wavelength; the flow must drive `R_t`
   to within ~5–9°. The read-out is a universal function of `c̃`, which varies smoothly with `R`, so
   sub-resolution localisation is possible in principle (I2S's gradient ascent on the same L = 6 signal
   works), but the *information* about fine pose in `c` is limited by L and `C_f`. Ablate L ∈ {4, 6, 8}
   (I2S: L > 6 hurt on ModelNet) and `C_f`.
2. *The 7×7 map is the bottleneck of the lift.* 49 numbers per channel are spread over 88 points; the Gaussian
   `σ`, coverage `ρ` and the per-radius gains are learnable but D4-tied, so the lift cannot learn the free
   pixel→point map I2S ends up with. If the equal-param twin (below) wins, try the 14×14 `layer3` map.
3. *Exactness is C4 + flip, like P1; roll jitter in training is 7.7° std.* The D4 symmetry only halves /
   quarters the hypothesis space of a ~0.1–0.9M head behind a 42.5M non-equivariant backbone; the honest
   prior from `geometric_redesign.md` §5 stands: perception dominates, gains ≤ 0.3–0.5°, inside 2× the seed
   spread unless 3 seeds are run.
4. *Optimisation.* The first layer is 3,640 / 7,280 wide with inputs of very different scale across ℓ; the
   norm gate helps but a per-ℓ RMS normalisation of `c̃` (invariant) may be needed; `l_gain`, `σ`, `ρ` can
   drift (clamp `ρ ∈ [0.5, 1]`). The 1×1 conv 2048 → 32 is half the small budget, as in `DirectionCloudHead`.
5. *Flip symmetrisation assumes bilateral symmetry of every class* — the same assumption the loader's flip
   relabelling `F R F` already makes, so it costs nothing new; but it doubles the (tiny) read-out.
6. *Wigner-D on GPU* is latency-bound (§4.3); measure before blaming the design.

**Fair comparison plan (equal parameters, same recipe as the P1-small / DiT-small runs: ResNet-101,
batch 64, EMA, warp + synth, `n_time_samples = 8`, 20 Euler steps, 32-sample medoid, class-mean of
per-class median; 3 seeds where it matters).**

| run | params | purpose |
|---|---|---|
| **A. so3fourier-small** (32/8/44, L = 6, Z2 on) | 0.246M | the design |
| **B. non-equivariant twin**: same `SphereConditionHead`, read-out `MLP([flatten(R), t, flatten(c)])` *without* canonicalisation, `h` chosen for equal params | 0.25M | isolates the equivariance: same input, same capacity |
| C. DiT-small (existing, 0.249M, 9.09°) and P1-small (0.232M, 9.41°) | | the brief's baselines |
| D. cells-DiT: DiT over the 49 per-cell scalars + 2-D positions (control of `geometric_redesign.md` §3.1) | 0.25M | separates "per-cell input" from "Fourier head" |
| E. large: so3fourier-large (64/16/124) vs DiT 1.19M (8.85°) / GATr 1.13M (8.95°) / v4 1.6M (8.61°) | 1.12M | the main comparison, 3 seeds |
| F. ablations on A: L ∈ {4, 8}; Z2 off; grid-ReLU (`SO3Activation`) instead of the gate; S²-only read-out (canonicalise `Ψ` instead of `c`: `Linear(C_s·49 → h)`, no S² conv); + I2S auxiliary CE on channel 0 (36k grid, weight 0.1) | | which part of the I2S recipe matters |
| G. zero-training probe: D4-TTA on the existing DiT-small / P1-small checkpoints | 0 | upper bound on what exact D4 is worth at all |

*Decision rule.* Equivariance "matters" on Pascal3D+ only if A beats B by ≥ 0.3° (≈ 2× the 8.93/9.12 seed
spread) at both sizes over 3 seeds; the Fourier family "beats GATr" only if E beats the 1.13M GATr and the
1.6M v4 at ≤ 1.12M params. If A ≈ B ≈ D, the per-cell input (not the symmetry) is the gain and the
project's earlier conclusion (plain transformer suffices) holds for this family too. Report the velocity
equivariance error under `rot90` on real test images as a diagnostic alongside the metric, for every run.

**First experiment:** A vs B vs C (one seed each, small size, identical recipe), ~3 GPU-hours total at
the current epoch times; then 3 seeds of the better of A/B and E.

---

## 7. References

1. Klee, Biza, Platt, Walters. *Image to Sphere: Learning Equivariant Features for Efficient Pose Prediction.*
   ICLR 2023, arXiv:2302.13926 (local `i2s.txt`; installed package `image2sphere`).
2. Howell, Klee, Biza, Zhao, Walters. *Equivariant Single View Pose Prediction via Induced and Restricted
   Representations.* NeurIPS 2023, arXiv:2307.03704 (read via arxiv.org/html).
3. Cohen, Geiger, Köhler, Welling. *Spherical CNNs.* ICLR 2018 (S²/SO(3) convolution theorem, grid ReLU).
4. Geiger & Smidt. *e3nn.* arXiv:2207.09453 (`o3.wigner_D`, `spherical_harmonics`, `SO3Activation`).
5. Brehmer, de Haan, Behrends, Cohen. *Geometric Algebra Transformer.* NeurIPS 2023 (the heads compared against).
6. Puny et al. *Frame Averaging for Invariant and Equivariant Network Design.* ICLR 2022 (the Z2 Reynolds average).
7. `pose3d/reports/geometric_redesign.md` (symmetry inventory of the loader; P1; ℓ ≤ 1 bottleneck of Cl(3,0) tokens).

---

## Appendix A. Prototype module (`/tmp/eq_research_4/so3fourier_head.py`, as tested)

```python
import math
import healpy as hp
import numpy as np
import torch
import torch.nn as nn
from e3nn import o3


def hemisphere_grid(nside: int = 4) -> torch.Tensor:
    """HEALPix pixel centres with z > 0, (P, 3). D4-symmetric about e3."""
    x, y, z = hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))
    pts = np.stack([x, y, z], -1)[z > 1e-9]
    return torch.tensor(pts, dtype=torch.get_default_dtype())


def wigner_blocks(R, lmax):
    """[D^l(R)]_l for a batch of rotation matrices (B, 3, 3)."""
    abc = o3.matrix_to_angles(R)
    return [o3.wigner_D(l, *abc) for l in range(lmax + 1)]


class SphereConditionHead(nn.Module):
    """fmap (B, c_in, g, g) -> SO(3) Fourier coefficients (B, C_f, sum_l (2l+1)^2)."""

    def __init__(self, c_in=2048, c_s=32, c_f=8, lmax=6, grid=7, nside=4, coverage=0.9, sigma=0.2,
                 taper_beta=math.radians(75), max_beta=math.radians(90)):
        super().__init__()
        self.lmax, self.c_s, self.c_f, self.grid = lmax, c_s, c_f, grid
        self.proj = nn.Conv2d(c_in, c_s, 1)
        pts = hemisphere_grid(nside)
        self.register_buffer("pts", pts, persistent=False)
        max_r = pts[:, :2].norm(dim=1).max()
        self.register_buffer("q", pts[:, :2] / max_r, persistent=False)          # orthographic image position
        off = (torch.arange(grid) - grid // 2).to(pts.dtype) / (grid // 2)
        rows, cols = torch.meshgrid(off, off, indexing="ij")
        p = torch.stack([cols.flatten(), rows.flatten()], -1)                    # (u, v) = (col, row): e1, e2
        self.register_buffer("p", p, persistent=False)
        rad2 = (p ** 2).sum(-1)
        uniq = rad2.unique()
        self.register_buffer("ridx", torch.searchsorted(uniq, rad2), persistent=False)
        self.radial_gain = nn.Parameter(torch.ones(len(uniq)))                    # D4-invariant per-cell gain
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma)))
        self.coverage = nn.Parameter(torch.tensor(coverage))
        beta = torch.acos(pts[:, 2].clamp(-1, 1))
        self.register_buffer("taper", ((beta - max_beta) / (taper_beta - max_beta)).clamp(max=1.0), persistent=False)
        Y = o3.spherical_harmonics(list(range(lmax + 1)), pts, normalize=True, normalization="component")
        self.register_buffer("Y", Y, persistent=False)                            # (P, (L+1)^2)
        self.l_gain = nn.Parameter(torch.ones(c_s, lmax + 1))                     # isotropic learned S2 kernel
        self.psi = nn.ParameterList([nn.Parameter(torch.randn(c_s, c_f, 2 * l + 1) / math.sqrt(c_s * (2 * l + 1)))
                                     for l in range(lmax + 1)])                   # S2 -> SO(3) Fourier filters
        self.gate_a = nn.Parameter(torch.ones(c_f, lmax + 1))
        self.gate_b = nn.Parameter(torch.zeros(c_f, lmax + 1))

    def lift_kernel(self):
        d2 = ((self.p[:, None, :] - self.coverage * self.q[None, :, :]) ** 2).sum(-1)
        k = torch.exp(-d2 / (2 * torch.exp(2 * self.log_sigma)))
        k = k / k.sum(0, keepdim=True)
        return k * self.radial_gain[self.ridx][:, None] * self.taper[None, :]    # (g^2, P)

    def s2_coeffs(self, fmap):
        feats = self.proj(fmap).flatten(2)                                        # (B, C_s, g^2)
        sig = torch.relu(feats @ self.lift_kernel())                              # (B, C_s, P)
        psi = sig @ self.Y / math.sqrt(self.Y.shape[0])                           # (B, C_s, (L+1)^2)
        gains = torch.cat([self.l_gain[:, l:l + 1].expand(-1, 2 * l + 1) for l in range(self.lmax + 1)], 1)
        return psi * gains[None]

    def forward(self, fmap):
        psi = self.s2_coeffs(fmap)
        out, start = [], 0
        for l in range(self.lmax + 1):
            n = 2 * l + 1
            c = torch.einsum("bxm,xyn->bymn", psi[:, :, start:start + n], self.psi[l])   # (B, C_f, m, n)
            norm = c.flatten(2).norm(dim=-1) / n                                  # left- and right-invariant
            gate = torch.sigmoid(self.gate_a[:, l] * norm + self.gate_b[:, l])
            out.append((c * gate[..., None, None]).flatten(2))
            start += n
        return torch.cat(out, -1)                                                 # (B, C_f, 455)


class SO3FourierVelocity(nn.Module):
    """(R (B,3,3), t (B,), c (B, C_f, N)) -> body velocity bivector (B, 3) = (e12, e13, e23)."""

    def __init__(self, c_f=8, lmax=6, hidden=44, t_dim=64, flip_symmetrize=True):
        super().__init__()
        self.c_f, self.lmax, self.t_dim, self.flip_symmetrize = c_f, lmax, t_dim, flip_symmetrize
        n = sum((2 * l + 1) ** 2 for l in range(lmax + 1))
        self.first = nn.Linear(c_f * n, hidden)        # = a bank of `hidden` SO(3) correlations evaluated at R
        self.film = nn.Linear(t_dim, 2 * hidden)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.out = nn.Linear(hidden, 3)
        nn.init.zeros_(self.out.weight), nn.init.zeros_(self.out.bias)
        abc = o3.matrix_to_angles(torch.diag(torch.tensor([1.0, -1.0, -1.0])))
        self.DF = [((-1) ** l) * o3.wigner_D(l, *abc) for l in range(lmax + 1)]   # D^l(F), F = diag(-1,1,1)
        self.register_buffer("S", torch.tensor([-1.0, -1.0, 1.0]), persistent=False)

    def _t_embed(self, t):
        half = self.t_dim // 2
        freqs = torch.exp(-math.log(1e4) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
        a = 1000.0 * t[:, None] * freqs[None]
        return torch.cat([a.cos(), a.sin()], -1)

    def canonicalize(self, R, c):
        """c~_l = D^l(R)^T c_l, list of (B, C_f, 2l+1, 2l+1)."""
        out, start = [], 0
        for l, Dl in enumerate(wigner_blocks(R, self.lmax)):
            n = 2 * l + 1
            cl = c[:, :, start:start + n * n].view(c.shape[0], self.c_f, n, n)
            out.append(torch.einsum("bmk,bymn->bykn", Dl, cl))
            start += n * n
        return out

    def _mlp(self, ct_flat, t):
        g = self.first(ct_flat)
        gamma, beta = self.film(self._t_embed(t)).chunk(2, -1)
        return self.out(self.mlp(g * (1 + gamma) + beta))

    def forward(self, R, t, c):
        ct = self.canonicalize(R, c)
        v = self._mlp(torch.cat([x.flatten(1) for x in ct], 1), t)
        if not self.flip_symmetrize:
            return v
        mirrored = torch.cat([torch.einsum("mk,bykn->bymn", DF.to(x), x).flatten(1) for DF, x in zip(self.DF, ct)], 1)
        return 0.5 * (v + self.S * self._mlp(mirrored, t))
```

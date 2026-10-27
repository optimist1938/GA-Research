# Geometric redesign of the Clifford-Flow pose estimator: what symmetries the task really has, and how to give GATr a reason to exist

Branch `so2-cond-head` (ecac58d..d8c3650), October 2026. Research + design only; nothing was trained.

Everything marked **[verified]** was checked numerically on this machine with the scripts in
`/tmp/geo_research/` (`loader_geom.py`, `gatr_pin_check.py`, `gatr_pin_check2.py`, `so2_l2_check.py`; each < 10 s CPU).
Everything marked **[reasoned]** is derivation or literature only. Numbers quoted for our runs are the
ones in the task brief (single seed unless stated).

---

## 0. Summary of findings

1. **The loader already implements the "virtual camera" of Hypothesis 2 exactly.** The crop is the
   homography `P = K_new · G · K⁻¹` of a pure camera rotation `G` and the label is `G · R_annot`
   (`image2sphere/pascal_dataset.py::Pascal3DReal.__getitem__`, `pose3d/datasets/packed.py::_warp_to_224`).
   **[verified]** But the camera is a *fictional* one: Pascal3D+ fixes `focal·viewport = 1·3000 px`
   for every image, so a 224-px crop has a half-field-of-view of **0.6–3.8°** and every bearing ray lies
   within **0.7–4.6° of the optical axis**. **[verified]** The "sphere" of bearing rays is a cap of a few
   degrees: lifting pixels to rays buys numerically nothing over the plane, and the exact SO(3) symmetry
   of the pinhole model collapses, on the data, to SO(2) roll plus (near-)translations: Howell et al.'s
   SE(2)-on-the-image-plane setting [3].
2. **The only non-trivial symmetries actually realised by the data are small:** roll jitter with
   std **7.7°** (max ≈ 30°), bbox-centring rotations `G` of ≤ ~5°, and the horizontal flip. The flip
   relabelling `(az, el, θ) → (−az, el, −θ)` is *exactly* `R' = F R F` with `F = diag(−1, 1, 1)` on both the
   camera and the object side, crop included. **[verified]** That is a reflection, i.e. an element of
   Pin(3) that GATr is equivariant to and an SO(3)-equivariant network or a plain transformer is not.
   This is the one symmetry in the task where GATr's group is *strictly larger* than the alternatives'.
3. **The current SO(2) head is Howell's induction layer truncated at ℓ = 1.** Each token keeps only the
   m = 0 and m = ±1 circular moments of its 49-cell weight map; a `cos 2φ` or `cos 4φ` pattern leaves
   ~1e-7 in the directional slots. **[verified]** A Cl(3,0) multivector can only carry ℓ ≤ 1 (grades
   0,3 → ℓ=0; grades 1,2 → ℓ=1) **[reasoned]**, so *per-token* multivectors cannot encode where in the
   image a feature sits beyond a dipole. v5 (SO(2) tokens alone, equal params) tracking the 8.95 baseline
   is what this bottleneck predicts. Higher angular frequency can only come from *sets* of tokens with
   distinct positions, i.e. from a point/ray cloud, which is exactly the input type GATr was built for.
4. **The vector-dual read-out used by `pose_tokens=frame/frame_ch` is polar; angular velocity is
   axial.** Under rotations the two are indistinguishable, under the flip they differ by a sign, so the
   current frame-token model cannot be flip-equivariant even in principle. The bivector read-out
   (`e12, e13, e23`) is Pin-correct. **[verified with a real GATr forward]**
5. **GATr's `join_reference` must stay `"data"`** (the repo's default): with `"canonical"` reflections
   break (as the GATr paper says [1]); rotation equivariance holds for both. **[verified]**
6. **The up prior can be injected without breaking any symmetry:** the loader knows the photo's up
   direction in the crop frame, `g = G(−e2)`; feeding `g` as a (polar) direction token is exact under
   every camera rotation and the flip, and removes the 7.7° roll-jitter noise that the constant `−e2`
   token (`--so2_up_token`) leaves for the model to absorb. **[reasoned; `g` values checked]**
7. **Everything above lives in the ~98k-parameter vector field.** The ResNet-101 (≈43.5M of 43.6M
   params) is not equivariant to anything and all vector fields land in 8.6–9.3°. The honest prior is
   that the perception bottleneck dominates; the proposals below are ranked by how cheaply they can be
   falsified, and §5 says where the gain would not materialise.

---

## 1. The geometry of the task as the loader implements it

### 1.1 The Pascal3D+ camera and the label
Pascal3D+ annotates each object with `azimuth a, elevation e, theta θ, distance d, focal f = 1,
viewport = 3000, principal point (px, py)`. `get_camera_matrix` builds (`fv = f·viewport = 3000 px`)

```
P = T(pp) · Y · R2d(θ) · diag(fv, fv, −1) · [R_ae | −R_ae C],      R_ae = Rx(e − π/2) · Rz(−a),
C = d (cos e sin a, −cos e cos a, sin e),  Y = diag(1, −1, 1),  T(pp) = [[1,0,px],[0,1,py],[0,0,1]]
```

and `split_camera_matrix` RQ-decomposes `P[:3,:3]` into an upper-triangular `K` with positive diagonal
and a proper rotation `R_annot`; the in-plane rotation θ, the `−1` and the y-flip are absorbed into
`R_annot`. So `R_annot` is the object→camera rotation of a fictional pinhole camera with
`K = diag(3000, 3000, 1)` (up to signs) whose principal point is the projection of the object centre:
**the annotated camera already looks straight at the object** (allocentric/viewing-ray-relative pose,
cf. the egocentric–allocentric distinction in 3D-RCNN [10]). Note the loader uses `px` for *both*
principal-point coordinates (`principal_point = [px, px]`); this is inherited from Mohlin et al.'s code
[4], is applied identically in train and test, and amounts to a per-image tilt of the label frame of
`atan((py − px)/3000)` ≤ ~6° that every method using this loader shares. **[reasoned from the code]**

### 1.2 The crop is a virtual camera rotation, exactly
`get_back_proj_bbx` lifts the bbox corners to unit rays `K⁻¹[u, v, 1]`, `get_desired_camera` finds the
minimal spherical cap covering them (centre `z`), builds `G = [y; x; z]` with `y = desired_up ⊥ z`,
`x = −y × z`, and chooses `K_new = [[112 f', 0, 112], [0, 112 f', 112], [0,0,1]]`, `f' = 1 / max |G·ray|_{x,y}/z`.
The image is resampled with `P = K_new · G · K⁻¹` (`P /= P[2,2]`), which is precisely the homography
induced by rotating a pinhole camera about its centre, and the label is `extrinsic_after = G · R_annot`.
**[verified: the code literally computes this product]** Consequences:

| quantity | value | status |
|---|---|---|
| half-FOV of the 224 crop for bbox widths 60 / 120 / 250 / 400 px | 0.57° / 1.15° / 2.39° / 3.81° | [verified] |
| max angle between a 7×7 cell's bearing ray and `e3` at those focal lengths | 0.69° / 1.39° / 2.90° / 4.62° | [verified] |
| centring rotation `‖log G‖` for a bbox in the image corner (500×375 image) | 3.05° | [verified] |
| training roll jitter from `desired_up = [3,0,0] + N(0, 0.4²)` | std 7.68°, max ≈ 30° | [verified] |
| bbox jitter | ±10 % of width/height per edge | [code] |
| test-time `desired_up` | `[1, 0, 0]` → the crop keeps the photo's roll | [code] |
| `g = G(−e2)` (photo up in the crop frame), corner bbox, no jitter | `(0.001, −0.9997, 0.026)` | [verified] |

So the "SO(3) on bearing rays" symmetry is exact *in the fiction* but the fiction is a telephoto lens:
the only rotations the data ever realises are rolls of ≲ 30° and pan/tilts of ≲ 5°. Under a 3° FOV,
a pan/tilt is numerically a translation of the image, and translations are what the CNN (approximately)
already handles. This is Howell et al.'s point [3]: the symmetry that can be imposed on the image is
SO(2) (really SE(2)), and the SO(3) structure of the output is reached through the *induced*
representation `Ind_{SO(2)}^{SO(3)}`, i.e. signals on S².

### 1.3 The flip is a reflection
`(a, e, θ) → (−a, e, −θ)` together with mirroring the image, bbox and principal point gives
`label_flip = F · label · F`, `F = diag(−1, 1, 1)`, to machine precision, crop included. **[verified]**
Geometrically: mirror the camera through its `x = 0` plane (that is the image flip) *and* mirror the
object through its own `x = 0` plane. For a bilaterally symmetric object the second mirror is the
identity on the shape, so the flipped image is a genuine image of the same category in pose `F R F`.
All 12 Pascal classes are (approximately) bilaterally symmetric; the augmentation is exact for the
rendered ShapeNet models and approximate for real instances (asymmetric texture, steering wheels, etc.).

### 1.4 Synthetic data
RenderForCNN [11] renders use `f·viewport = 3000`, `d = 4`, principal point at the image centre and the
full image as bbox, then the same warp. Their θ distribution is also narrow. Labels are relabelled
`a = a_file − 90` for ShapeNet-v2 classes (`packed.py` docstring).

---

## 2. Which symmetries are exact, approximate, broken; and what the pipeline loses

### 2.1 Symmetry inventory

| transformation of (image, label) | exact? | realised in the data? | used by the current model? |
|---|---|---|---|
| roll about the optical axis: image rotated by φ, `R → Rz(φ) R` | exact (pinhole, any FOV) | yes, std 7.7° (train); ≈ 0 (test); plus annotation θ spread | SO(2) head: exact only for 90° on the 7×7 grid; backbone: learned/approx. |
| pan/tilt by `G`: image warped by `K G K⁻¹`, `R → G R` | exact in the fictional camera | ≤ 5° (bbox centring), otherwise only via bbox jitter | no (never built in; indistinguishable from translation at this FOV) |
| horizontal flip: `R → F R F` | exact for bilaterally symmetric shapes | 50 % of training samples | no: tokens have `e12/e123` slots treated as invariants and a polar read-out, both wrong under `F` |
| image translation (object off-centre) | approximate (≡ pan/tilt at 3° FOV) | ±10 % jitter | CNN approx.; pooled tokens ignore position; SO(2) head's dipoles are position-dependent |
| camera translation / object scale | pose invariant; image scale changes | yes (bbox normalisation removes most) | n/a |
| object-frame symmetries (180° about vertical for tables/boats, etc.) | label convention breaks them consistently (I2S, [2]) | yes | only via the flow being multimodal + medoid |

### 2.2 What is lost, and where (pixels → velocity)

1. **Backbone (ResNet-101, 43.5M).** Not equivariant to roll, flip or anything. Everything downstream
   is "exactly equivariant given exactly-equivariant per-cell scalars", which the backbone only learns
   approximately from the augmentation. This is the same situation as I2S [2] and Howell [3]
   (both use a pretrained ResNet-101 and report state of the art anyway).
2. **Baseline condition path (`cond_tokens=pooled`, `conv_adapter=False`).** Global average pooling
   destroys all position information; the reshape declares 8 arbitrary channels a multivector, so the
   Clifford condition head's O(3) equivariance acts on a transformation the data never performs. GATr's
   group action is wasted here by construction.
3. **SO(2) head.** Keeps the exact 90°-roll equivariance but, per token, only ℓ ≤ 1 circular moments
   of each of its `6k` weight maps **[verified]**: an orientation feature with 180° periodicity
   (edges, elongation; m = ±2) or any finer angular structure is invisible to the directional slots and
   survives only as a rotation-invariant scalar. The `e12` and `e123` slots are filled with invariants,
   but they are axial (sign-flipping under `F`): the head is not flip-covariant.
4. **Frame tokens + vector-dual read-out.** Correct for rotations (left action: see §3.1 proof), but
   polar; cannot be flip-equivariant **[verified]**.
5. **Constant `−e2` up token.** Breaks roll equivariance deliberately, but it is only the *approximate*
   photo-up (off by the jitter in training). The exact photo-up `g = G(−e2)` is available for free.
6. **GATr's PGA.** All Cl(3,0) vectors are written into the PGA vector grade (`e1, e2, e3` with
   `e0 = 0`), i.e. as *planes through the origin*, and no token ever has `e0` content: translations
   never act, the join's reference multivector is the mean of planes, and the E(3) part of GATr is dead
   weight. Harmless (O(3) equivariance survives), but it is why GATr currently has no edge over an
   O(3)-equivariant Clifford MLP or a plain transformer.

### 2.3 Verdicts on the five hypotheses

* **H1 (roll is the only exact symmetry used; roll-invariance throws "up" away).** Confirmed in the
  first half, refined in the second: the equivariant (not invariant) model does not lose "up", it has to
  *re-derive* it from the ℓ = 1 dipoles of non-equivariant features through a narrow bottleneck, which
  is what the 17-epoch plateau and the saddle fixed by `--so2_split_norm` look like. With the up
  vector supplied as a token that *transforms*, equivariance and the prior coexist (H5).
* **H2 (full SO(3) via bearing rays; the crop is a virtual camera).** The crop-to-virtual-camera
  mapping is exact and already computed; intrinsics (`K_new`), `G` and `g` can be returned from
  `_warp_to_224` with a two-line change. The label convention is consistent (pose in the virtual
  camera). But the fictional focal length makes the ray lift degenerate (cap of ≲ 4°) and the dataset
  realises only rolls and ≲ 5° pan/tilts, so there is no *information* in the SO(3) extension; the
  useful content of H2 is (i) an exact handle on `g` and `G`, (ii) that image translation = small
  rotation, so a *direction*-token design is the right one, with positions well separated by a
  hemisphere lift rather than by the true rays.
* **H3 (GATr has no geometric input).** Confirmed. Rays are near-degenerate (above); the two honest
  geometric inputs are a *planar/hemispherical cloud of cell directions* (cheap; exact roll + flip)
  and a *depth-lifted 3D point cloud* (expensive; gives translations and the join something to do, but
  only roll and flip stay exact because relative depth is known up to an affine map along `e3`).
* **H4 (backbone breaks equivariance first).** Confirmed. Options ranked by cost: accept learned
  approximate equivariance (what I2S/Howell do); D4 frame averaging at test time (8 backbone passes,
  exact, zero training: a probe of how much exactness is worth); canonicalisation (§3.3); an
  E(2)-steerable ResNet [7] (no ImageNet weights at 101 depth: unrealistic here).
* **H5 (gravity/up prior with exact symmetry).** Solved by the `g` token (§3.1); Pascal3D has no
  gravity, but photo-up is the prior the dataset actually has (θ is small).

---

## 3. Proposals, ranked

Conventions: `ρ_u(x) = u x̂ u⁻¹` is GATr's Pin(3,0,1) action (x̂ = grade involution for odd `u`) [1].
For a reflection `F` in the camera `x = 0` plane (`u = e1`), **[verified with GATr's own geometric product]**:

| PGA object | grade / slots | under rotation `G` | under `F` | translation-invariant |
|---|---|---|---|---|
| direction as "plane through origin" `n` | vector `(e1,e2,e3)`, `e0 = 0` | `G n` | `F n` (polar) | no (gains `e0`) |
| direction as "line through origin" `a` | bivector `(e12,e13,e23)` | `G a` | `−F a` (axial) | yes |
| ideal point `d` | trivector, `e123 = 0` | `G d` | `−F d` (axial) | yes |
| point `p` | trivector `embed_point` | `G p` | `−embed_point(F p)` (projective sign) | moves |

Because of the last row, **points cannot be flip-covariant** under the natural construction (the
whole token flips sign), whereas polar directions as vector grade and axial directions as `e_ij`
bivectors are. Hence every proposal below that wants the exact flip uses only origin-directions,
and the pose frame is fed as two polar axes plus one axial axis.

### 3.1 P1 (rank 1): hemisphere direction-cloud tokens with a Pin-exact head and the loader's up vector

**Data flow.** `image (3×224×224) → ResNet-101 → 2048×7×7 (or layer3: 1024×14×14) → per-cell 1×1 conv
to s = 64 scalars → token j = (direction s_j as PGA vector grade, 64 scalar channels)`; constant tokens
`e3` (optical axis) and `g = G(−e2)` (photo up, from the loader); pose tokens `f2 = R e2`, `f3 = R e3`
(vector grade) and `l = (R e1) e123` (bivector `e12 = l3, e13 = −l2, e23 = l1`); time `t` broadcast as a
scalar channel, query flags as one-hot scalar channels. GATr (`join_reference="data"`, as now).
Read-out: the bivector part `(e12, e13, e23)` of the `l` token = spatial angular velocity `w`;
body velocity `v = r̃ w r` as now; Euler step unchanged.

**Directions.** `s_j = (x_j, y_j, √(1 − x_j² − y_j²))` with `(x_j, y_j) ∈ [−ρ, ρ]²` the cell centre in
normalised crop coordinates, ρ ≈ 0.9 (I2S's orthographic hemisphere lift [2], a special case of
Howell's induction layer [3]). The *true* bearing rays `K_new⁻¹[u, v, 1]` are a valid alternative but
span ≤ 4.6°, so `⟨f_i, s_j⟩ ≈ R_3i` for every cell and geometric attention would be blind to position
**[verified angles]**; the hemisphere lift is the same symmetry content with well-conditioned geometry.
Variant P1-b: bilinearly resample the map onto a HEALPix hemisphere (I2S's grid, ~hundreds of points)
for continuous-roll exactness of the *head* (the backbone grid still only has C4).

**Why this fixes the verified bottlenecks.** (i) Every cell is its own token, so angular frequencies
ℓ ≥ 2 are represented by the *set* of directions, not truncated at ℓ = 1; the GATr attention keys on
`⟨s_j, s_k⟩`, `⟨f_i, s_j⟩`, `⟨g, s_j⟩`, `⟨f_i, g⟩` (all Pin-invariant), i.e. "how does the hypothesised
object axis `f_i` relate to the direction of cell j and to up", which is the right feature for a
velocity field on SO(3). (ii) Flip-exact head. (iii) Exact up vector.

**Proof sketch (equivariance).** Let `T` be a camera rotation `G` or the flip `F`, acting on the token
set by `ρ_T`. The tokens built from the transformed image equal `ρ_T` of the original tokens *provided
the per-cell scalars are invariant* (cells permute under 90° rolls and the flip; `s_j → T s_j`;
`e3 → T e3`, `g → T g`; `f_2, f_3 → T f_i`; for the flip `f'_i = F R F e_i = s_i F f_i` with
`s = (−1, 1, 1)`, and the axial embedding of `f_1` absorbs `s_1 = −1`: `ρ_F(l) = −F l = l'`). GATr is
Pin-equivariant with `join_reference="data"` [1], so the bivector read-out satisfies `w' = ρ_T(w)`:
`G w` for rotations, `−F w` (axial) for the flip. Body velocity: `v = Rᵀ w → (GR)ᵀ G w = v` (invariant,
so the Euler step `r ← r exp(dt v)` commutes with left multiplication: that is the left-vs-conjugation
issue resolved exactly as in `_frame_velocity`); for the flip `v' = (FRF)ᵀ(−F w) = −F v`, which is the
axial form of `v' = F v F`, and `R' exp(dt v') = F R F · F exp(dt v) F = F (R exp(dt v)) F`: the
flow of the flipped image is the flipped flow. **[all four identities verified numerically on a random
GATr; the vector-dual read-out passes the rotation tests and fails the flip test]**

**Up prior.** `g` is an input direction that transforms with the camera, so the model is exactly
equivariant *and* can learn "f_3 (object up) ≈ g" as a prior; at test time `g` is the photo-up rotated
by the ≤ 5° centring `G`. Ablations: constant `−e2` (current), `g` (proposed), none (pure symmetry).

**Parameters.** 1×1 conv 2048→64: 131k; GATr with `in_s_channels = 64 + 6`, hidden mv 8 / s 32–64,
4 blocks: ≈ 0.1–0.3M. Total ≈ 43.6M + 0.2M (drop the 0.5M `SO2ConditionHead` and Clifford head).

**Compute.** 7×7: 54 tokens vs 67 now → ≈ same, ~210 s/epoch. 14×14: 201 tokens; with
`n_time_samples = 8` the vector field runs 8× per image, rough estimate +25–30 %/epoch **[reasoned]**.

**Code changes.** `pose3d/models/so2_head.py`: new `DirectionCloudHead(c_in, s_out, grid, lift=
"hemisphere"|"rays")` returning `(mv (B, N+2, 16) PGA, scalars (B, N+2, s_out+flags))` (PGA directly,
not Cl(3,0)); `pose3d/models/gatr_denoiser.py::GATrVectorField`: accept scalar channels per token,
embed the frame as 2 vectors + 1 bivector (`pose_tokens="frame_pin"`), read out `(8, 9, 10)` of the `l`
token; `pose3d/models/clifford_flow.py::_frame_velocity`: skip the dual; `pose3d/datasets/packed.py::
_warp_to_224` and both `__getitem__`: also return `up = G[:, :3] @ [0, −1, 0]` (and `K_new`, `G`);
`pose3d/datasets/cache.py::InMemoryDataset`, `engine/trainer.py::_compute_loss`,
`engine/metrics.py`: pass `up` through (`model.compute_loss(img, rot, cls=..., up=...)`). Config flags:
`cond_tokens=dircloud`, `dircloud_lift`, `dircloud_scalars`, `up_token=const|loader|none`,
`pose_tokens=frame_pin`.

**Cheapest falsifying experiments (in order).**
1. *Zero-training probe:* evaluate the existing v4/8.95/DiT checkpoints with flip-TTA and D4-TTA
   (transform image, map the velocity/pose back, pool the 32×k samples with the medoid). If no
   checkpoint gains ≥ 0.2° (the seed spread is 8.93/9.12), exact roll/flip symmetry has no residual
   value and P1's gain can only come from the ℓ ≤ 1 bottleneck fix.
2. *One run on v4:* `pose_tokens=frame_pin` (bivector read-out, axial `f_1`) + `up_token=loader`,
   everything else as v4. Checks that the bivector read-out does not re-create the start-up saddle (the
   `l` token now carries bivector signal from step 0) and isolates the `g` effect.
3. *P1 vs its control:* P1 (7×7) against **DiT over the same 49 per-cell tokens + 2-D sinusoidal
   positions + `g`** at equal params. This is the experiment the project needs: if GATr-P1 is not
   ≥ 0.3° better than DiT-cells, equivariance is not worth anything on Pascal3D and the report's
   answer is "no".

### 3.2 P2 (rank 2): depth-lifted 3D point tokens — the only input where GATr's E(3) is not dead weight

**Data flow.** `crop → Depth Anything V2 (frozen; S: 25M, B: 97M [8]) → relative inverse depth
δ_j per cell → z_j = 1/(α δ_j + β)` with `(α, β)` either fixed per sample by normalising the cloud
(median z = 1, MAD = const) or predicted by a tiny invariant MLP; `p_j = z_j K_new⁻¹[u_j, v_j, 1]`
(use the fictional `K_new`, so `p_j ≈ (z_j x_j / f', z_j y_j / f', z_j)` — near-orthographic);
centre the cloud (`p_j −= median`), rescale, and feed **as polar vectors (vector grade) about the crop
centre**, not as PGA points (projective-sign problem above). Per-cell ResNet scalars as in P1, plus
`e3`, `g`, frame tokens as in P1, bivector read-out.

**Equivariance.** Exact for roll (rotates the cloud about `e3`, commutes with the unknown affine depth
map) and for the flip (polar vectors are flip-covariant; depth is mirror-symmetric). *Not* exact for
pan/tilt: the lifted cloud is known only up to an anisotropic affine map along `e3`, which does not
commute with rotations about `e1, e2` **[reasoned]**. Proof sketch otherwise identical to P1.
If PGA *points* are used instead (to use translations and the join), the exact flip is lost and the
cloud must still be centred; translation equivariance then only protects against bbox jitter, which the
centring already does: hence the polar-vector form.

**What it buys.** For the first time GATr sees 3D shape: the orientation of the dominant planes of a
car, the lid of a tv, the seat of a chair are geometric products of point differences, not texture
statistics. This attacks the *perception* bottleneck (elevation, front/back from shape), not the
symmetry one. The repo's earlier `i2p_pointcloud.py` runs are a warning: direct regression from DA
points gave 10.0–10.7°, and `late_ablation` *without* depth (10.12°) beat `late_fusion` with it
(10.74°). Those used PointNet-style max-pooling and no flow; the hypothesis P2 tests is that an
equivariant attention head over the same points, conditioning a flow, does better.

**Parameters / compute.** +25M (S) or +97M (B) frozen, ≈ +0.3M trainable. DA-S forward on 224² ≈
+15–20 %/epoch, DA-B ≈ +50–75 % **[reasoned from ViT FLOPs]**; or precompute per training image once
on the un-warped photo as 3D points under the fictional `K` and rotate them by the per-sample `G`
(exact under the rotation homography), which brings the cost back to ≈ P1.

**Code.** New `pose3d/models/depth_cloud.py` (DA wrapper exists in `ipdf_depth.py::
DepthAnythingV2Backbone`), `clifford_flow.py` option `cond_tokens=depthcloud`, loader returns `K_new`.

**Falsification.** P2 vs P1 at the same backbone and seed. If ≤ 0.2° apart, 3D geometry is not what
the vector field is missing. A cheaper precursor: P1 with the depth `z_j` added as a *scalar* channel
only (no geometry): if that already gives the gain, the geometric embedding is not the cause.

### 3.3 P3 (rank 3): canonicalisation and frame averaging as the honest control for "exactness"

**Design.** Kaba et al. [5]: a small exactly-equivariant network predicts a roll angle φ̂ (here: the
`atan2` of a learned ℓ = 1 dipole from an E(2)-steerable 3-layer CNN [7] on the 224 crop, or the
existing `SO2ConditionHead` dipoles); rotate the crop by `−φ̂`, run the unchanged non-equivariant
pipeline (ResNet + DiT or GATr), left-multiply the result by `Rz(φ̂)`. Exact roll equivariance for any
backbone (exactly C4 given the grid; continuous up to resampling). Flip: `φ̂ → −φ̂` under `F` if the
canonicaliser is D4-steerable; combine with a flip-canonicaliser (sign of a learned pseudo-scalar) or
keep the flip as augmentation. Frame averaging [6] is the training-free version: average the body
velocity over D4 (8 backbone passes).

**Why rank 3.** Pascal3D photos are upright and the test crops keep the photo's roll, so the
canonicaliser will learn φ̂ ≈ 0 and the architecture degenerates to the current one. Its value is as
a *diagnostic*: it is the cheapest way (TTA, no training) to measure whether any exact symmetry would
help at all, for DiT and GATr alike. If D4-TTA helps DiT as much as GATr, symmetry is not GATr's edge.

**Cost.** TTA: 8× eval only. Canonicaliser: +0.1M params, +5 %/epoch.

### 3.4 Not proposed, and why
* **Full spherical-CNN / SO(3)-convolution output (I2S [2], Howell [3]).** The induction layer with
  `ℓ_max = 6` is the principled version of P1; it predicts a density on SO(3) rather than a flow, and
  Howell's best is 9.2° class-mean vs our 8.61/8.95. Replacing the flow is a different project; P1-b
  (HEALPix hemisphere tokens) is the GATr-shaped way to get the same ℓ-content.
* **True-ray / Plücker tokens (Cameras-as-Rays [9]).** Correct in a real camera; degenerate at a 3°
  FOV (table in §1.2). Worth revisiting only on a dataset with real intrinsics (e.g. ObjectNet3D,
  CO3D), where P1 with `lift="rays"` is the drop-in.
* **Equivariant backbone.** No pretrained E(2)-steerable ResNet-101; training one from scratch on
  ImageNet is out of budget, and I2S/Howell show a plain ResNet-101 suffices for SOTA.

---

## 4. Unit tests (to add next to `pose3d/tests/test_so2_head.py`)

All in float64, random weights (undo the zero-init of the read-out as `_flow` does), `join_reference="data"`.

1. `test_pin_action_signs`: with GATr's `geometric_product`/`grade_involute`, `ρ_{e1}(embed_point(p)) ==
   −embed_point(F p)`, vector grade `→ F n`, `e_ij` bivector `→ −F a`, ideal point `→ −ideal(F d)`.
   (Documents why points are not used; **this exact test passed in `gatr_pin_check2.py`**.)
2. `test_dircloud_head_rotates_with_the_map` (C4): `head(rot90(fmap))` equals the head's directions
   rotated by `Rz(−90°)` with the per-cell scalars permuted; `e3` and `g` tokens untouched by the head
   (`g` comes from the loader).
3. `test_dircloud_head_flips_with_the_map`: `head(flip(fmap))` equals directions `F s_j`, scalars
   permuted `(u → −u)`.
4. `test_frame_pin_velocity_invariant_under_rotation`: random `G ∈ SO(3)` (not just `Rz`): rotate
   directions, `g`, `e3`, frame (`f_1` through the axial embedding); body velocity unchanged to 1e-8.
   (The current test only uses `Rz(−90°)`; a general `G` also checks the `e3`/`g` bookkeeping.)
5. `test_frame_pin_velocity_under_flip`: `v(F·tokens, R' = F R F) == −F v(tokens, R)` (axial
   components `(e12, e13, e23) → (−, −, +)`) and the integrated rotor satisfies `R'_{t+dt} = F R_{t+dt} F`.
6. `test_vector_dual_readout_breaks_the_flip` (negative control, like `test_rotor_token_breaks_that_invariance`).
7. `test_canonical_join_reference_breaks_the_flip` (negative control; guards against a GATr
   version/config change).
8. `test_loader_returns_up_and_homography`: on synthetic inputs to `_warp_to_224`, `up == G[:, :3] @
   [0, −1, 0]`, `P == K_new G K⁻¹` after normalisation, and `label == G R_annot`; with `flip=1`,
   `label_flip == F label F` for the un-jittered crop (this is `loader_geom.py`).
9. `test_depthcloud_tokens_are_roll_and_flip_covariant` (P2): `rot90`/flip the (depth, fmap) pair; the
   centred polar vectors rotate by `Rz(−90°)` / reflect by `F`; the normalisation (median/MAD) is invariant.
10. `test_translation_does_not_change_bivector_readout` (P2 with PGA points, if ever used):
    `v(p_j + t) == v(p_j)`; and the matching negative test that vector-grade directions pick up `e0`
    under `ρ_T` (so that a future "translate the tokens" test is written correctly).

---

## 5. Honest risks

1. **Perception dominates the metric.** Published per-class medians (Howell [3] / I2S [2]):
   boat 17.0/21.7°, bicycle 12.6/12.7°, motorbike 11.9/11.5°, sofa 12.1/10.5°, tv 9.9/10.6°, chair
   9.4/9.5° versus bus 3.0/3.3° and car 4.5/4.9°. The class-mean is set by front/back and 180°
   ambiguities of thin or symmetric objects, by low-resolution boats, and by annotation noise. No
   symmetry in §2 addresses a front/back flip; at best P1/P2 sharpen the easy classes by fractions of a
   degree. Expect ≤ 0.3–0.5° from P1, inside 2× the seed spread (8.93 vs 9.12) unless 3 seeds are run.
2. **The backbone is the model.** 43.5M of 43.6M parameters are non-equivariant; the exactness of
   the head is conditional on approximately-invariant per-cell scalars. The D4/flip-TTA probe (§3.1,
   experiment 1) measures this before any training is spent.
3. **The data barely exercises the symmetries.** Roll jitter std 7.7°, pan/tilt ≤ 5°, flip 50 %.
   Equivariance pays off when the data spans the group (GATr's own ablations [1] and I2S's 10k-vs-100k
   SYMSOL finding [2]: with enough data a plain transformer catches up). Pascal3D is in the
   "enough data" regime for roll; the flip is the one place the symmetry halves the effective hypothesis
   space, and it only applies to the 98k-parameter head.
4. **GATr-specific optimisation hazards.** The bivector read-out stalled before (`gatr_denoiser.py`
   comment); the axial frame token should fix it, but if not, the fallback is a per-token "handedness"
   pseudoscalar channel initialised to 1 — which *breaks* the flip symmetry, so it must be reported as
   such. Also the hemisphere lift's `ρ` and the scalar-channel width are new hyper-parameters.
5. **Depth (P2) failed once already in this repo** (`i2p_pointcloud.py`: depth hurt). Relative depth
   on cropped, often blurry, cluttered Pascal images is noisy; the anisotropic ambiguity along `e3`
   means the "3D" cloud is a 2.5-D relief whose exact symmetry group is again only roll + flip.
6. **A fair control can make the whole line moot.** If DiT over 49 per-cell tokens + `g` (the control
   of experiment 3) matches P1, the answer to the project question is that on Pascal3D+ the
   geometric-algebra transformer has no reason to beat an ordinary one, and that is a legitimate result.

---

## 6. References

1. Brehmer, de Haan, Behrends, Cohen. *Geometric Algebra Transformer*. NeurIPS 2023, arXiv:2305.18415.
   (Pin(3,0,1) action `u x̂ u⁻¹`; join needs a reference multivector to be reflection-equivariant;
   plain transformers catch up with enough data.)
2. Klee, Biza, Platt, Walters. *Image to Sphere: Learning Equivariant Features for Efficient Pose
   Prediction*. ICLR 2023, arXiv:2302.13926. (Orthographic hemisphere lift of the 7×7 ResNet map;
   Pascal3D+ class-mean median 9.8 ± 0.4°; local copy `i2s.txt`.)
3. Howell, Klee, Biza, Zhao, Walters. *Equivariant Single View Pose Prediction Via Induced and
   Restricted Representations*. NeurIPS 2023, arXiv:2307.03704. (Theorem 1: an SO(2)-equivariant map
   from image signals to spherical signals has kernel `κ(n̂, r) = Σ_ℓ F_ℓ(r)ᵀ Y_ℓ(n̂)` with `F_ℓ`
   SO(2)-steerable in `ρ ⊗ Res^{SO(3)}_{SO(2)} D^ℓ`; I2S and the icosahedral projections are special
   cases; Pascal3D+ 9.2°.)
4. Mohlin, Bianchi, Sullivan. *Probabilistic orientation estimation with matrix Fisher
   distributions*. NeurIPS 2020, arXiv:2006.09740. (Origin of the loader and of the warp/flip augmentation.)
5. Kaba, Mondal, Zhang, Bengio, Ravanbakhsh. *Equivariance with Learned Canonicalization Functions*.
   ICML 2023, arXiv:2211.06489.
6. Puny, Atzmon, Ben-Hamu, Misra, Grover, Smith, Lipman. *Frame Averaging for Invariant and
   Equivariant Network Design*. ICLR 2022, arXiv:2110.03336.
7. Weiler, Cesa. *General E(2)-Equivariant Steerable CNNs*. NeurIPS 2019, arXiv:1911.08251.
8. Yang et al. *Depth Anything V2*. NeurIPS 2024, arXiv:2406.09414. (Relative, affine-invariant depth;
   metric variants fine-tuned separately; 25M–1.3B params.)
9. Zhang, Lin, Kumar, Yang, Ramanan, Tulsiani. *Cameras as Rays: Pose Estimation via Ray Diffusion*.
   ICLR 2024, arXiv:2402.14817.
10. Kundu, Li, Rehg. *3D-RCNN: Instance-level 3D Object Reconstruction via Render-and-Compare*.
    CVPR 2018. (Egocentric vs allocentric pose.)
11. Su, Qi, Li, Guibas. *Render for CNN*. ICCV 2015 (local copy `r4cnn.txt`).
12. Esteves, Sud, Luo, Daniilidis, Makadia. *Cross-Domain 3D Equivariant Image Embeddings*. ICML 2019,
    arXiv:1812.02716. (Spherical-CNN image embeddings; icosahedral special case of [3].)
13. Xiang, Mottaghi, Savarese. *Beyond PASCAL: A Benchmark for 3D Object Detection in the Wild*.
    WACV 2014. (Annotation fields; `focal = 1`, `viewport = 3000` convention read from the loader.)

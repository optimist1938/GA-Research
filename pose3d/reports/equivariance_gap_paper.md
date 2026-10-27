# Why "Does equivariance matter at scale?" does not predict a GATr win on Pascal3D+

Branch `so2-cond-head`, October 2026. Literature mapping only; nothing was trained. Scratch files in
`/tmp/eq_research_1/` (`extract.py`, `fig2b.py`: text and vector-graphics extraction from the arXiv PDFs with pypdf; the
scaling-law arithmetic is a ten-line script, < 1 s CPU).

Tags: **[text]** = read in the paper's own text or tables; **[fig]** = numbers read off the PDF's vector
graphics (marker coordinates mapped through the axis ticks, accurate to ~2 % in log-space); **[repo]** =
checked in this repository's code or reports; **[inferred]** = my derivation or judgement.

Paper: Brehmer, Behrends, de Haan, Cohen, *Does equivariance matter at scale?*, arXiv:2410.23179v2 (27 Jul
2025), TMLR 07/2025 [text]. I read the v2 PDF end to end (18 pages incl. appendices).

---

## 0. One-paragraph answer

The paper's result is: on a task whose symmetry group (E(3)) is **exact, acts on the raw inputs, is fully exercised
by the data, and is respected by every layer of the network**, an equivariant transformer reaches ~2× lower test
loss than a plain transformer at every compute budget from 1e16 to 1e19 FLOPs, and is more data-efficient, although
augmentation makes the plain transformer equally data-efficient once the number of epochs is in the 1e4–1e5
range [text, fig]. Our setup violates almost every precondition: the task's exact symmetry is only in-plane roll
and the mirror (not SO(3)); the data realises roll with a 7.7° std [repo]; 97.5 % of the parameters and ~65–95 % of
the training FLOPs sit in a non-equivariant ResNet-101 that precedes the equivariant part [repo, inferred]; the
baseline is trained with exactly the augmentation that, in the paper, closes the data-efficiency gap; the
metric is dominated by perception failures and ambiguity that no symmetry addresses; and the comparisons are
within ~2–4× the seed spread. The paper therefore does not predict a gap in our setting, and the absence of one is
not in tension with it. Section 4 ranks the reasons and gives cheap tests.

---

## 1. What the paper measures, and what it finds

### 1.1 Task, data, symmetry
* **Task.** Rigid-body dynamics: input = triangular meshes of 3–10 objects at times t0 and t1 = t0 + Δt (plus a
  floor plane), output = all mesh vertex positions at t2 = t1 + Δt. Loss and metric: MSE of predicted vertex
  positions. Kubric/PyBullet simulation recreating MOVi-B (Allen et al. 2023), 4·10^5 training trajectories × 96
  steps, 98–2160 faces per object, 5470 faces per scene on average; 1000 trajectories each for val/test. [text,
  §3.1, App. A]
* **Symmetry.** "manifestly equivariant under E(3)" (rotations, translations, reflections) *provided gravity is
  fed as a feature and rotated along with the scene*; the floor is canonically at z = 0 and gravity points to −z in
  the data, so the *realised* symmetry of the data distribution is E(2) in the horizontal plane (translations and
  rotations about gravity); the authors note one could use E(2) but chose E(3). [text, §3.1, §3.2]
* **Desiderata the authors state** (important for us): "a geometric problem in which the symmetries and
  representations are known and exact", a "low floor and a high ceiling" (small models score orders of magnitude
  worse than large ones), and cheaply generated synthetic data. [text, §3.1]
* **Irreducible loss.** They fit the scaling law with the offset E (irreducible loss) **fixed to 0** after
  cross-validation: "there is no clear reason to expect such an irreducible error of practically relevant size for
  the deterministic physics task". [text, §3.3]

### 1.2 Model families
* **Baseline.** Pre-LN transformer with multi-query attention; one token per mesh face; positions/velocities via
  random Fourier features (128 frequencies, σ = 0.1); no positional encoding, so it is permutation-equivariant.
  [text, §3.2, App. B]
* **Equivariant.** GATr (Brehmer et al. 2023), E(3)-equivariant via projective geometric algebra: positions as
  PGA trivectors, relative vectors and face normals as PGA vectors, velocities as PGA bivectors. [text, App. B]
* **Both** get the same two task-specific additions: hierarchical attention (half the heads masked to the same
  object) and *enforced rigidity* (per-face translation + rotation quaternion, averaged per object, applied to the
  input mesh). Rigidity "improved performance substantially". [text, §3.2]
* **Size families.** One parameter n: baseline = 2n blocks, 64n scalar channels, 2n heads; equivariant = 2n blocks,
  4n scalar + n multivector channels, 2n heads (Table 1). The equivariant model is deliberately narrower: it
  "benefits from a more narrow architecture". Hyperparameters were tuned manually for both at low and
  intermediate budgets. [text, Table 1]
* **Optimisation.** Adam, lr 5e-4 cosine (1e-3 or 2e-3 for budgets < 1e18, best of the two reported), batch 64
  (~3.5·10^5 tokens), early stopping. [text, §3.2]

### 1.3 How "scale" is defined
* **Compute** = nominal training FLOPs C ≈ ξ·N·D (N params, D training tokens), ξ = 6 for the baseline and
  ξ ≈ 61.2 for their GATr mix (pure-multivector GATr would be ξ = 6·16²/9 ≈ 171): the equivariant model does
  ~10× more FLOPs per parameter. Budgets 1e16–1e19 FLOPs, several (N, D) trade-offs per budget. [text, §3.3]
* **Model size** N: ~1e4–1e8 params (Fig. 4 axes). [fig]
* **Data** = number of *unique* training tokens, 2·10^6 to 2·10^11, at a fixed compute budget of 1e18 FLOPs and
  fixed (N, D), so fewer unique tokens = more epochs: "the lower end of this scan corresponds to training for
  6·10^5 epochs, while every sample is seen only once on the upper end". [text, §3.3]
* **Wall-clock caveat.** Nominal FLOPs ≠ wall time: "two models with the same nominal FLOP count would differ by
  as much as an order of magnitude in real training duration", the GATr implementation being the slower one
  (Fig. 6). [text, §3.3, App. C]

### 1.4 Findings, with numbers
1. **Compute scaling (infinite data, single epoch, no augmentation).** Fitted laws (E = 0):
   L_base = 1.27/N^0.909 + 0.202/D^0.379, L_equi = 2.82e-4/N^0.348 + 469/D^0.734 (Eqs. 6–7, Table 2); compute-optimal
   L*_base = 1.03/C^0.268, L*_equi = 0.14/C^0.236 (Eq. 8). "the equivariant model achieves a loss that is lower by
   approximately a factor of 2" at every budget; exponents compatible within CIs. [text, §4.1]
   Raw points read off Fig. 1 [fig]: best loss at 1e16 FLOPs 2.1e-5 (equi) vs 5.0e-5 (base); at 1e17 1.2e-5 vs
   3.1e-5; at 1e18 7.5e-6 vs 1.3e-5; at 1e19 4.9e-6 vs 8.3e-6. Ratio 1.7–2.6×.
2. **Compute-optimal size.** At 1e18 FLOPs the compute-optimal equivariant model has N* ≈ 9e4 parameters vs
   ≈ 1e6 for the baseline (from Eqs. 2, 4 with Table 2; [inferred, arithmetic]); "a compute-optimal equivariant
   transformer has less parameters than a compute-optimal baseline transformer", the gap closing at larger
   budgets (Fig. 5). [text, §4.1] **So the paper's comparison is compute-matched, not parameter-matched.**
3. **Data scaling at fixed 1e18 FLOPs (Fig. 2).** Test loss vs unique tokens, read off the figure [fig]:

   | unique tokens | baseline | baseline + augmentation | equivariant | equi/aug ratio |
   |---|---|---|---|---|
   | 2e6  | 3.0e-3 | 8.1e-5 | 8.4e-5 | 1.0 |
   | 2e7  | 3.1e-4 | 4.3e-5 | 4.1e-5 | 1.05 |
   | 2e8  | 5.5e-5 | 1.9e-5 | 1.4e-5 | 1.4 |
   | 2e9  | 1.6e-5 | 1.4e-5 | 8.1e-6 | 1.7 |
   | 2e10 | 1.3e-5 | 1.3e-5 | 7.7e-6 | 1.7 |
   | 2e11 | 1.3e-5 | 1.4e-5 | 7.9e-6 | 1.8 |

   Text: "Equivariance improves data efficiency compared to the baseline, but data augmentation can close this
   gap"; "data augmentation does not make a difference when training for a single epoch on a large dataset";
   "when training for thousands of epochs on a small dataset, data augmentation makes a baseline transformer as
   data-efficient as an equivariant model". [text, §4.2, Discussion 1]
   **Reading the numbers [inferred]:** augmentation fully closes the gap only at the two smallest datasets
   (2e6–2e7 unique tokens, i.e. ≥ ~1e4 epochs: both models are then equally data-limited at ~4–8e-5). From 2e8
   tokens upward the equivariant model keeps its ~1.4–1.8× *compute*-efficiency advantage over the augmented
   baseline. Converting unique tokens to epochs: the fixed D at 1e18 is ≈ 1.7e11 tokens by the fitted laws
   (my arithmetic) or ≈ 1.2e12 (implied by the paper's "6·10^5 epochs" at 2e6 tokens); either way the 2e9 point is
   ~100–600 epochs and the 2e8 point ~1e3–6e3 epochs. **Augmentation = equivariance only at ≥ 1e4 epochs; at
   ~100 epochs equivariance still wins by ~1.7× in loss, but that is the compute-efficiency gain of finding 2, not
   a data-efficiency gain.**
   The augmentation group used is E(2): "two-dimensional translations as well as rotations around the direction of
   gravity" (gravity is not an input to the baseline). [text, §3.3]
4. **Allocation.** For the baseline, scale tokens faster than parameters (a = 0.29); for GATr the reverse
   (a = 0.68) (Table 2, Fig. 5). Hypothesised reasons include that GATr "can express certain primitives
   particularly efficiently: the free movement and gravitational acceleration of rigid bodies can be represented
   with few multivector channels", while collision modelling needs scalar capacity. [text, Discussion 3]
5. **Stated limitations.** One task, two model families; no SOTA; nominal FLOPs; ≤ 1e19 FLOPs; "we cannot
   conclusively settle the question raised in the title". Also App. B: GATr's equivariant linear maps are *not* a
   subspace of the baseline's, so the two are not nested hypothesis classes. [text, §5, App. B]

---

## 2. The paper's preconditions, checked against our setup

| # | Condition in the paper | Paper | Ours | Satisfied? |
|---|---|---|---|---|
| C1 | Symmetry of the task is **exact and known** | E(3) on mesh dynamics, exact by construction [text] | Only camera roll (SO(2)) and the left–right mirror are exact; out-of-plane rotations do not act on the image (Howell et al. 2023, `geometric_redesign.md` §2) [repo]. The crop is a virtual camera with a 0.6–3.8° half-FOV, so "SO(3) on rays" is numerically roll + translation [repo] | **Partly**: a much smaller group; mirror exact only for bilaterally symmetric instances |
| C2 | Symmetry acts on the **raw inputs** in an exact geometric representation | Points, normals, velocities embedded as PGA primitives [text, App. B] | Inputs are pixels; GATr sees ResNet features. v4/so2: ℓ ≤ 1 circular moments of 7×7 maps; 8.95 baseline: arbitrary channel groups declared multivectors; P1: fixed cell directions + per-cell scalars [repo]. Only P1's "direction" is geometric, and it carries no information (the directions are constants; all pose information is in the non-equivariant scalars) | **No** (P1: representation geometric, content not) |
| C3 | The **whole network** is equivariant | Every layer of GATr [text] | ResNet-101 (42.5M of ~43.6M params) is not equivariant to anything; equivariant part is 0.23–1.6M params [repo]. P1's field is exactly C4 × flip equivariant *given invariant per-cell scalars*, which the backbone only approximates | **No** |
| C4 | The data distribution **exercises** the symmetry | Objects at arbitrary planar positions/headings; E(2) orbit densely sampled [text, inferred] | Roll jitter std 7.7°, max ~30° (train); test crops keep photo roll (≈ 0); pan/tilt ≤ 5°; flip 50 % [repo]. Pascal3D photos are upright | **Weakly** (the roll orbit is a ±8° sliver; only the flip is fully exercised) |
| C5 | **Data regime** where equivariance helps *beyond* augmentation | Equi > augmented baseline by 1.4–1.8× for ≥ 2e8 unique tokens (≤ ~1e3 epochs); equal at ≥ 1e4 epochs [fig] | 18,371 real images seen 100× with fresh augmentation; 2.38M renders each seen ~2.3× (epoch = 4 × real: 1 pass of real + 3 × real random synthetic draws, `packed.py`) [repo]. In epochs we sit in the paper's "equivariant still wins" band; but the win there is compute efficiency, which only applies if the compute-dominant part is equivariant (C3) | **Formally yes, materially no** |
| C6 | **Baseline augmentation** | Compute scan: none (single epoch). Data scan: E(2) augmentation, which closes the data gap at ≥ 1e4 epochs [text, fig] | DiT, 8.95 and all GATr runs share the same loader: random flip, roll jitter, bbox jitter (`use_warp`) [repo]. So our baseline already receives the exact-symmetry augmentation for the only exact symmetries we have (roll, flip) | Baseline is augmented, as in the paper's data scan |
| C7 | **Compute-matched** comparison; equivariant model has ~10× fewer params at the optimum | N*_equi ≈ 9e4 vs N*_base ≈ 1e6 at 1e18 FLOPs [inferred from Table 2] | We match parameters (1.07–1.6M vs 1.19M; 0.23 vs 0.25M). With ξ_GATr ≫ 6 our GATr heads get several times more nominal FLOPs than the DiT at equal params [inferred] | Our protocol *favours* GATr by the paper's accounting, and it still does not win |
| C8 | **Low floor, high ceiling; no irreducible loss** (E = 0) | Loss spans 3 orders of magnitude across the scan [fig] | Metric is a quantile (median) and class-mean of medians over 12 classes, two with 7 and 27 val images; ~11 % failures > 30°, a third of them 180° front/back flips; label noise (principal-point convention, ≤ 6° tilt) [repo]. All heads land in 8.6–9.4° with seed spread 0.2° | **No**: large irreducible component, compressed dynamic range |
| C9 | Output structure enforced identically for both models (rigidity) | Yes [text] | Yes: same SO(3) flow matching, Euler sampler, medoid [repo] | Yes |
| C10 | Hyperparameters tuned per family | Manual tuning of depth/width/heads/lr for both [text] | DiT: one config (dit_dim 112/48); GATr: several architectural variants explored, lr shared [repo, inferred] | Roughly comparable; if anything GATr got more tuning |

Two further mismatches worth stating:
* **The DiT baseline is conditioned on the globally pooled ResNet vector cut into 256 tokens** (commit 8ffa6db:
  "pooled vector cut into 256 tokens, as CliffordFlow without the ConvAdapter") [repo]. It has *no spatial
  information at all*, yet it matches or beats P1 (which sees all 49 cells with their positions) on class-mean
  (8.85 vs 9.35) and ties on overall median (7.71 vs 7.69). In the paper both models see identical information.
  This is the single strongest piece of evidence that the head is not where the error is made.
* **Symmetry of the target vs symmetry of the representation.** In the paper, equivariance constrains a map from
  geometry to geometry. In ours, the only exactly equivariant map is "backbone feature map → velocity", and the
  backbone map is itself only approximately covariant (ImageNet-pretrained ResNets are trained to be
  flip-*invariant*, the wrong symmetry for pose, before fine-tuning) [inferred; see Gruver et al. below].

---

## 3. Other literature on when equivariance helps (what I actually opened)

* **Brehmer et al., GATr, NeurIPS 2023 (arXiv 2305.18415; HTML, first 100k chars).** n-body: GATr "clearly
  outperforms all non-equivariant baselines" and is more sample-efficient than SEGNN/SE(3)-Transformer, which it
  matches with enough data; no numbers in the text. Arterial wall-shear stress: *canonicalising* the meshes lets
  the plain transformer nearly match GATr, i.e. the gain is largely the symmetry handling, not the algebra.
  The join is only equivariant to even transformations and needs a pseudoscalar from the inputs (the "data"
  join reference) for mirror equivariance; symmetry breaking can be added as inputs "similar to positional
  embeddings". Relevant: our `join_reference="data"` requirement [repo] and the up-token design.
* **Gerken et al., *Equivariance versus augmentation for spherical images*, ICML 2022 (arXiv 2202.03990, HTML).**
  Classification (invariant target): a 5M-param CNN with heavy augmentation reaches the 150k-param S2CNN's
  accuracy (97.49 vs 97.64 %). Segmentation (equivariant target): CNNs up to 5.5M params with 20× augmentation
  never reach the 204k S2CNN; on *unrotated* data the CNNs slightly win. Lesson: augmentation closes the gap when
  the target is invariant or the symmetry is not exercised; exact equivariance wins when the *output* must
  transform and the data spans the group. Our output does transform (pose), but the data spans only ±8° of roll.
* **Gruver et al., *The Lie derivative for measuring learned equivariance*, ICLR 2023 (arXiv 2210.02984, abs).**
  Across hundreds of pretrained CNNs/ViTs/Mixers, equivariance error falls with model size and accuracy
  "regardless of architecture"; trained transformers can be *more* equivariant than CNNs; many violations come
  from aliasing in pointwise nonlinearities. Relevant: a fine-tuned ResNet-101 + DiT can learn the ±8° roll and
  the flip from augmentation about as well as a C4-exact head, and the exact head still sits behind an aliasing
  7×7 grid.
* **Elesedy & Zaidi, *Provably strict generalisation benefit for equivariant models*, ICML 2021 (arXiv 2102.10333,
  HTML).** For a G-invariant input distribution and equivariant target, the gap between a predictor and its
  orbit average is exactly ‖f^⊥‖², the norm of the predictor's anti-symmetric part; for linear least squares
  E[gap] = σ_ξ² · dim(anti-symmetric subspace)/(n − d − 1). The benefit is (i) proportional to the dimension of the
  non-equivariant part of the hypothesis class that the constraint removes, (ii) proportional to label noise,
  (iii) inversely proportional to sample count, and (iv) requires an invariant input distribution. For us:
  constraining a ~1M-parameter head attached to a 42.5M unconstrained backbone removes a tiny fraction of the
  joint hypothesis class; and the input distribution is not roll-invariant (upright photos).
* **Wang, Walters, Yu, *Approximately equivariant networks for imperfectly symmetric dynamics*, ICML 2022
  (arXiv 2201.11969, abs).** When the symmetry is only approximate, *relaxed* equivariance beats both strict
  equivariance and no equivariance. Our mirror symmetry is approximate for real instances (asymmetric texture,
  steering wheels); exact flip equivariance in P1 may be mildly harmful.
* **Petrache & Trivedi, NeurIPS 2023 (arXiv 2305.17592, abs).** Generalisation bounds for partial/approximate
  equivariance; optimal when the model's equivariance error matches the data's. Same message.
* **Kaba & Ravanbakhsh, *Symmetry breaking and equivariant neural networks*, 2023 (arXiv 2312.09016, abs).**
  Equivariant maps cannot break symmetry at the level of a single sample. For a near-symmetric (front/back
  ambiguous) object the equivariant velocity field must be (near-)symmetric too; the flow's stochasticity and the
  medoid are what resolve it, not the architecture. Consistent with our failure analysis (a third of failures are
  180° flips) — no equivariant head changes those.
* **Klee et al., *Image2Sphere*, ICLR 2023 (arXiv 2302.13926, PDF).** SO(3)-equivariant output head on a
  non-equivariant ResNet-101 (same hybrid as ours). Pascal3D+ class-mean median 9.8° (Table 2 in Howell's
  paper reproduces it). Data-efficiency evidence: ModelNet10-SO(3) with 20 instead of 100 views per instance,
  I2S 46.3° MedErr vs Murphy et al. 59.5°; SYMSOL with 10k instead of 100k renders, I2S keeps its likelihood
  (2.98 vs 3.41) while IPDF collapses (−7.94 vs 4.10) — but with 100k renders IPDF *beats* I2S (4.10 vs 3.41).
  Table 4: spatial (per-pixel) projection matters more than how the filter is parametrised; an MLP to Fourier
  coefficients is far worse (69.2° vs 46.3°). Lesson: the equivariant head's advantage appears at 10–20 % of the
  data and *reverses* with abundant data, on synthetic renders with the full SO(3) orbit exercised.
* **Howell et al., *Equivariant single view pose prediction via induced and restricted representations*, 2023
  (arXiv 2307.03704, PDF).** The image-plane symmetry is SO(2) ⊂ SO(3) ("there is no a-priori model for how
  two-dimensional images transform under out-of-plane object rotations"); the correct head is the induced
  representation Ind_{SO(2)}^{SO(3)}. Pascal3D+ class-mean 9.2° (ResNet-101) / 10.2° (ResNet-50), SOTA in that
  table; the 10 %-data SYMSOL experiment (Table 2) shows the same data-efficiency pattern as I2S. Appendix E.2:
  inducing to S² then convolving vs inducing straight to SO(3) differ by < 0.5°. Our v4 (8.61°) and even the
  non-equivariant DiT (8.85°) already beat both equivariant SOTA numbers; the flow-matching decoder and
  training recipe, not the symmetry, account for that.
* **Vadgama et al. (Bekkers lab), *Probing equivariance and symmetry breaking in convolutional networks*, 2025
  (arXiv 2501.01999, HTML).** On QM9 generation/regression and ShapeNet, SE(3)-equivariant denoisers beat
  translation-only ones and "increased capacity does not close the gap"; adding an explicit global reference
  frame as input (symmetry breaking that matches the task geometry) consistently helps by ~1 IoU point. Relevant
  to our up-token (`g`) design: *breaking* the symmetry with the right geometric input is what helped there.
* **Ngo & Ravanbakhsh, *Scaling laws and symmetry, evidence from neural force fields*, Oct 2025 (arXiv
  2510.09768, abs).** Power-law exponents improve with the degree of equivariance in force fields; no crossover
  reported. Again a domain where the symmetry is exact on the inputs.
* **de Ruijter & Cesa, cryo-EM pose inference, 2024 (arXiv 2406.01630, abs)** and **Esteves et al., ICML 2019
  (arXiv 1812.02716, abs)**: pose-from-image with D4- / spherical-equivariant encoders; qualitative gains
  (faster, more reliable convergence; embeddings sufficient for relative pose), no data-scaling numbers.
* **Joshi, *Equivariance is dead, long live equivariance?* (blog, 2025).** AlphaFold3 and conformer-field
  diffusion dropped equivariance for augmentation; argues denoisers tolerate approximate equivariance and that
  compute/throughput, not data efficiency, decides at scale. Bekkers' rebuttal in the comments cites 2501.01999.

Not opened (cited from the paper's own bibliography only): Behboodi et al. 2022, Sannai et al. 2021, Lyle et al.
2020 (generalisation bounds); Tay et al. 2023 and Qiu et al. 2024 (inductive bias and scaling).

---

## 4. Ranked reasons our results show no gap, with cheap tests

Seed spread of the class-mean is ~0.2° (8.93/9.12 on the 8.95 recipe) [repo]; treat any difference < 0.4° as
unresolved.

1. **The equivariant part is not where the error is made (C3, C8).** The ResNet-101 is 97.5 % of the parameters
   and, by nominal FLOPs, ~65–95 % of the training compute (ResNet-101 ≈ 15.6 GFLOPs fwd ×3 × 7.35M samples ≈ 3e17;
   DiT head ≈ 3e16; P1 GATr head ≈ 2e17 with ξ ~ 60) [inferred]. The paper's compute-efficiency gain applies to the
   compute that *is* equivariant. Our failures are perception/ambiguity (11 % > 30°, a third of them 180° flips), and
   a globally pooled DiT with no spatial information matches P1. *Test:* freeze the fine-tuned ResNet from any run
   and train all heads (DiT, P1, v4) on cached feature maps for a few epochs; if they tie, the head is irrelevant.
   Cheaper: per-class and per-failure-type breakdown of DiT vs P1 vs v4 on the same val images; if the ranking is
   identical image by image, the heads are interchangeable.
2. **The symmetry the data exercises is tiny (C4), and the baseline already gets it as augmentation (C6).** Roll
   std 7.7°, test roll ≈ 0, flip 50 %. By Elesedy–Zaidi the benefit is proportional to the anti-symmetric part the
   constraint removes; by Gruver et al. and the paper's own Fig. 2 an augmented non-equivariant model learns a
   well-sampled, low-dimensional symmetry in far fewer than 1e4 epochs. *Test (zero training):* D4/flip TTA on the
   DiT, v4 and P1 checkpoints (transform the crop, map the pose back, pool 8×32 samples with the medoid). If DiT
   gains ≤ 0.2°, it is already roll/flip-equivariant in practice and there is nothing for exactness to buy.
   *Test 2:* retrain the DiT with `--no-use_warp` (no roll jitter/flip): if it loses ≥ 0.5°, the symmetry matters
   and only then does an exactly equivariant head have a job.
3. **GATr receives no geometric information to be equivariant about (C2).** In the paper the group acts on
   positions, normals and velocities that *carry the signal*. In P1 the multivector part is a constant grid of
   directions; the signal is in scalar channels the group does not touch. GATr's bilinears and join have nothing
   to multiply but the grid and the pose frame. The 8.95 recipe is worse: channel groups called multivectors. A
   geometry probe found only v4's tokens carry pose in a form one equivariant linear layer can read [repo].
   *Test:* linear-probe the pose (R e3, say) from (a) P1's input tokens, (b) the output tokens of each GATr block,
   for DiT vs GATr; and ablate P1's directions to a single constant (`s_j = e3` for all j): if the error is
   unchanged, the "geometry" was decorative. *Test 2:* P1 with 14×14 (layer3) cells, where cells span a wider
   angular range: if nothing changes, the direction tokens are not used.
4. **The task metric has a large irreducible floor and a compressed range (C8).** The paper's loss spans three
   decades with E = 0; our class-mean varies 8.6–9.4° across *every* head, with 0.2° seed noise and two classes with
   7 and 27 val images. A factor-2 loss advantage in the paper is a 1.4× RMSE advantage on a deterministic target;
   on a metric dominated by label noise and ambiguity it would be invisible. *Test:* evaluate on the subset of
   "easy" classes (bus, car, aeroplane, tv; ≥ 100 val images each) and on the error distribution's lower half
   (e.g. 25th percentile per class); if GATr heads lead there and lose on sofa/diningtable, the class-mean is
   hiding a real but small effect. Run 3 seeds of DiT and P1 before concluding anything < 0.4°.
5. **Parameter-matched instead of compute-matched (C7), and the paper's optimum is a *small* equivariant model.**
   The paper's compute-optimal GATr at 1e18 FLOPs has ~9e4 params; its advantage is in loss per FLOP, with ~10× more
   FLOPs per parameter. Our P1-small (0.23M) vs DiT-small (0.25M) is the closest analogue and shows the same
   non-result (9.41 vs 9.09), with P1 again getting more nominal FLOPs. *Test:* measure actual head FLOPs
   (fvcore/torch profiler, one forward each, < 1 min CPU) and wall-clock per epoch; report the comparison at matched
   head FLOPs, which likely means a 2–4× wider DiT. If the DiT at matched FLOPs wins clearly, the paper's framing
   argues *against* GATr here.
6. **Approximate and wrong-direction symmetries (Wang et al., Petrache–Trivedi).** The mirror is exact only for
   bilaterally symmetric instances; the ImageNet-pretrained backbone starts flip-*invariant*; v4's `e12/e123`
   slots and the vector-dual read-out are not flip-covariant [repo]. Exact flip equivariance in P1 may be
   enforcing a symmetry the real data breaks. *Test:* measure the fine-tuned backbone's flip covariance error on
   val (‖f(flip x) − flip f(x)‖/‖f(x)‖ on the 7×7 map; one pass, < 1 min on cached features), and P1 with flip
   augmentation off vs on; and a P1 variant with a learned "handedness" pseudoscalar token (relaxed
   equivariance) — if it helps, strictness was the cost.
7. **Optimisation, not expressivity, in the GATr heads.** The paper notes equivariant transformers may lack mature
   initialisation and sit at different width/depth optima; our own history shows GATr variants stalling 17 epochs
   at a saddle and needing `--so2_split_norm` and up tokens to start [repo]. *Test:* compare training-loss curves
   and the epoch at which each head reaches 95 % of its final val accuracy; try P1 with lr ×2 and a narrower/deeper
   config (the paper's GATr prefers narrow), since the paper reports larger lr helped small budgets.
8. **Symmetry breaking is what helps, not symmetry (Vadgama et al.).** Our best run (v4, 8.61) is the one that
   *breaks* roll equivariance with an up token; P1 with exact equivariance is the worst. *Test:* P1 + the loader's
   exact up vector `g` as a transforming token (already designed in `geometric_redesign.md` §3.1) vs P1 with a
   constant `−e2` vs none; the ordering tells whether the prior or the symmetry is doing the work.

**What would count as the paper's effect showing up here.** A gap that (a) grows when training data is reduced
(e.g. 10 % of real images, no synthetic — the I2S/Howell SYMSOL pattern), (b) survives when the baseline is given
the same augmentation, and (c) is present at matched head FLOPs. A single cheap run that tests (a): both heads on
10 % of real data without renders, 3 seeds. If P1 does not beat DiT by ≥ 1° there, equivariance in the head has no
data-efficiency effect on Pascal3D+ either, and the question is closed on this dataset.

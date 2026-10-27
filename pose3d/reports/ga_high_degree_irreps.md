# Can geometric-algebra networks learn high-degree irreducible representations (l ≥ 6)?

Literature survey, 2026-10-25. Scope: every paper found that discusses whether multivector (Clifford /
geometric algebra, GA) networks such as CGENN, GATr, GLGENN, Clifford-steerable CNNs can carry or
learn SO(3) irreducible representations of degree l ≥ 2, plus the papers that supply a fix.

## Notation

* **Irreducible representation of degree l** (written `l` below): the (2l+1)-dimensional space on which
  3D rotations act through the Wigner-D matrix `D^l`. `l = 0` is a scalar, `l = 1` a vector, `l = 2` a
  symmetric traceless rank-2 tensor (quadrupole), `l = 6` a 13-dimensional space. Spherical-harmonics
  networks (e3nn, TFN, SEGNN, MACE, Image2Sphere) keep features of every degree up to a cutoff `lmax`.
* **Cl(3,0)**: the 8-dimensional Clifford algebra of 3D space; a multivector has grades 0 (scalar),
  1 (vector), 2 (bivector), 3 (trivector). **Cl(3,0,1)** is the 16-dimensional projective algebra used
  by GATr; its extra null generator `e0` encodes translations. **Cl(4,1)** is the 32-dimensional
  conformal algebra of C-GATr.
* **Steerable feature**: a hidden-layer quantity that transforms linearly under rotations as some `l`.
  "Carrying `l = 6`" means the hidden state contains a 13-dimensional block transforming as `D^6`.

## Short answer

**No, as hidden features.** Every GA network built on 3D generators (CGENN, GATr in Cl(3,0,1), E-GATr,
C-GATr, GLGENN, Clifford-steerable CNNs, Multivector Neurons, CG-EGNN, Clifford sheaf networks) has
hidden states that decompose under SO(3) into copies of `l = 0` and `l = 1` only. No grade of any of
these algebras contains an `l ≥ 2` block, and equivariant linear layers are grade-wise, so no linear
readout can produce an `l = 6` output. The degree-6 content can only be reached *nonlinearly*, as a
polynomial of degree 6 in vector features, which costs depth and is not trained for by any published
GA architecture.

**Yes, in principle as a function.** Geometric products of vectors generate polynomial invariants and
covariants of unbounded degree, and invariant theory (Villar et al. 2021, Domina et al. 2025) shows
that any `l = 6` equivariant output can be written as `Σ_i f_i(invariants) · (v_{i1} ⊗ … ⊗ v_{i6})_6`,
i.e. scalar functions times the degree-6 symmetric-traceless product of six vectors. A GA network can
produce the `f_i` and the `v_i`, but the final contraction to `l = 6` must be added by hand.

**Fixable: yes**, with one of three documented constructions (section "Fixes"). None has been tested at
`l = 6` inside a GA network; the published hybrid (CliffordSTF) stops at `l = 3`.

## Why (the argument, assembled from the papers below)

1. Under SO(3), grade 0 and grade 3 of Cl(3,0) are `l = 0`; grade 1 and the Hodge dual of grade 2 are
   `l = 1`. So Cl(3,0) = 2·(l=0) ⊕ 2·(l=1). Under O(3) the four grades are 0e, 1o, 1e, 0o (e/o = even/odd
   parity). [GLGENN footnote 2; CliffordSTF §3.1 and App. A.1; CSNN §5]
2. Adding rotation-invariant generators (the `e0` of Cl(3,0,1), or the `e+`, `e−` of Cl(4,1)) only
   multiplies these blocks: Cl(3,0,1) = 4·(l=0) ⊕ 4·(l=1). Grade k of the algebra is the exterior power
   Λ^k(R^3 ⊕ R^m) = ⊕ Λ^a(R^3) ⊗ Λ^b(R^m), and Λ^a(R^3) is only ever `l = 0` or `l = 1`. [CliffordSTF
   App. A.3 states the absence of `l ≥ 2` for GATr; the count 4 ⊕ 4 is ours]
3. Equivariant linear maps between multivectors are grade projections (plus `e0`-multiplications in
   PGA). [GATr Prop. 1; CGENN eq. 13; de Haan et al. App.] By Schur's lemma there is no non-zero
   equivariant linear map from a sum of `l ≤ 1` blocks onto `l = 6`. Hence no GA layer, however wide,
   can hold or emit an `l = 6` steerable feature.
4. The geometric product of two vectors realises only the `l = 0` and `l = 1` parts of
   `1 ⊗ 1 = 0 ⊕ 1 ⊕ 2`; the `l = 2` part is projected away. `l ≥ 2` content therefore only appears
   nonlinearly, as scalar coefficients that depend on higher-degree invariants, after stacked products.
   [CliffordSTF §1, §3.1; CS-CNN App. B shows the planar analogue: the angular-frequency-2 kernel
   component is missing from one Clifford-steerable convolution and reappears only after composing two]
5. Empirically the gap matters when the target has `l ≥ 2` structure: all `l ≤ 1` models, Clifford
   ones included, get force *directions* wrong on rMD17 (force-cosine similarity < 0.25) until an
   `l = 2, 3` track is added. [CliffordSTF Table/abstract; survey §VI-D lists this as a structural
   failure mode of GA networks]

## Papers that discuss the question directly

| Paper | Link | What it says about `l ≥ 2` in GA networks |
|---|---|---|
| Polat, Serpedin, Kurban, Kurban, *Geometric Algebra Meets Cartesian Tensors: Higher-Order Equivariance for Interatomic Potentials* (2026) | https://arxiv.org/abs/2606.29584 | The only paper that names the limitation as a theorem-level statement: Cl(3,0) and Cl(3,0,1) features are 2·(l=0) ⊕ 2·(l=1); the per-edge geometric product cannot reach `l = 2` "regardless of depth"; higher grade ≠ higher `l`. Fix: CliffordSTF, a parallel symmetric-traceless tensor track at ranks 2 and 3 (`lmax = 3`), coupled bilinearly to the multivector track; raises force-cosine from 0.055 to 0.551. Higher `lmax` "through iterated STF constructions" is left as future work. |
| Zhong, Fong, Wang, Allen-Blanchette, *Geometric Algebra for Neural Networks: A Survey* (Oct 2026) | https://arxiv.org/abs/2610.30121 | §VI-D "Failure modes": "Expressivity can also be structurally limited: vector–vector geometric products in G3,0,0 lack the rank-2 (L = 2) component", citing CliffordSTF. Table 3 contrasts GA latent representations with steerable/TFN irreps. |
| Kamiya, Nicholls, *Clifford Sheaf Neural Networks* (2026) | https://arxiv.org/abs/2610.01322 | §5 "First-order representational placement": grades 0, 3 are spin-0, grades 1, 2 spin-1; the model "operates entirely with L = 0 and L = 1 representations and carries no rank-two tensors", placing all Clifford models with PaiNN rather than Equiformer. Also shows that sandwich maps reach only half of the linear endomorphisms of Cl(3,0). |
| Zhdanov, Ruhe, Weiler, Lucic, Brandstetter, Forré, *Clifford-Steerable CNNs* (ICML 2024) | https://arxiv.org/abs/2402.14730 | Limitations list: multivectors do not cover all field types ("general tensor fields"); grades may not be irreducible; the steerable kernel space is "not necessarily complete". App. B / Fig. 8: compared with the complete e2cnn basis, a Clifford-steerable kernel misses the angular-frequency-2 vector→vector kernel; it is recovered by composing two convolutions. |
| Filimoshina, Shirokov, *GLGENN* (ICML 2025) | https://arxiv.org/abs/2506.09625 | Footnote 2 states the grade ↔ irrep dictionary (scalar = 0e, vector = 1o, bivector = 1e, trivector = 0o) and calls the bridge to irreps-based models an open topic. |
| de Haan, Cohen, Brehmer, *Euclidean, Projective, Conformal: Choosing a Geometric Algebra for Equivariant Transformers* (AISTATS 2024) | https://arxiv.org/abs/2311.04744 | §4.1: Conjecture 2 (numerically verified to 4 inputs) that EGA/CGA geometric products + grade projections express every equivariant *multilinear map between multivectors*; PGA needs the join. Expressivity is only analysed within the multivector space, i.e. within `l ≤ 1`; universality is left open. |
| Brehmer, de Haan, Behrends, Cohen, *Geometric Algebra Transformer* (NeurIPS 2023) | https://arxiv.org/abs/2305.18415 | Prop. 1: equivariant linear maps are grade projections and `e0`-multiplications (9 parameters). Discussion: "GATr is not yet shown to be a universal approximator". In the scaling comparison (App. C.4) SEGNN / SE(3)-Transformer were run with `l ≤ 1`, so that comparison never probed `l ≥ 2`. |
| Ruhe, Brandstetter, Forré, *Clifford Group Equivariant Neural Networks* (NeurIPS 2023) | https://arxiv.org/abs/2305.11141 | Positions multivectors as an alternative to irreps: "a product structure while maintaining a finite dimensionality as opposed to tensor product representations". All polynomials in multivectors are equivariant; higher-order interactions "indirectly modeled via multiple successive layers". No `l ≥ 2` features. |
| Polly, *When Do Geometric Algebra Layers Beat Scalarization?* (2026) | https://arxiv.org/abs/2607.06634 | Controlled study on `l ≤ 1` targets only. Finds GA layers help only when the target composes rotors in depth; explicitly states its tasks pose "no symmetry-induced expressivity obstruction", i.e. it does not test `l ≥ 2`. |
| Tran, Vo, Huu, Nguyen, *A Clifford Algebraic Approach to E(n)-Equivariant High-order GNNs* (2024) | https://arxiv.org/abs/2410.04692 | Proves universality of k-hop message passing for *permutation* structure only (Remark 5.3: the Clifford part adds orthogonal equivariance but the theorem is not about it). "High-order" here means higher-order neighbourhoods, not higher `l`. |
| Favaro, Plehn, Qu, Spinner, *Virtues and Vices of Equivariant Transformers* (2026) | https://arxiv.org/abs/2608.02735 | Lorentz analogue: L-GATr (full 16-dim multivector, includes the antisymmetric rank-2 bivector) and L-GATr-slim (scalars + vectors only) perform identically on top tagging, "which indicates that the antisymmetric rank-2 tensor representation does not help". Evidence that the extra grades of GA do not act as higher-order irreps in practice. |

Papers checked that mention "higher-order" GA features but mean grades, not degrees `l`: Liu et al.,
*Clifford Group Equivariant Simplicial Message Passing Networks* (https://arxiv.org/abs/2402.10011);
Liu, Ruhe, Forré, *Multivector Neurons* (https://arxiv.org/abs/2406.04052); Ruhe et al., *Geometric
Clifford Algebra Networks* (https://arxiv.org/abs/2302.06594). The GATr OpenReview discussion could not
be read (login wall).

## Theory on what degree is needed and whether depth can substitute for `lmax`

| Paper | Link | Relevance |
|---|---|---|
| Villar, Hogg, Storey-Fisher, Yao, Blum-Smith, *Scalars are universal* (NeurIPS 2021) | https://arxiv.org/abs/2106.06610 | Any O(3)-equivariant vector function is `Σ_j f_j(dot products) v_j`. This is why `l ≤ 1` GA nets are universal for scalar/vector outputs given enough depth, and why the obstruction is specific to `l ≥ 2` outputs. |
| Domina, Bigi, Pegolo, Ceriotti, *Representing spherical tensors with scalar-based machine-learning models* (2025) | https://arxiv.org/abs/2505.05404 | Generalises the above to any `l` (called λ): an `l`-tensor of vector inputs equals scalar functions times "maximal couplings" `(r_{i1} ⊗ … ⊗ r_{il})_l`; with three independent vectors `2l+1` scalar functions suffice (eq. 70). Also documents the failure at symmetric configurations where the vectors degenerate, and a correction. This is the mathematical recipe for turning GA (scalar + vector) outputs into `l = 6`. |
| Dym, Maron, *On the Universality of Rotation Equivariant Point Cloud Networks* (ICLR 2021) | https://arxiv.org/abs/2010.02449 | TFN-type nets are universal only with unbounded `lmax`; the mechanism is that degree-`l` irreps express degree-`l` polynomials. |
| Hordan, Derksen, Petrache, Dym, *Degree Bounds for Separating Polynomial and Neural Invariants for Point Clouds* (Oct 2026) | https://arxiv.org/abs/2610.26612 | Separation of n points needs invariant polynomials of degree ~n²; TFN/MACE need `lmax` ~ n²; Prop. 12: bounded `lmax` and body order cannot separate counter-rotating rings. For GA nets the equivalent resource is depth (products double the degree per layer), which no paper has quantified. |
| Pacini et al., *On Universality Classes of Equivariant Networks* (2025) and *On Universality of Deep Equivariant Networks* (2025) | https://arxiv.org/abs/2506.02293, https://arxiv.org/abs/2510.15814 | Depth and readout layers decide universality for permutation representations; the framework does not yet cover continuous groups or multivector features, so there is no theorem either way for CGENN/GATr. |

## Fixes

1. **Hybrid irreps track.** Keep the multivector track and add explicit higher-`l` channels, coupled
   bilinearly. CliffordSTF (https://arxiv.org/abs/2606.29584) does this with closed-form symmetric
   traceless Cartesian tensors (STF) at ranks 2 and 3, verified against Wigner-3j symbols, and says an
   STF4 extension "is straightforward". Rank-6 STF tensors (13 components) are available in closed form
   through the irreducible-Cartesian-tensor decompositions of Shao et al. (rank up to 9,
   https://arxiv.org/abs/2412.18263), ICTP (https://arxiv.org/abs/2405.14253), CACE
   (https://arxiv.org/abs/2402.07472), HotPP (https://arxiv.org/abs/2402.15286) and TensorNet
   (https://arxiv.org/abs/2306.06482); alternatively an e3nn head with `lmax = 6` fed by the GA net's
   scalars and vectors. Cost: a second feature type, parity bookkeeping (CliffordSTF is only SO(3), not
   O(3), equivariant; App. A.4).
2. **Maximal-coupling readout.** Let the GA net emit `k ≥ 3` vectors `q_α` and `2l+1 = 13` scalars
   `f`, and form `T_6 = Σ f · (q_{α1} ⊗ … ⊗ q_{α6})_6` as in Domina et al. (eq. 70 for `l ≥ 3`). Exactly
   equivariant, no Clebsch–Gordan tables inside the network, but it degenerates when the output vectors
   become linearly dependent (symmetric inputs; Domina §III-D gives a correction), and the degree-6
   dependence has to be learned by the scalar coefficients.
3. **Implicit scalar-field readout** (IPDF style). Feed the query direction or query rotor into the GA
   net as an input multivector and read a *scalar* score `s(x, n)`; equivariance is automatic and the
   degree-`l` content of `s(x, ·)` is obtained by quadrature on the sphere or on SO(3). A Wigner-D
   entry `D^l(R)` is a polynomial of degree `2l` in the rotor components, so resolving `l = 6` needs a
   degree-12 polynomial, i.e. about four geometric-product layers with the query rotor as input
   (degree doubles per product layer). This is the only fix that needs no new feature type, at the
   price of one network evaluation per query.

What depth alone buys: with the query as input, the GA net is universal in the sense of Villar et al.,
so `l = 6` is reachable; without a query input, the network cannot output `l = 6` at all, whatever its
depth.

## Relevance to pose3d

Clifford Flow predicts a vector field on SO(3) given the current rotor; the field value is an `l ≤ 1`
object (a bivector / tangent vector), so the flow formulation already sidesteps the obstruction: the
high-frequency structure of the pose distribution is encoded in the nonlinear dependence of the field
on the sample rotor, which is fix 3 in disguise. Image2Sphere (https://arxiv.org/abs/2302.13926) and
the Wigner-D regressor (https://arxiv.org/abs/2411.00543) instead carry `lmax = 6` steerable features
and read them out linearly. If an experiment needs a GATr backbone to *output* an SO(3) distribution in
Fourier form (Wigner-D coefficients to `l = 6`) rather than a flow, fix 1 (an e3nn or STF head) or fix 2
is required; a linear GATr readout cannot produce those coefficients. No paper has measured how much
angular resolution a trained GATr reaches through depth, so that is an open experiment.

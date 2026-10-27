# Clifford / geometric algebra deep learning: paper catalogue

Generated from `literature/papers.yaml` on 2026-10-27 by `python -m ga_lit render`. Do not edit by hand; edit the YAML or use the CLI.

**145 accepted papers** (50 core, 43 adjacent, 52 context), 0 waiting in the inbox. Browse interactively: `literature/site/index.html`.

Relevance: **core** = geometric algebra inside the model or method; **adjacent** = GA-related machine learning or a GA method next to a learned component; **context** = cites or benchmarks against GA models without using one.

## 2026

### Surveys, reviews and position papers

Overviews of geometric-algebra (GA) neural networks and essays arguing for GA as a representation.

- **Geometric Algebra for Neural Networks: A Survey** — Zhong, Fong, Wang, Allen-Blanchette. *arXiv, 2026-10-21.* [arxiv](https://arxiv.org/abs/2610.30121) · [code](https://github.com/CAB-Lab-Princeton/AwesomeGANN)
  First full survey of GA-native neural networks (Princeton / Stanford); its companion AwesomeGANN list is the best single starting point. — also: Software and resources
- **A Comprehensive Review of CliffordNet: Geometric Algebra as the Foundation for Efficient Vision Architectures** *(context)* — Mamun. *Zenodo, 2026-09-01.* [url](https://zenodo.org/records/22992570)
  Secondary review of a single paper (CliffordNet); low weight. — also: Vision, remote sensing and medical imaging
- **One algebra for all: geometric algebra methods for neurosymbolic XR scene authoring, animation and neural rendering** *(adjacent)* — Kamarianakis, Protopsaltis, Papagiannakis. *Philosophical Transactions of the Royal Society A, 2026-08-13.* [doi](https://doi.org/10.1098/rsta.2025.0109)
  Position paper on GA for neural rendering and generative-AI scene editing in extended reality.
- **Merits of geometric algebra applied to cryptography and machine learning** — Silva. *Philosophical Transactions of the Royal Society A, 2026-08-13.* [doi](https://doi.org/10.1098/rsta.2025.0107)
  Clifford fully homomorphic encryption plus "geometric neural networks" that classify encrypted Cl(3,0) point clouds with multivector weights instead of matrices. — also: New GA / Clifford architectures
- **Review of Clifford algebra-based AI methods in signal analysis** — Byrtus, Prinzi, Vašík, Vitabile. *Philosophical Transactions of the Royal Society A 384(2326), 2026-08-13.* [doi](https://doi.org/10.1098/rsta.2025.0099)
  Review of Clifford-algebra neural methods for signals with a medical-imaging focus; part of the theme issue "Modern applications of geometric algebra". — also: Vision, remote sensing and medical imaging
- **Overview of W. K. Clifford's geometric algebras** *(context)* — Hitzer. *Philosophical Transactions of the Royal Society A, 2026-08-13.* [doi](https://doi.org/10.1098/rsta.2025.0094)
  Background overview covering the algebras used in deep learning (projective, conformal, spacetime algebra).
- **Geometric Structures for Learning and Optimization in Robotics** *(adjacent)* — Calinon. *Annual Review of Control, Robotics, and Autonomous Systems, 2026-05-05.* [doi](https://doi.org/10.1146/annurev-control-022624-013742)
  Review of geometric approaches to robot skill learning; geometric algebra is one of its three frameworks. — also: Robotics, driving and control learning
- **Toward a Functional Geometric Algebra for Natural Language Semantics** — Pustejovsky. *arXiv, 2026-04-28.* [arxiv](https://arxiv.org/abs/2604.25902)
  Argues that Clifford algebras should replace linear algebra as the substrate of semantic representations in neural language models. — also: Sequence models, language and LLM tooling
- **Advanced Graph Neural Networks for Smart Mining: A Systematic Literature Review of Equivariant, Topological, Symplectic, and Generative Models** *(context)* — Rojas, Jorquera, Garcia. *Mathematics 14(5):763, 2026-02-25.* [doi](https://doi.org/10.3390/math14050763)
  PRISMA review whose scope explicitly includes Clifford (geometric algebra) graph neural networks. — also: Graphs, multimodal and representation learning
- **Symmetry group equivariant convolutions for representation learning: a survey** *(context)* — Basheer, Mishra. *International Journal of Data Science and Analytics 22(1) (online 2025-12-29), 2026-01-01.* [doi](https://doi.org/10.1007/s41060-025-00994-7)
  General equivariance survey listed by the October 2026 GA survey.

### New GA / Clifford architectures

General-purpose layers and models whose weights or activations are multivectors.

- **Clifford Sheaf Neural Networks** — Kamiya, Nicholls. *arXiv, 2026-10-01.* [arxiv](https://arxiv.org/abs/2610.01322)
  Cellular-sheaf network whose stalks are Clifford algebras; K-term sandwich restriction maps give an equivariant, grade-mixing sheaf Laplacian. — also: Graphs, multimodal and representation learning
- **Projective Geometric Algebra Graph Neural Networks: Edge-Modulated Message Passing via the Geometric Product** — anonymous. *TMLR submission, 2026-09-19.* [openreview](https://openreview.net/forum?id=zA86PE77B9)
  Strictly E(3)-equivariant message passing where edge messages are geometric products in projective GA; a local alternative to GATr's global attention. — also: Graphs, multimodal and representation learning
- **TS-CAN: Time Series Forecasting via the Clifford Geometric Product** — anonymous. *ICLR 2027 submission, 2026-09-19.* [openreview](https://openreview.net/forum?id=ruus1Sc1Uq)
  Replaces dot-product interactions in time-series forecasters with the geometric product (in-phase plus quadrature components). — also: Sequence models, language and LLM tooling
- **Weyl Algebra Equivariant Neural Networks** *(context)* — anonymous. *ICLR 2027 submission, 2026-09-19.* [openreview](https://openreview.net/forum?id=IFp4LqmXqP)
  Not Clifford-based, but positioned explicitly against Clifford-algebra equivariance (symplectic / differential-operator symmetries Clifford algebras cannot express). — also: Theory and analysis of GA layers
- **Rotors: Lightweight Geometric Recurrence for Edge Intelligence** — anonymous. *ICLR 2027 submission (desk rejected), 2026-09-18.* [openreview](https://openreview.net/forum?id=wMspnEY5us)
  Sequence model whose state is rotated on cooperating SO(D) manifolds; O(1) inference memory. — also: Sequence models, language and LLM tooling
- **Wider Groups, Fewer Weights: Efficient Pseudo-Orthogonal Equivariance via Geometric Algebras** — anonymous. *ICLR 2027 submission, 2026-09-18.* [openreview](https://openreview.net/forum?id=chx1AMpGVP)
  Equivariant layers for groups larger than the Clifford group that act identically to O(p,q), trading expressiveness for far fewer parameters. — also: Theory and analysis of GA layers
- **Algebraic Interaction: A Recipe for Neural Architecture Generation** — anonymous. *ICLR 2027 submission, 2026-09-17.* [openreview](https://openreview.net/forum?id=8RRovMh8Vr)
  Swaps the "feature algebra" of linear attention, Mamba and equivariant networks; Clifford algebras are one instantiation and improve linear attention accuracy.
- **Path-Locked Neural Composition: Exact Evaluation-Path Invariance with Geometric Algebra** — anonymous. *ICLR 2027 submission, 2026-09-16.* [openreview](https://openreview.net/forum?id=bkIXwcoMQ2)
  Learned motion calibration composed with exact projective-GA motor products; visual odometry on ETH3D. — also: Robotics, driving and control learning
- **Hypercomplexification of U-Net by Clifford Algebra Embeddings** — Byrtus, Prinzi, Vašík, Vitabile. *LNCS (ENGAGE 2025 proceedings), 2026-06-04.* [doi](https://doi.org/10.1007/978-3-032-22539-9_5)
  U-Net with Clifford-algebra embedding blocks; breast-cancer segmentation on dynamic contrast-enhanced MRI. — also: Vision, remote sensing and medical imaging
- **Conditional Clifford-Steerable CNNs for PDE Modeling** — Szarvas, Zhdanov. *ICML 2026 (PMLR 306; arXiv preprint Oct 2025), 2026-04-30.* [arxiv](https://arxiv.org/abs/2510.14007) · [openreview](https://openreview.net/forum?id=LJ8rxwl9N6)
  Clifford-steerable convolutions whose kernels are conditioned on the input multivector field; O(p,q) / E(p,q) equivariant PDE surrogates. Preprint 2025, published 2026. — also: Scientific machine learning
- **Versor: A Geometric Sequence Architecture** — Huy, Hirst. *arXiv, 2026-02-10.* [arxiv](https://arxiv.org/abs/2602.10195) · [code](https://github.com/VersorAI/Versor)
  Conformal GA Cl(4,1) states evolved by rotors; Recursive Rotor Accumulator (linear-time recurrence) and Geometric Product Attention; custom bit-masked Clifford kernels. — also: Sequence models, language and LLM tooling
- **Clifford Kolmogorov-Arnold Networks** — Wolff, Alesiani, Duhme, Jiang. *arXiv; IJCNN 2026, 2026-02-05.* [arxiv](https://arxiv.org/abs/2602.05977) · [doi](https://doi.org/10.1109/ijcnn69854.2026.11728617)
  Kolmogorov-Arnold network layers with Clifford-valued spline functions; randomized quasi-Monte-Carlo grids tame the 2^n blow-up of higher-dimensional algebras.
- **CliffordNet: All You Need is Geometric Algebra** — Ji. *arXiv, 2026-01-11.* [arxiv](https://arxiv.org/abs/2601.06793) · [code](https://github.com/ParaMind2025/CAN)
  Vision backbone where the geometric product uv = u·v + u∧v is the only mixing operator; linear complexity; 77.8% CIFAR-100 with 1.4M parameters. — also: Vision, remote sensing and medical imaging

### Theory and analysis of GA layers

Expressivity, identifiability, group-theoretic and optimisation studies of GA networks.

- **Rotor Recurrences: What Compact Recurrent States Can Predict** — anonymous. *ICLR 2027 submission, 2026-09-18.* [openreview](https://openreview.net/forum?id=pqFoZLIy1e)
  Sharp prediction limits for rotor-transition recurrences in three and four dimensions. — also: Sequence models, language and LLM tooling
- **When Do Geometric Algebra Layers Beat Scalarization? A Controlled Study on SO(3)-Equivariant Vector Laws** — Polly. *arXiv, 2026-07-07.* [arxiv](https://arxiv.org/abs/2607.06634)
  Controlled comparison of Cl(3,0) networks against invariant-scalar MLP baselines; GA wins by an order of magnitude only when the target composes group elements in depth.
- **Grade Discovery as an Identifiability Diagnostic for Clifford-Valued Features** — anonymous. *TMLR submission (rejected), 2026-05-26.* [openreview](https://openreview.net/forum?id=8Q9Xmttlum)
  Tests whether the grade (scalar / vector / bivector / pseudoscalar) of a feature channel can be recovered from paired observations.
- **Unveiling Hidden Convexity in Deep Learning: A sparse signal processing perspective** *(adjacent)* — Zeger, Pilanci. *IEEE Signal Processing Magazine, 2026-05-01.* [doi](https://doi.org/10.1109/msp.2026.3677783)
  Shows how geometric-algebra (wedge-product) structure emerges in convex reformulations of deep networks. — also: Surveys, reviews and position papers
- **Standard Attention as a Small-Angle Limit of Riemannian Geometric Algebra Transformers** — Joseph. *Research Square, 2026-02-11.* [doi](https://doi.org/10.21203/rs.3.rs-8811263/v1)
  Dot-product attention derived as the small-angle limit of rotor flow on Spin(3) inside Cl(3,0). — also: New GA / Clifford architectures
- **On Lie Groups Preserving Subspaces of Degenerate Clifford Algebras** *(adjacent)* — Filimoshina, Shirokov. *Advances in Applied Clifford Algebras; arXiv, 2026-01-10.* [arxiv](https://arxiv.org/abs/2601.07191) · [doi](https://doi.org/10.1007/s00006-025-01431-5)
  Lie groups in degenerate (projective-GA-type) algebras, motivated by building equivariant networks.

### Scientific machine learning

Molecules, materials, PDE surrogates, fluids, geoscience and lattice field theory.

- **Learning Low-Order Approximations of the Quark Propagator in Lattice QCD** *(adjacent)* — anonymous. *ICLR 2027 submission, 2026-09-16.* [openreview](https://openreview.net/forum?id=GuDDklAx8s)
  Gauge-equivariant networks expressed in the Clifford (gamma-matrix) basis for scientific discovery of propagator expansions. — also: High-energy physics
- **A Geometric Algebra-Inspired Spatio-temporal Learning Model for Sea Level Variability Reconstruction Under Incomplete Observations** — Li, Huang, Huang, Luo et al.. *Advances in Applied Clifford Algebras, 2026-09-11.* [doi](https://doi.org/10.1007/s00006-026-01457-3)
  GA-inspired spatio-temporal learning model for reconstructing sea-level fields from incomplete observations. — also: Sequence models, language and LLM tooling
- **CliffordIP: Clifford algebra equivariant interatomic potentials for heterogeneous catalysis** — Polat, Serpedin, Kurban, Kurban. *npj Computational Materials, 2026-08-04.* [doi](https://doi.org/10.1038/s41524-026-02259-8)
  Message-passing potential on Cl(3,0) 8-dimensional multivectors, O(3)-equivariant via Pin(3); 20-24% lower energy error on catalysis subsets of OC20 / OC22.
- **Geometric Algebra Meets Cartesian Tensors: Higher-Order Equivariance for Interatomic Potentials** — Polat, Serpedin, Kurban, Kurban. *arXiv, 2026-06-28.* [arxiv](https://arxiv.org/abs/2606.29584)
  CliffordSTF couples Clifford multivectors to symmetric-traceless tensor tracks to fix the poor force directions of L <= 1 Clifford potentials.
- **Geometric Algebra-Based Time Series Reconstruction of Sea Level Variability - A Case Study Along the Northeast Pacific Coast (1993-2023)** — Huang, Li, Luo, Yu et al.. *LNCS (ENGAGE 2025 proceedings), 2026-06-04.* [doi](https://doi.org/10.1007/978-3-032-22539-9_9)
  Companion case study of the GA sea-level reconstruction model. — also: Sequence models, language and LLM tooling
- **When Does Equivariance Help? Canonical Alignment in Neural Fluid Surrogates** — Rygiel, Suk, Yeung, Brune et al.. *arXiv; ICLR 2027 submission, 2026-05-12.* [arxiv](https://arxiv.org/abs/2605.18816) · [openreview](https://openreview.net/forum?id=VLMVwDLU6q)
  Compares GATr-type equivariant surrogates with canonically aligned non-equivariant ones on vascular CFD meshes.
- **Geometric deep learning for local growth prediction on abdominal aortic aneurysm surfaces** — Alblas, Rygiel, Suk, Kappe et al.. *Artificial Intelligence in Medicine 172:103323, 2026-02-01.* [doi](https://doi.org/10.1016/j.artmed.2025.103323)
  Projective-GA (GATr-based) model predicting local growth on aneurysm surface meshes. — also: Vision, remote sensing and medical imaging
- **Design and Evaluation of a Geometric Algebra-Based Graph Neural Network for Molecular Property Prediction** — Petersen, Schmidt. *NLDL 2026 (PMLR 307), 2026-01-19.* [url](https://proceedings.mlr.press/v307/petersen26a.html)
  GA-based graph neural network evaluated on molecular property prediction at the Northern Lights Deep Learning conference. — also: Graphs, multimodal and representation learning

### High-energy physics

The Lorentz-equivariant geometric algebra transformer (L-GATr) line and other spacetime-algebra models.

- **Sequential SBI for Resonance Searches at the LHC** *(context)* — Bahl. *arXiv, 2026-10-21.* [arxiv](https://arxiv.org/abs/2610.30245) · [inspire](https://inspirehep.net/literature/3217957)
  Cites L-GATr.
- **Lorentz-Equivariant Geometric Algebra Transformers for Top-Quark Event Reconstruction** — Abasov, Dudko, Markina, Perfilov et al.. *arXiv, 2026-10-21.* [arxiv](https://arxiv.org/abs/2610.29678) · [inspire](https://inspirehep.net/literature/3217369)
  L-GATr on ttbar/tWb separation, FCNC detection and full event reconstruction; companion code repository.
- **How Far Can Foundation Models Be Distilled for Efficient Jet Tagging?** — Agarwal, Khare, Kumar. *arXiv, 2026-10-21.* [arxiv](https://arxiv.org/abs/2610.29439) · [inspire](https://inspirehep.net/literature/3217213)
  Distils foundation models into L-GATr-slim students; a 20k-parameter student beats every published tagger trained from scratch.
- **This is SParTA: A modular, efficient and scalable transformer architecture for jet-tagging** *(context)* — Baruah. *arXiv, 2026-10-19.* [arxiv](https://arxiv.org/abs/2610.27106) · [inspire](https://inspirehep.net/literature/3216611)
  Benchmarks against L-GATr.
- **QGrAT: Quantum Grade-Aware Transformer for Jet Tagging** — anonymous. *arXiv, 2026-10-12.* [arxiv](https://arxiv.org/abs/2610.17351) · [inspire](https://inspirehep.net/literature/3214450)
  Quantum transformer whose features are organised by the grades of the spacetime Clifford algebra. — also: Other machine-learning uses of GA
- **MadNIS, Ready for Production at LO** *(context)* — Heimel. *arXiv, 2026-10-12.* [arxiv](https://arxiv.org/abs/2610.17177) · [inspire](https://inspirehep.net/literature/3214424)
  Cites L-GATr.
- **RINO: Renormalization Group Distillation with No Labels** *(context)* — Hao. *arXiv, 2026-10-11.* [arxiv](https://arxiv.org/abs/2610.15919)
  Cites L-GATr.
- **A compact, interrogable Tree Tensor Network Jet Tagger on Lund declustering trees** *(context)* — Napolitano. *arXiv, 2026-10-08.* [arxiv](https://arxiv.org/abs/2610.12408)
  Cites L-GATr.
- **A Unified Geometric Framework for Understanding Collider Event Manifolds** *(context)* — Cai. *arXiv, 2026-10-07.* [arxiv](https://arxiv.org/abs/2610.09320)
  Cites L-GATr.
- **How to scale your HEP ML models: A recipe for robust architecture comparisons at scale** *(context)* — Vigl. *arXiv, 2026-10-05.* [arxiv](https://arxiv.org/abs/2610.06784)
  Cites L-GATr.
- **Machine learning for the LHC physics program: a 2025-2026 stocktake** *(context)* — Thaler. *arXiv, 2026-09-26.* [arxiv](https://arxiv.org/abs/2609.32874)
  Review citing L-GATr.
- **Towards Foundation Models on Hardware Accelerators for Particle Physics** *(context)* — Benyas. *arXiv, 2026-09-22.* [arxiv](https://arxiv.org/abs/2609.26899)
  Cites the economical (L-GATr-slim) taggers.
- **More QCD Masterclass Lectures on Jet Physics and Machine Learning** *(context)* — Larkoski. *arXiv, 2026-09-09.* [arxiv](https://arxiv.org/abs/2609.09785)
  Lecture notes citing L-GATr and the GA input-language paper.
- **A High- and Variable-Dimensional Measurement of the Z+jets Differential Cross Section with the ATLAS Experiment** *(context)* — Greif. *arXiv, 2026-08-28.* [arxiv](https://arxiv.org/abs/2608.28449)
  Cites L-GATr.
- **Know What You Don't Flow** *(context)* — Butter. *arXiv, 2026-08-21.* [arxiv](https://arxiv.org/abs/2608.21509)
  Cites L-GATr.
- **Assessing Parameter Redundancy in Transformers for Jet Tagging** *(context)* — Cheng. *arXiv, 2026-08-17.* [arxiv](https://arxiv.org/abs/2608.16061)
  Benchmarks against L-GATr.
- **Unknown Unknowns: Model Misspecification in Machine Learning for Physics** *(context)* — Cruz-Martinez. *arXiv, 2026-08-13.* [arxiv](https://arxiv.org/abs/2608.13633)
  Cites L-GATr.
- **Generative Amplification with Surrogate Monte Carlo** *(context)* — Bahl. *arXiv, 2026-08-06.* [arxiv](https://arxiv.org/abs/2608.06450)
  Cites L-GATr.
- **Machine Learning is Good for Physics - and Vice Versa** *(context)* — Krämer. *arXiv, 2026-08-06.* [arxiv](https://arxiv.org/abs/2608.05812)
  Essay citing L-GATr.
- **Virtues and Vices of Equivariant Transformers** — Favaro, Plehn, Qu, Spinner. *arXiv, 2026-08-03.* [arxiv](https://arxiv.org/abs/2608.02735)
  Lorentz-equivariant (spacetime-algebra) transformers for large-scale jet and flavour tagging, inference-optimised.
- **Neural Control Variates at LO and NLO** *(context)* — Heimel. *arXiv, 2026-07-26.* [arxiv](https://arxiv.org/abs/2607.23591)
  Cites L-GATr.
- **Predict before you train: Scaling Laws for particle physics foundation models** *(context)* — Uslu. *arXiv, 2026-07-25.* [arxiv](https://arxiv.org/abs/2607.23377)
  Cites L-GATr.
- **Learning Standard Model structure from LHC data with Riemannian flow matching** *(context)* — Kato. *arXiv, 2026-07-17.* [arxiv](https://arxiv.org/abs/2607.16144)
  Cites L-GATr.
- **Local Conformal Predictions for Calibrated Surrogates** *(context)* — Dubey. *arXiv, 2026-07-01.* [arxiv](https://arxiv.org/abs/2607.01354)
  Cites L-GATr.
- **Machine learning fully hadronic events with spectral functions** *(context)* — Altakach. *arXiv, 2026-06-25.* [arxiv](https://arxiv.org/abs/2606.27420)
  Cites L-GATr.
- **Interpreting "Interpretability" and Explaining "Explainability" in Machine Learning in Physics** *(context)* — Gambhir. *arXiv, 2026-06-24.* [arxiv](https://arxiv.org/abs/2606.26228)
  Cites L-GATr.
- **One Generator, Any Process: LLM-Conditioning for the LHC** *(context)* — Bahl. *arXiv, 2026-06-22.* [arxiv](https://arxiv.org/abs/2606.23791)
  Cites L-GATr.
- **What Do Lorentz-Equivariant Jet Taggers Learn?** — anonymous. *arXiv; ICML 2026 AI4Physics workshop, 2026-06-19.* [arxiv](https://arxiv.org/abs/2606.21790) · [openreview](https://openreview.net/forum?id=xr5xIic0Ei)
  Probes, equivariance tests and grade ablations on L-GATr, L-GATr-slim and LLoCa-T jet taggers. — also: Theory and analysis of GA layers
- **Event Generation with Parallel Langevin Sampling and Learned Stein Diagnostics** *(context)* — Verheyen. *arXiv, 2026-06-12.* [arxiv](https://arxiv.org/abs/2606.14854)
  Cites L-GATr.
- **CaloTrilogy: Toward a Breakthrough in One-Step, End-to-End, Physics-Guided Shower Generation** *(context)* — Jiang. *arXiv, 2026-06-02.* [arxiv](https://arxiv.org/abs/2606.04165)
  Cites L-GATr.
- **Particle-Lund Multimodality in Jet Taggers** *(context)* — Gouskos. *arXiv, 2026-05-26.* [arxiv](https://arxiv.org/abs/2605.26821)
  Benchmarks against L-GATr.
- **Pointwise Metrics Mislead: An Evaluation Protocol for Multimodal Inverse Problems** *(context)* — Baattrup. *arXiv, 2026-05-21.* [arxiv](https://arxiv.org/abs/2605.22891)
  Cites L-GATr.
- **Patch Hierarchical Attention Transformer for Efficient Particle Jet Tagging** *(context)* — Wang. *arXiv, 2026-05-20.* [arxiv](https://arxiv.org/abs/2605.21789)
  Benchmarks against L-GATr.
- **The Monte Carlo Ecosystem in High-Energy Physics: A Primer** *(context)* — Beekveld. *arXiv, 2026-05-15.* [arxiv](https://arxiv.org/abs/2605.16036)
  Cites L-GATr.
- **Geometric algebra as the input language of collider foundation models** — Abasov, Dudko, Markina, Perfilov et al.. *arXiv, 2026-05-15.* [arxiv](https://arxiv.org/abs/2605.15910) · [inspire](https://inspirehep.net/literature/3156709)
  A whole collider event as one Cl(1,3)⊗V_flav multivector; theorem on which Lorentz invariants higher grades add; L-GATr-type transformer.
- **Open LHC Monte Carlo Event Generation** *(context)* — Bothmann. *arXiv, 2026-05-12.* [arxiv](https://arxiv.org/abs/2605.12229)
  Cites L-GATr.
- **Generative models on phase space** *(context)* — Bogorad. *arXiv, 2026-04-02.* [arxiv](https://arxiv.org/abs/2604.02415)
  Cites L-GATr.
- **MadNIS at NLO** *(context)* — Crescenzo. *arXiv, 2026-03-23.* [arxiv](https://arxiv.org/abs/2603.22407)
  Cites L-GATr.
- **Explicit or Implicit? Encoding Physics at the Precision Frontier** — Bresó-Pla, Greif, Mikuni, Nachman et al.. *arXiv; Physical Review D 114, 036034 (2026), 2026-03-09.* [arxiv](https://arxiv.org/abs/2603.08802) · [openreview](https://openreview.net/forum?id=uhT0wi62Et)
  Explicit symmetry (L-GATr) versus large-scale pre-training at the precision frontier of collider physics.
- **End-to-end event reconstruction for precision physics at future colliders** — Garcia, Herrmann, Krzmanc, Selvaggi. *arXiv, 2026-03-04.* [arxiv](https://arxiv.org/abs/2603.04084)
  Geometric algebra transformer networks combined with object-condensation clustering for particle-flow reconstruction.
- **The Latent Information Geometry of Jet Classification** *(context)* — Kuntz. *arXiv, 2026-03-02.* [arxiv](https://arxiv.org/abs/2603.02310)
  Cites L-GATr.
- **Unfolding without Iterations, Adversaries, or Surrogates** *(context)* — Ore. *arXiv, 2026-02-27.* [arxiv](https://arxiv.org/abs/2602.24282)
  Cites L-GATr.
- **Neural Scaling Laws for Boosted Jet Tagging** *(context)* — Vigl. *arXiv, 2026-02-17.* [arxiv](https://arxiv.org/abs/2602.15781)
  Cites L-GATr.
- **MadSpace - Event Generation for the Era of GPUs and ML** *(context)* — Heimel. *arXiv, 2026-02-06.* [arxiv](https://arxiv.org/abs/2602.06895)
  Cites L-GATr.
- **CoLLM: AI engineering toolbox for end-to-end deep learning in collider analyses** *(context)* — Esmail. *arXiv, 2026-02-06.* [arxiv](https://arxiv.org/abs/2602.06496)
  Cites L-GATr.
- **Modern Machine Learning and Particle Physics Phenomenology at the LHC** *(context)* — Ubiali. *arXiv, 2026-02-03.* [arxiv](https://arxiv.org/abs/2602.03728)
  Review citing L-GATr.
- **PanopTag: Simultaneously Tagging All Jets in a Particle Collision Event** *(context)* — Qureshi. *arXiv, 2026-01-23.* [arxiv](https://arxiv.org/abs/2601.16417)
  Cites L-GATr.
- **Scaling laws for amplitude surrogates** *(context)* — Bahl. *arXiv, 2026-01-19.* [arxiv](https://arxiv.org/abs/2601.13308)
  Cites L-GATr (amplitude surrogates).
- **jBOT: Semantic Jet Representation Clustering Emerges from Self-Distillation** *(context)* — Tsoi. *arXiv, 2026-01-16.* [arxiv](https://arxiv.org/abs/2601.11719)
  Cites L-GATr.
- **AllShowers: One model for all calorimeter showers** *(context)* — Buss. *arXiv, 2026-01-16.* [arxiv](https://arxiv.org/abs/2601.11716)
  Cites L-GATr.
- **How to Trust Learned Loop Amplitudes** *(context)* — Bahl. *arXiv, 2026-01-02.* [arxiv](https://arxiv.org/abs/2601.00950)
  Cites L-GATr (amplitude surrogates).

### Vision, remote sensing and medical imaging

Image backbones, hyperspectral classification, registration, segmentation and enhancement.

- **Geometric algebra enhanced twin-branch contrastive network design for few-shot hyperspectral image classification** — Zhang, Cao, Liu. *Pattern Recognition, 2026-11-01.* [doi](https://doi.org/10.1016/j.patcog.2026.113549)
  GA-enhanced twin-branch contrastive network for few-shot hyperspectral classification.
- **Preprocessing of Retinal Fundus Image Using Clifford Algebra** *(adjacent)* — Roy, Mitra, Setua. *Preprints.org, 2026-05-11.* [doi](https://doi.org/10.20944/preprints202605.0666.v1)
  Clifford colour space and rotor-based illumination correction as a preprocessing stage for diagnosis pipelines (not itself a neural network).
- **UHD Low-Light Image Enhancement via Real-Time Enhancement Methods with Clifford Information Fusion** — Wang, Wu, Zhao, Gao. *arXiv, 2026-04-10.* [arxiv](https://arxiv.org/abs/2604.09321)
  Feature fusion in a 2D Clifford multivector space (scalar, vector, bivector) with Clifford similarity; millisecond 4K/8K inference.
- **Geometric algebra transformer with spectral-spatial-volumetric feature extraction for hyperspectral image classification** — Fabi, Schiavella, Amerini. *Pattern Recognition Letters, 2026-04-01.* [doi](https://doi.org/10.1016/j.patrec.2026.02.001) · [openreview](https://openreview.net/forum?id=jfb7cr7u30)
  H-GATr embeds spectral, spatial and volumetric cues as multivectors without extra parameters and feeds them to a geometric algebra transformer.
- **Conformal Geometric Algebra Graph Learning Theory for 3D Human Skeleton Action** — Cao, Wang, Hu. *ECTI DAMT & NCON 2026, 2026-02-04.* [doi](https://doi.org/10.1109/ectidamtncon67592.2026.11459996)
  Conformal-GA graph learning applied to 3D skeleton action recognition. — also: Graphs, multimodal and representation learning
- **GIENet: Graph-Infused Encoder with Geometric Algebra Single-Pass Fusion for Unsupervised Deformable Registration** — Zhou, Cao, Feng, Zhang. *SSRN, 2026-01-01.* [doi](https://doi.org/10.2139/ssrn.6205817)
  Graph-infused encoder with a GA single-pass fusion decoder for unsupervised 3D deformable registration. — also: Graphs, multimodal and representation learning
- **Geometric Algebra-based Dual-branch Graph Convolutional Network for Deformable Medical Image Registration with Bi-Directional Flow** — Cao, Anwar, Yan. *IEEE Journal of Biomedical and Health Informatics, 2026-01-01.* [doi](https://doi.org/10.1109/jbhi.2026.3718748)
  GA-based dual-branch graph convolutional network for deformable registration with bidirectional flow. — also: Graphs, multimodal and representation learning

### Graphs, multimodal and representation learning

Graph neural networks, sheaves, binding / vector-symbolic representations with Clifford structure.

- **ICE: Task-Aligned Clifford Latent Fields for Multimodal Graph Foundation Models** — anonymous. *arXiv; ICLR 2027 submission, 2026-09-24.* [arxiv](https://arxiv.org/abs/2609.29398) · [openreview](https://openreview.net/forum?id=wIfzslZuuF)
  Clifford latent fields whose higher-grade channels recover pair relations across multimodal foundation graphs.
- **Recursive Binding on a Budget: Subspace Carving in Order-p Tensor Memories** *(adjacent)* — Pence, Yamada, Singh. *arXiv; ICML 2026 submission, 2026-06-10.* [arxiv](https://arxiv.org/abs/2606.11391) · [openreview](https://openreview.net/forum?id=Oc5d0gsRTs)
  Shows tensor-product representations are a special case of binding in a Clifford algebra and gives a Clifford formulation of the method. — also: Theory and analysis of GA layers
- **Conformal geometric algebra graph learning theory for non-Euclidean data** — Cao, Wang. *ICCAIS 2025 (SPIE proceedings), 2026-04-24.* [doi](https://doi.org/10.1117/12.3112939)
  Theory of graph learning on non-Euclidean data in conformal GA. — also: Theory and analysis of GA layers
- **LION: A Clifford Neural Paradigm for Multimodal-Attributed Graph Learning** — Li, Wu, Chen, Sun et al.. *arXiv (re-posted as 2608.24795 in Aug 2026), 2026-01-29.* [arxiv](https://arxiv.org/abs/2601.21453) · [openreview](https://openreview.net/forum?id=TgmpQaICO2)
  Modality-aware geometric manifold grounded in Clifford algebra; decoupled propagate-then-aggregate graph network for multimodal-attributed graphs.

### Robotics, driving and control learning

Policies, scene models and pose pipelines built on projective or conformal GA.

- **Synthesis of discrete mechanisms: a hybrid approach based on geometric algebra and genetic algorithms** *(adjacent)* — Kadan, Petrov. *Yugra State University Bulletin, 2026-10-13.* [doi](https://doi.org/10.18822/byusu20260372-76)
  Synthesises kinematic mechanisms from a discrete set of components by searching projective-geometric-algebra representations with a genetic algorithm. — also: Other machine-learning uses of GA
- **Screw Attention: Rigid-Body Algebra Inside a Transformer** *(context)* — anonymous. *arXiv, 2026-10-01.* [arxiv](https://arxiv.org/abs/2610.00904)
  Uses screw theory and spatial transforms, not Clifford algebra; listed because it is the closest non-GA competitor for rigid-body-aware attention.
- **From Language to Task Maps: Compiling Semantic Relations While Preserving Task-Relevant Freedom** *(adjacent)* — Park, Lee, Lee, Hwang. *arXiv, 2026-09-28.* [arxiv](https://arxiv.org/abs/2609.34412)
  Compiles language-specified relations into conformal-GA task maps composed with learned policies through RMPflow.
- **Hybrid Diffusion Policies with Projective Geometric Algebra for Efficient Robot Manipulation Learning** — Sun, Wang, Yang, Chen et al.. *ICRA 2026 (arXiv preprint Jul 2025), 2026-06-01.* [arxiv](https://arxiv.org/abs/2507.05695) · [doi](https://doi.org/10.1109/icra57385.2026.11696239) · [openreview](https://openreview.net/forum?id=ZOxNLgPuhA)
  hPGA-DP uses a projective GA transformer (P-GATr) as state encoder and action decoder around a standard diffusion denoiser; preprint 2025, published 2026.
- **The Cartan-Topos Protocol: A Unified Geometric and Categorical Framework for Resilient Multi-Agent Coordination** *(adjacent)* — Hernández, Sánchez-Soto. *arXiv, 2026-05-30.* [arxiv](https://arxiv.org/abs/2606.00714)
  Clifford rotors and motors for SE(3) pose synchronisation in multi-agent coordination; not deep learning.
- **Efficient Equivariant Transformer for Self-Driving Agent Modeling** — Xu, Chen, Wong, Zhang et al.. *arXiv; CVPR 2026, 2026-04-01.* [arxiv](https://arxiv.org/abs/2604.01466) · [openreview](https://openreview.net/forum?id=vmIehea8D9) · [url](https://cvpr.thecvf.com/virtual/2026/poster/36432)
  DriveGATr encodes scene elements as multivectors of the 2D projective GA R*(2,0,1); SE(2)-equivariant attention without pairwise relative encodings; Waymo Open Motion. — also: New GA / Clifford architectures

### Wireless channels and neural fields

GA attention inside neural radiance fields and Gaussian splatting for radio propagation.

- **A Geometric Algebra-Informed 3DGS Framework for Wireless Channel Prediction** — Shen, Enamorado, Mao, Wang. *arXiv; CVPR 2026 submission, 2026-05-18.* [arxiv](https://arxiv.org/abs/2605.19065) · [openreview](https://openreview.net/forum?id=fcXTtuAMvs)
  GAI-GS couples 3D Gaussian splatting with a GA-based attention mechanism for wireless modelling.
- **A Geometric Algebra-informed NeRF Framework for Generalizable Wireless Channel Prediction** — Shen, Enamorado, Mao, Wang. *arXiv; IEEE INFOCOM 2026, 2026-04-13.* [arxiv](https://arxiv.org/abs/2604.11983) · [doi](https://doi.org/10.1109/infocom59046.2026.11571298)
  GAI-NeRF puts geometric-algebra attention over ray-object interactions inside a neural radiance field for channel prediction.

### Sequence models, language and LLM tooling

Rotor-based recurrences, time-series forecasters, semantics and GA-aware language-model tools.

- **Generative geospatial modelling with geometric algebra** *(adjacent)* — anonymous. *Philosophical Transactions of the Royal Society A, 2026-08-13.* [doi](https://doi.org/10.1098/rsta.2025.0103)
  GA as a bridge to large-language-model spatial reasoning for geospatial generation. — also: Surveys, reviews and position papers
- **GA-VisAgent: A Multi-Agent application for code generation and visualization in interactive learning** *(adjacent)* — Jian, Jianbo, Yuhao, Zhenxia. *arXiv, 2026-05-02.* [arxiv](https://arxiv.org/abs/2605.01299)
  LLM agents built on a GA-specialised language model (GAGPT) that generate and visualise geometric-algebra code. — also: Software and resources
- **Adaptive Domain Models: Bayesian Evolution, Warm Rotation, and Principled Training for Geometric and Neuromorphic AI** *(adjacent)* — Haynes. *arXiv, 2026-03-18.* [arxiv](https://arxiv.org/abs/2603.18104)
  Hybrid design with grade-typed Clifford representations for geometry and spiking layers for temporal patterns, serving as domain checkers for language-model pipelines. — also: New GA / Clifford architectures
- **Semantic Plane Alignment: Measuring Summarization Completeness via Tensor-based Geometric Algebra** *(adjacent)* — Sudjianto. *SSRN, 2026-01-01.* [doi](https://doi.org/10.2139/ssrn.6168708)
  Planes spanned by embeddings (GA outer products) as an evaluation metric for summary completeness. — also: Other machine-learning uses of GA

### Clifford-valued network dynamics

Stability and synchronisation theory of recurrent networks with Clifford-number states (control theory).

- **A restart-graded L1 scheme for fractional-order Clifford-valued delay neural networks: Weak singularities and sharp convergence** *(adjacent)* — Rajchakit. *Nonlinear Science, 2026-12-01.* [doi](https://doi.org/10.1016/j.nls.2026.100203)
  Numerical scheme with convergence proof for fractional-order Clifford-valued delay networks.
- **Synchronization of neutral-type Clifford-valued inertial neural networks with mixed delays by non-reduced-order and non-separation approaches** *(adjacent)* — Deng, Li, Tan. *Neurocomputing, 2026-12-01.* [doi](https://doi.org/10.1016/j.neucom.2026.134991)
  Inertial (second-order) Clifford-valued networks analysed without order reduction.
- **Observer-Based Output-Feedback Synchronization of Fractional-Order Fuzzy Clifford-Valued Neural Networks: A Cascade Certificate and the Quantified Price of Output Feedback** *(adjacent)* — Rajchakit. *Complex Systems Stability & Control, 2026-10-23.* [doi](https://doi.org/10.53941/cssc.2026.100028)
  Leader-following synchronisation when only part of the multivector state is measured.
- **H-infinity synchronisation of fractional-order Clifford-valued reaction-diffusion neural networks with leakage delays via sampled-data control** *(adjacent)* — Rajchakit, Lim, Tajudeen, Huang. *International Journal of Systems Science, 2026-10-21.* [doi](https://doi.org/10.1080/00207721.2026.2736095)
  Reaction-diffusion Clifford-valued networks under sampled-data control.
- **Relaxed synchronization analysis of fractional-order Clifford-valued delayed neural networks via novel inner product-based inequality** *(adjacent)* — Manoj, Sriraman. *Physica Scripta, 2026-10-20.* [doi](https://doi.org/10.1088/1402-4896/aeb198)
  Inner-product inequalities for Clifford numbers that avoid real-valued decomposition.
- **Stability and Control of Fractional-Order Fuzzy Stochastic Clifford-Valued Neural Networks with Hybrid Markovian Switching, Cyber-Attacks, and Event-Triggered Multi-Agent Synchronization** *(adjacent)* — Rajchakit. *Nonlinear Analysis and Computer Simulations, 2026-10-08.* [doi](https://doi.org/10.53941/nacs.2026.100016)
  Mean-square Mittag-Leffler synchronisation under deception attacks and event-triggered communication.
- **Fixed-time synchronization of Clifford biquaternion neural networks with two-sided coefficients and application to multispectral image protection** *(adjacent)* — Li, Zhang, Kou, Liu. *Neural Networks (volume dated Jan 2027), 2026-10-01.* [doi](https://doi.org/10.1016/j.neunet.2026.109528)
  Eight-dimensional Clifford biquaternion networks with two-sided coefficients; fixed-time synchronisation and multispectral image encryption. — also: Hypercomplex-valued networks
- **Exponential lag synchronization of inertial Clifford-valued neural networks with proportional delays via sampled-data pinning control and applications to multi-spectral image watermarking** *(adjacent)* — Rajchakit, Lim. *Neurocomputing, 2026-10-01.* [doi](https://doi.org/10.1016/j.neucom.2026.134343)
  Lag synchronisation with a watermarking application.
- **Synchronization dynamics of fractional-order Clifford-valued BAM delayed neural networks and their applications** *(adjacent)* — Nagappan, Ramalingam, Rajchakit, Rajchakit et al.. *Journal of Mathematics and Computer Science, 2026-09-23.* [doi](https://doi.org/10.22436/jmcs.043.03.02)
  Synchronisation of fractional-order Clifford-valued BAM networks.
- **Quasi-Projective Synchronization of Heterogeneous Fractional-Order Clifford-Valued Neural Networks via Intermittent Pinning Control** *(adjacent)* — Rajchakit, Lim, Tajudeen, Huang. *International Journal of Robust and Nonlinear Control, 2026-09-07.* [doi](https://doi.org/10.1002/rnc.70702)
  Heterogeneous drive and response networks with Clifford-valued states.
- **Aperiodic intermittent control strategy for synchronization of Clifford-valued neural networks on time scales using matrix measure** *(adjacent)* — Agrawal, Abbas, Debbouche. *Applied Mathematics and Computation, 2026-09-01.* [doi](https://doi.org/10.1016/j.amc.2026.130034)
  Matrix-measure synchronisation criteria on time scales, unifying continuous and discrete time.
- **Global Mittag-Leffler synchronization of fractional-order Clifford-valued neural networks with mixed time-varying delays via adaptive quantized pinning control** *(adjacent)* — Rajchakit, Lim. *International Journal of Dynamics and Control, 2026-08-07.* [doi](https://doi.org/10.1007/s40435-026-02228-5)
  Synchronisation criteria under finite-resolution communication without decomposing the Clifford-valued system.
- **Slide Mode Control Based Finite and Fixed-Time Projective Synchronization for Delayed Clifford-Valued Neural Networks** *(adjacent)* — Kumari, Malik. *Mathematical Methods in the Applied Sciences, 2026-07-30.* [doi](https://doi.org/10.1002/mma.70915)
  Sliding-mode controllers for finite- and fixed-time projective synchronisation.
- **Finite-time stabilization of fractional-order fuzzy Clifford-valued neural networks using novel inequalities** *(adjacent)* — Sriraman, Manoj. *Physica Scripta, 2026-07-02.* [doi](https://doi.org/10.1088/1402-4896/ae8133)
  Non-decomposition finite-time stabilisation analysis.
- **Multistability of equilibria for Clifford-valued Cohen-Grossberg neural networks with discontinuous activation functions and time delays** *(adjacent)* — Yang, Qiao, Duan, Miao. *Mathematics and Computers in Simulation, 2026-07-01.* [doi](https://doi.org/10.1016/j.matcom.2025.12.015)
  Counts and locates multiple equilibria of Clifford-valued Cohen-Grossberg networks.
- **A unified matrix measure strategy for QGPL synchronization of Clifford-valued competitive model with parameter mismatch: Application to LSTM model** *(adjacent)* — Agrawal, Abbas. *Neurocomputing, 2026-06-01.* [doi](https://doi.org/10.1016/j.neucom.2026.133377)
  Quasi-generalised projective lag synchronisation of a Clifford-valued competitive model with an LSTM application.
- **Finite-time stability of fractional-order delayed Clifford-valued neural networks via a novel non-separation approach** *(adjacent)* — Ramalingam, Nagappan. *Computational and Applied Mathematics, 2026-03-10.* [doi](https://doi.org/10.1007/s40314-026-03690-9)
  Finite-time stability without separating the Clifford components.
- **Projective Synchronization of Fractional-Order BAM Clifford-Valued Delayed Memristive Neural Networks: An Inner Product-Based Inequality** *(adjacent)* — Manoj, Sriraman. *Mathematical Methods in the Applied Sciences, 2026-02-19.* [doi](https://doi.org/10.1002/mma.70592)
  Bidirectional associative memory networks with memristive Clifford-valued weights.
- **Global Exponential Weyl Almost Periodic Synchronization for Hypercomplex-Valued High-Order Hopfield Neural Networks with Delays** *(adjacent)* — Li, Xu. *Qualitative Theory of Dynamical Systems, 2026-02-01.* [doi](https://doi.org/10.1007/s12346-026-01459-3)
  Hypercomplex-valued high-order Hopfield networks (Clifford algebras as the unifying case). — also: Hypercomplex-valued networks
- **Advances in the Mathematical Theory of WPAA Dynamics for Impulsive High Order Neural Systems in Clifford Algebras** *(adjacent)* — Aouiti, Dridi. *Journal of Nonlinear Dynamics and Applications, 2026-01-21.* [doi](https://doi.org/10.62762/jnda.2025.838385)
  Weighted pseudo-almost automorphic solutions of impulsive high-order Hopfield networks in Clifford algebras.
- **Noise Robust Projection Rule for Split-Quaternion-Valued Hopfield Networks** *(adjacent)* — Kobayashi. *IEICE Transactions on Fundamentals, 2026-01-01.* [doi](https://doi.org/10.1587/transfun.2026map0002)
  Split quaternions are a four-dimensional Clifford algebra; the paper frames itself as Clifford-algebra neural networks. — also: Hypercomplex-valued networks

### Hypercomplex-valued networks

Quaternion, biquaternion and general hypercomplex layers framed as Clifford algebras.

- **Fully tensorial approach to hypercomplex-valued neural networks** *(adjacent)* — Niemczynowicz, Kycia. *Information Sciences (arXiv 2407.00449), 2026-11-01.* [arxiv](https://arxiv.org/abs/2407.00449) · [doi](https://doi.org/10.1016/j.ins.2026.123796)
  Unified tensor implementation of hypercomplex (including Clifford) layers. — also: Software and resources
- **Performance Increases in Physics-Informed Machine Learning using Hypercomplex-based Automatic Differentiation (HYPAD)** *(adjacent)* — Roberts, Aristizabal, Millwater. *ACM Transactions on Mathematical Software, 2026-10-16.* [doi](https://doi.org/10.1145/3858612)
  HYPAD: hypercomplex-number automatic differentiation that speeds up derivative computation when training physics-informed neural networks. — also: Scientific machine learning
- **Ghost Features and Spooky Transfer Learning for Hypercomplex-Valued Neural Networks** *(adjacent)* — Vieira, Valle. *arXiv, 2026-08-07.* [arxiv](https://arxiv.org/abs/2608.07735) · [code](https://github.com/mevalle/v-nets/)
  Hypercomplex layers whose real part reproduces a trained real layer; the imaginary parts add "ghost features" that can be transferred into trained networks. — also: New GA / Clifford architectures
- **Global Fixed-Time Synchronization of Fractional-Order Dual-Split-Quaternion Valued Memristive Neural Networks: Full Proofs and 6D Relativistic Image Encryption** *(adjacent)* — Rajchakit. *Complex Systems Stability & Control, 2026-07-07.* [doi](https://doi.org/10.53941/cssc.2026.100013)
  Dual-split-quaternion algebra (Clifford-adjacent hypercomplex) for synchronisation and image encryption. — also: Clifford-valued network dynamics
- **Hypercomplex Phase Retrieval** *(context)* — anonymous. *arXiv, 2026-02-27.* [arxiv](https://arxiv.org/abs/2602.23946)
  Clifford-algebra hypercomplex signal processing for phase retrieval; not a neural network.

### Other machine-learning uses of GA

Classifiers, filters, causal discovery and quantum machine learning built on GA.

- **Hybrid Geometric Algebra Correntropy Criterion** *(adjacent)* — Zhao, Hou, Long. *Information Theoretic Learning-based Filter (Springer), 2026-10-06.* [doi](https://doi.org/10.1007/978-3-032-29623-8_7)
  GA-valued adaptive filtering with a correntropy criterion.
- **Machine Learning the Phase Structure and Algebraic Invariants of Clifford- and Grassmann-Valued Cellular Automata** *(adjacent)* — Mirza. *Research Square, 2026-07-31.* [doi](https://doi.org/10.21203/rs.3.rs-10394430/v1)
  Machine learning applied to discrete dynamical systems driven by Clifford and Grassmann algebras.
- **Geometric Algebra Quantum Gate Decomposition** *(context)* — Amraoui, Toffano. *Axioms 15(7):523, 2026-07-13.* [doi](https://doi.org/10.3390/axioms15070523)
  Rotor-based quantum gate synthesis; GA-adjacent, not learning.
- **Adaptive Sphere Classification via Conformal Geometric Algebra** — Athmani. *Research Square, 2026-03-31.* [doi](https://doi.org/10.21203/rs.3.rs-9244644/v1)
  Supervised classifier whose learning rule is the conformal-GA point-in-sphere inner product.
- **Quantum classification and search algorithms using spinorial representations** *(adjacent)* — anonymous. *arXiv, 2026-03-17.* [arxiv](https://arxiv.org/abs/2603.16564)
  Quantum classification and search algorithms built on Clifford algebras and spinors.
- **Causal Discovery using Geometric Algebra** *(adjacent)* — Sudjianto, Setiawan. *SSRN, 2026-01-01.* [doi](https://doi.org/10.2139/ssrn.7291638)
  Pairwise causal-direction scoring recast with outer products and multivariate angles. — also: Scientific machine learning

### Software and resources

Libraries, curated lists and theme issues.

- **Preface to 'Modern applications of geometric algebra' (Phil. Trans. R. Soc. A theme issue 384/2326)** *(context)* — Hitzer, Hildenbrand, Papagiannakis. *Philosophical Transactions of the Royal Society A, 2026-08-13.* [doi](https://doi.org/10.1098/rsta.2026.0174) · [url](https://doi.org/10.1098/rsta.2025.0104)
  Theme issue bundling the 2026 GA papers above; includes a GA implementation and memory-requirements overview (rsta.2025.0104). — also: Surveys, reviews and position papers

## 2025

### High-energy physics

The Lorentz-equivariant geometric algebra transformer (L-GATr) line and other spacetime-algebra models.

- **Economical Jet Taggers - Equivariant, Slim, and Quantized** — Petitjean, Plehn, Spinner, Köthe. *arXiv (Dec 2025), 2025-12-18.* [arxiv](https://arxiv.org/abs/2512.17011) · [inspire](https://inspirehep.net/literature/3094401)
  Late-2025 precursor that introduces L-GATr-slim, the slim Lorentz-equivariant tagger used by several 2026 papers.

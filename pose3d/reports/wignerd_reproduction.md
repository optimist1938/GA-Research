# Wigner-D regression reproduction (Pascal3D+)

Model: `pose3d/models/wignerd.py`, an Image2Sphere-style head predicting 455 = Σ_{ℓ=0}^{9}(2ℓ+1)² Wigner-D
coefficients; decoding is the argmax of ⟨Ψ, flat_wigner(R_k)⟩ over a level-5 HEALPix SO(3) grid (2,359,296 rotations).

| Quantity | Paper | Ours |
|---|---|---|
| Median rotation error | 8.6° | 9.09–9.20° final, 8.69–8.99° best epoch |

Findings:

* The gap to the paper is about 0.5°; the loss-scale convention (sum vs. mean over the 455 coefficients) is
  ambiguous in the paper and moves results by a similar amount.
* A mean/unit-variance variant gives about 20° and is not used.
* The diningtable class accounts for ≈0.26° of the gap.
* On SYMSOL (images, 10 % protocol) the same head reaches grid LL −0.755 (T=1) and −0.635 (T=2);
  lr 0.1 collapsed to the uniform value −2.289.
* The degree-1 block of the flat Wigner vector divided by √3 equals R, so an SVD projection yields a rotation matrix
  without the grid.

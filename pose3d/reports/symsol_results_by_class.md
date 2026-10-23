# SYMSOL (images) results by shape

Setup: SYMSOL I, rendered images, ResNet-50 backbone, exact log-likelihood (LL, higher is better; the uniform
distribution on SO(3) scores −2.29), MRE = mean rotation error in degrees of the best mode (lower is better).
Numbers are from single runs unless a range is given.

## Overall log-likelihood

| Model | LL |
|---|---|
| Spherical-Convolution-Denoiser (5 shapes: 10.43, 10.40, 10.69, 7.48, 7.31; error 0.67°; grid 7.52) | 9.26 |
| GATr + 49 harmonics, point cloud (optimist1938) | 7.4–7.6 |
| GA-1 | 6.31 |
| GATr spherecloud + 49 harmonics (`harm6`) | 3.24 |
| GATr spherecloud control (no harmonics) | 1.24 / 1.11 |
| DiT variants | ≈0.1–1.3 |
| MLP flow | −1.88 |
| Pooled GATr (original Clifford Flow, image conditioning) | −2.17 |
| Wigner-D regression (Image2Sphere-style, grid LL, T=1) | −0.755 (T=2: −0.635) |

Point-cloud conditioning (not images): GATr+RoPE 9.17, hybrid 8.70, SO3F 6.32, control 1.47;
unlimited data: harmonics 10.96 vs control 8.82.

## Wigner-D regression per shape (images)

| Shape | LL | MRE (°) |
|---|---|---|
| tetrahedron | −1.51 | 27.2 |
| cube | −1.93 | 30.9 |
| icosahedron | −2.33 | 30.5 |
| cone | 1.21 | 3.3 |
| cylinder | 0.79 | 6.7 |

Pooled GATr (original Clifford Flow) has MRE 44.9° over all classes.

## Reading

* The 49 harmonics lift the symmetric polyhedra with few modes (tetrahedron, cube) but not the icosahedron, and
  stay far below the Spherical-Convolution-Denoiser on images.
* The point-cloud numbers and the image numbers use different conditioning and must not be compared directly.

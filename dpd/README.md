# dpd

**Best score**: `gated` with a separate 3-layer gate per band, ensemble of 3 seeds, test NMSE vs d
**−19.9 / −16.1 / −23.5 dB** (vs x: −30.9 / −31.9 / −35.6 dB) on bands A / B / C.
Best single model per parameter: `gated_tcn` 64 ch, −19.8 / −15.8 / −22.2 dB with 86K parameters;
with 18.5K parameters (Huawei's LUT: 16.8K) it still reaches −18.4 / −15.1 / −19.5 dB.

Setup and how to run: [`USAGE.md`](USAGE.md). Results: [`reports/`](reports/). Reading material:
[`awesome-reference/`](awesome-reference/README.md).

Task: Huawei 3-band DPD, learn d_A, d_B, d_C = F(x_A, x_B, x_C) with memory (`GeoData_TB.mat`,
221K samples, Fs = 368.64 MHz, train = first 80%, test = last 20%). Two NMSE normalizations are
reported: by d (strict) and by x (Huawei's `nmse(xRef, e)` in simpleModel). Scores are the test
mean over 3 seeds where a ± is given.

| Status | Idea | Description |
|---|---|---|
| reference | `lut_ls`: Huawei LUT | x_c(k−m)·LUT(\|x_A\|,\|x_B\|,\|x_C\|), Chebyshev order 5, m ∈ −3..3, LS. vs d −10.6 / −7.9 / −7.9, vs x −21.6 / −23.8 / −20.0 dB, 16.8K params. Matches Huawei's "GMP/LUT > 20 dB", so Huawei normalizes by x. [report](reports/cpu-baselines.typ) |
| done | `mlp_poly`: team's first MLP | Poly features of x, conj(x), memory 20, \|d\|-weighted MSE, not equivariant. vs d −12.8 / −10.3 / −11.8 dB, 63K params. [report](reports/cpu-baselines.typ) |
| done | `gated`, magnitudes only | U(1)³-equivariant: x_c(k−m)·g_m(invariants), g an MLP on \|x_b(k−i)\|². vs d −17.2 / −11.5 / −18.5 dB, 30K params: +4.5 / +1.2 / +6.7 dB over `mlp_poly` with half the parameters. [report](reports/cpu-baselines.typ) |
| adopted | `gated` + cross-time invariants | Adds Re/Im x_b(k−i)·conj x_b(k−j), \|i−j\| ≤ 2 (the geometric product within a band plane). vs d **−18.5 / −15.2 / −19.4** dB, vs x −29.5 / −31.1 / −31.5 (Huawei's eRef: −33.5 / −33.5 / −31.0). [report](reports/cpu-baselines.typ) |
| done | `lut_cp`: low-rank LUT | The 3D LUT as a rank-R CP product of 1D Chebyshev functions. Rank 4 (5.6K params, −67%) is 0.2–0.5 dB *better* than `lut_ls`; ranks 8 and 16 add nothing. Param-count KPI lever. [report](reports/cpu-baselines.typ) |
| flag, off | `--band_limit_mhz 120` | Project the output onto d's band (d has < 2e-4 of its power outside ±120 MHz). +0.3 dB on `lut_ls` and `gated`. |
| adopted | `gated`, 3 layers (`--layers 3`) | vs d −18.81 / −15.61 / −20.08 dB (3 seeds), +0.3 / +0.4 / +0.7 over 2 layers. Now the default. [report](reports/cpu-baselines.typ) |
| **adopted, best** | `gated`, a gate MLP per band (`--per_band`) | vs d −19.35 / −15.73 / −22.82 dB (3 seeds, 219K params), +0.5 / +0.1 / +2.7 over the shared gate. A wider shared gate (`--hidden 256`, 153K) did *not* help, so the gain is from separating the bands, not from size. Now the default. Ensemble of 3 seeds: **−19.91 / −16.11 / −23.51**. [report](reports/cpu-baselines.typ) |
| promising | `gated_tcn`: dilated conv gate | g is a 1D conv net over the per-sample invariant series (weights shared in time, receptive field ±15…31). 64 ch × 5 levels, 86K params: −19.78 / −15.84 / −22.24 (1 seed). 32 ch × 4 levels, **18.5K params**: −18.41 / −15.12 / −19.45 (150 epochs), i.e. +7.9 / +7.2 / +11.5 dB over Huawei's LUT at about its size. Needs seeds; the lead for the parameter KPI. [report](reports/cpu-baselines.typ) |
| done | seed ensembles | Averaging 3 seeds: +0.35…+0.7 dB. The errors of two seeds correlate 0.77–0.88: most of the residual is shared by all models. |
| worse | wider invariant windows | `--max_lag_diff` 5 / 10 / 20, `--inv_memory 20`, `--memory 6`: equal or up to −3 dB worse (the flat MLP gets harder to optimize; no overfitting, train ≈ test). |
| worse | `--residual_lut` | gated on the residual of `lut_ls`: −16.4 / −12.9 / −15.0 dB, 2–4 dB worse than gated alone. |
| open question | noise floor | Valleys of d's in-band PSD put a white-noise floor near −19…−20 dB (A), −13…−20 (B), −25 (C) vs d, an upper bound for the noise. Band A may already be at it; C has room. |
| **KPI** | `gated_tcn` size sweep (parameter KPI) | Gain over `lut_ls` (16 848 params) vs d, 150 epochs, seed 0: 8 ch × 4 levels, 1.6K params (−91%): +1.6 / +2.0 / +5.6 dB; 12 × 4, 3.1K (−81%): +4.1 / +4.0 / +7.4; **16 × 4, 5.2K (−69%): +5.0 / +5.3 / +8.6**; 16 × 5, 6.3K (−63%): +5.6 / +5.2 / +9.3; 24 × 4, 10.8K (−36%): +7.0 / +6.7 / +10.3; 24 × 5, 13.2K (−22%): +7.5 / +6.7 / +11.2. 3 seeds: **16 × 4 (5.2K, −69%): −15.83 / −13.25 / −16.54 dB, mean gain +5.3 / +5.3 / +8.6, worst seed +5.0 / +5.3 / +8.4**; 16 × 5 (6.3K, −63%): mean gain +5.7 / +5.3 / +9.2, worst seed +5.6 / +5.1 / +8.9. The −50% parameter KPI together with +5 dB on every band holds on every seed from 16 channels on. Parameters only: per-sample MACs of a TCN are far above a LUT's. [report](reports/cpu-baselines.typ) |
| idea | full CGENN on Cl(6,0) | Vector in ℝ⁶ ≅ ℂ³, fixed bivectors e12, e34, e56 as inputs break O(6) to exactly U(1)³. Needs a GPU. |

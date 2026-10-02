#set page(margin: 2cm)
#set text(size: 10pt)
#set heading(numbering: "1.")

= DPD: first CPU round — U(1)#super[3]-equivariant models vs Huawei's LUT

*Branch* `dpd-cpu-baselines`. *Data* `GeoData_TB.mat`: 221K complex samples per band,
$F_s = 368.64$ MHz, carriers 1842.3 / 2140.2 / 2655 MHz. Train on the first 80%, test on the last 20%.
Everything ran on a 22-core CPU; the longest fit took 7 minutes.

== The symmetry
Rotating the baseband of band $b$ by $e^(i alpha_b)$ rotates $d_b$ by the same phase, independently for
each band: the group is $U(1)^3$, not the $S O(3)$ of the Pauli embedding
$x_A sigma_x + x_B sigma_y + x_C sigma_z$. Two checks:
- up to order 7, only two intermodulation products of the other bands land in band A (both order 5, at ±81 MHz);
- adding 70 symmetry-breaking regressors ($overline(x)_A$, $x_B$, $x_A overline(x)_B x_C$, ...) to an LS polynomial model changes test NMSE by 0.00 dB.

== Models
- `lut_ls` (Huawei's reference): $d_c(k) = sum_((m,l)) x_c(k-m) "LUT"(|x_A(k-l)|, |x_B(k-l)|, |x_C(k-l)|)$,
  Chebyshev tensor of order 5, $m in -3..3$, least squares.
- `lut_cp`: the same 3D LUT as a rank-$R$ CP product of 1D Chebyshev functions, Adam.
- `mlp_poly`: the team's first model — Re/Im of $x, x|x|^2, x|x|^4$ over 20 taps into an MLP; not equivariant.
- `gated`: $d_c(k) = sum_(m=-3)^3 x_c(k-m) g_(c,m)("inv"(k))$, $g$ an MLP (2 × 128) on $U(1)^3$ invariants
  over taps $k-10..k+10$: $|x_b(k-i)|^2$ and (cross-time) Re/Im of $x_b(k-i) overline(x_b(k-j))$,
  $1 <= j-i <= 2$. In $C l(6,0)$ terms this is the scalar + $e_(12)$ bivector part of the geometric
  product of two vectors in the plane of band $b$. Equivariance holds by construction (unit test).

== Results (test NMSE, dB; mean of 3 seeds, std ≤ 0.21 dB)

#table(
  columns: 8,
  align: (left, right, right, right, right, right, right, right),
  [*model*], [*params*], [*A vs d*], [*B vs d*], [*C vs d*], [*A vs x*], [*B vs x*], [*C vs x*],
  [`lut_ls` (Huawei)], [16 848], [−10.56], [−7.94], [−7.92], [−21.59], [−23.76], [−19.99],
  [`lut_cp` R=4], [5 616], [−10.88], [−8.46], [−8.16], [−21.91], [−24.27], [−20.23],
  [`lut_cp` R=8], [11 232], [−10.89], [−8.44], [−8.20], [−21.91], [−24.25], [−20.26],
  [`lut_cp` R=16], [22 464], [−10.86], [−8.41], [−8.19], [−21.89], [−24.22], [−20.26],
  [`mlp_poly`], [63 494], [−12.77], [−10.30], [−11.75], [−23.80], [−26.11], [−23.82],
  [`gated`, magnitudes], [30 122], [−17.23], [−11.46], [−18.47], [−28.26], [−27.27], [−30.53],
  [*`gated`, + cross-time*], [60 074], [*−18.48*], [*−15.24*], [*−19.42*], [*−29.51*], [*−31.05*], [*−31.49*],
  [Huawei eRef], [—], [—], [—], [—], [−33.53], [−33.46], [−30.98],
)

== Findings
+ *Normalization.* Huawei's LUT gives −20…−24 dB vs x, matching "GMP/LUT gives more than 20 dB":
  Huawei normalizes by x. Vs d the same model is only −8…−11 dB. The team's earlier −13.9 / −11.9 / −13.8 dB
  (vs d) is about −25 / −28 / −26 dB vs x.
+ *Right symmetry wins.* `gated` with magnitudes only beats the non-equivariant MLP by
  +4.5 / +1.2 / +6.7 dB with half the parameters.
+ *Phase changes between taps matter.* The cross-time invariants add +1.3 / +3.8 / +1.0 dB; band B
  (a narrow two-carrier signal whose d comes mostly from the other bands) gains most. Huawei's LUT
  sees magnitudes only and cannot use them.
+ *Parameter KPI.* The rank-4 CP LUT has 67% fewer parameters than Huawei's LUT and is 0.2–0.5 dB better.
+ *Band limit.* $d$ has < 2·10#super[−4] of its power outside ±120 MHz; the error of every model leaks there
  (figures). Projecting the output onto ±120 MHz (`--band_limit_mhz 120`) gains +0.3 dB (flag, off).
+ eRef (the PA error after Huawei's own DPD, a different quantity) is still 4.0 / 2.4 dB below `gated` on A / B;
  on C `gated` is 0.5 dB below it.

#figure(image("cpu-baselines-lut_ls_psd_B.png", width: 80%), caption: [`lut_ls`, band B: the error (green) follows d.])
#figure(image("cpu-baselines-gated_psd_B.png", width: 80%), caption: [`gated` + cross-time, band B: in-band error at the eRef level.])

== Round 2: looking for better variants

All `gated` runs below use the cross-time invariants. Seed 0 unless a mean is given.

#table(
  columns: 5,
  align: (left, right, right, right, right),
  [*variant*], [*params*], [*A vs d*], [*B vs d*], [*C vs d*],
  [base: 2 layers, shared gate], [60 074], [−18.58], [−15.30], [−19.19],
  [`--max_lag_diff 5`], [99 242], [−18.25], [−15.24], [−19.25],
  [`--max_lag_diff 10`], [149 162], [−17.96], [−15.06], [−18.97],
  [`--max_lag_diff 20`], [191 402], [−17.15], [−14.50], [−16.74],
  [`--inv_memory 20 --max_lag_diff 5`], [183 722], [−15.17], [−12.83], [−14.80],
  [`--memory 6`], [64 718], [−18.12], [−15.15], [−18.47],
  [`--hidden 256`], [152 874], [−17.99], [−15.11], [−18.01],
  [`--epochs 200`], [60 074], [−18.69], [−15.49], [−19.75],
  [`--layers 3` (3 seeds)], [76 586], [−18.81], [−15.61], [−20.08],
  [`--layers 3 --epochs 200`], [76 586], [−18.98], [−15.76], [−20.36],
  [`--layers 3 --residual_lut`], [93 434], [−16.38], [−12.87], [−14.97],
  [*`--layers 3 --per_band` (3 seeds)*], [218 922], [*−19.35*], [*−15.73*], [*−22.82*],
  [ensemble of 3 × `--layers 3`], [3 × 76 586], [−19.35], [−15.96], [−20.82],
  [*ensemble of 3 × `--per_band`*], [3 × 218 922], [*−19.91*], [*−16.11*], [*−23.51*],
  [`gated_tcn` 32 ch × 4, 60 ep], [18 538], [−17.35], [−14.40], [−17.92],
  [`gated_tcn` 32 ch × 4, 150 ep], [18 538], [−18.41], [−15.12], [−19.45],
  [`gated_tcn` 64 ch × 5, 100 ep], [86 314], [−19.78], [−15.84], [−22.24],
)

Findings:
+ *Separate the bands.* A gate MLP per output band gives +0.5 / +0.1 / +2.7 dB, while a twice-wider shared
  gate gives nothing: the bands interfere in a shared network. Adopted (`per_band` and `layers = 3` are
  the defaults now). The best result, an ensemble of 3 seeds, is −19.9 / −16.1 / −23.5 dB vs d
  (−30.9 / −31.9 / −35.6 vs x; band C is 4.6 dB below Huawei's eRef).
+ *Bigger flat windows hurt.* More lag pairs or taps only add inputs the MLP optimizes worse; train and test
  losses stay equal, so this is not overfitting.
+ *A conv gate is the parameter-efficient form.* `gated_tcn` shares its weights over time. With 18.5K
  parameters (Huawei's LUT: 16.8K) it is 7–11 dB better than the LUT and on par with the 60K-parameter
  flat gate; with 86K it is the best single model on A and B.
+ *Residual on the LUT hurts* by 2–4 dB: fitting the LS LUT first leaves a residual that is harder to learn.
+ *The residual is shared.* Errors of two seeds correlate 0.77–0.88. The in-band valleys of the PSD of $d$
  bound its white-noise floor at about −19…−20 dB (A), −13…−20 (B), −25 (C) vs d: band A may already be
  near the noise, band C is not.

== Round 3: the parameter KPI

Reference: `lut_ls`, 16 848 real parameters, −10.56 / −7.94 / −7.92 dB vs d. The SoW asks for −30% and
−50% parameters and, finally, +5 dB. `gated_tcn`, 150 epochs, seed 0; the gain is over `lut_ls`, vs d.

#table(
  columns: 6,
  align: (left, right, right, right, right, right),
  [*TCN (ch × levels)*], [*params*], [*vs `lut_ls`*], [*A gain*], [*B gain*], [*C gain*],
  [8 × 4], [1 594], [−91%], [+1.6], [+2.0], [+5.6],
  [8 × 5], [1 866], [−89%], [+2.0], [+2.4], [+5.7],
  [12 × 4], [3 138], [−81%], [+4.1], [+4.0], [+7.4],
  [12 × 5], [3 738], [−78%], [+4.3], [+4.1], [+8.3],
  [*16 × 4*], [*5 194*], [*−69%*], [*+5.0*], [*+5.3*], [*+8.6*],
  [16 × 5], [6 250], [−63%], [+5.6], [+5.2], [+9.3],
  [24 × 4], [10 842], [−36%], [+7.0], [+6.7], [+10.3],
  [24 × 5], [13 194], [−22%], [+7.5], [+6.7], [+11.2],
  [32 × 4], [18 538], [+10%], [+7.8], [+7.2], [+11.5],
  [64 × 5 (100 epochs)], [86 314], [+412%], [+9.2], [+7.9], [+14.3],
)

Seeds 1 and 2 of the two 16-channel nets: 16 × 4 has mean gain +5.3 / +5.3 / +8.6 and worst seed
+5.0 / +5.3 / +8.4 dB (−15.83 / −13.25 / −16.54 dB vs d); 16 × 5 has mean +5.7 / +5.3 / +9.2 and worst
+5.6 / +5.1 / +8.9 dB.

Every size from 8 channels on beats the LUT on every band; from 16 channels on (5.2K parameters, −69%) the
gain is ≥ +5 dB on every band. This covers the parameter KPI only: a conv net applies each weight once per sample, so a 16 × 4
TCN spends ~5K multiply-accumulates per sample, against a few hundred for an interpolated 3D LUT
(13 terms × 8 corners). The computation KPI needs its own round (distillation into a LUT, pruning).

== Next
- `gated_tcn` per band, with seeds; its size sweep for the −30 / −50% parameter KPIs.
- Noise floor: ask Huawei how d was measured (averaging, number of iterations) to know what is reachable.
- Full CGENN on $C l(6,0)$ with fixed bivectors $e_(12), e_(34), e_(56)$ (exactly $U(1)^3$-equivariant); GPU.
- Ask Huawei for `nmse.m` to confirm the normalization.

Not done: MAC counts; the Typst file was not compiled (no `typst` on the machine).

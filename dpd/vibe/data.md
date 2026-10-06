# Data reference

Neither file is in git — the repo's `.gitignore` excludes `data/`, and there is no LFS.
Put both in `dpd/data/`, or pass `--data`.

## `GeoData_TB.mat` (23 MB)

A tri-band concurrent PA capture, already reduced to model-ready form. This is the data the
headline results refer to.

| variable | shape | meaning |
|---|---|---|
| `xA`, `xB`, `xC` | `(1, 221000)` complex | input envelope of each carrier |
| `dA`, `dB`, `dC` | `(1, 221000)` complex | desired signal: the band-limited nonlinear residual to fit |
| `eRefA`, `eRefB`, `eRefC` | `(1, 221000)` complex | residual of the vendor's own model, for comparison |
| `FsLow` | int32 | 368640000 Hz |
| `fa`, `fb`, `fc` | | carrier frequencies: 1.8423, 2.1402, 2.655 GHz |

Per-band figures:

| band | carrier | peak &#124;x&#124; | occupied BW | no-model NMSE | vendor reference |
|---|---|---|---|---|---|
| A | 1.8423 GHz | 0.4669 | 74.9 MHz | -10.892 dB | -33.451 dB |
| B | 2.1402 GHz | 0.4640 | 59.9 MHz | -15.800 dB | -33.475 dB |
| C | 2.6550 GHz | 0.6298 | 69.8 MHz | -11.999 dB | -30.699 dB |

The carriers are mutually uncorrelated (peak cross-correlation at most 0.021 over +/-40 lags, in
both the complex signals and their envelopes), so these are three genuinely independent carriers,
not polyphase components of one.

Spacings are 298 MHz (A–B), 515 MHz (B–C) and 813 MHz (A–C). This shows up directly in the
modelling: `|x_B|` moves band A from -14.3 dB to -24.1 dB, while `|x_C|` gains 0.1 dB.
Cross-modulation falls off with carrier separation.

```python
from dpd import load_geodata_tb
geo = load_geodata_tb("data/GeoData_TB.mat")
x, scales = geo.normalised_stack("AB")   # (2, 221000), rows scaled to unit peak
band = geo.bands["A"]
band.x, band.d, band.e_ref              # raw, unnormalised
```

`normalised_stack` is required before fitting: the Chebyshev basis is indexed by
`round(|x| * 0.99 * 2^15)` and runs off its lookup table otherwise. Keep `band.x` (the
**unnormalised** signal) as the NMSE reference — see [metrics.md](metrics.md) for why this
matters.

## `DOV2.mat` (15 MB)

The single-band capture used by `NonLinearProblemSimple.m`. Raw, not reduced.

| variable | shape | meaning |
|---|---|---|
| `PDin` | `(1, 245760)` complex | predistorter input |
| `PDout` | `(1, 245760)` complex | observed output |
| `PDdpd` | `(1, 245760)` complex | predistorter output |
| `PDerr` | `(1, 245760)` complex | |
| `FB` | `(2, 262144)` complex | |

`dpd.data.preprocess_dov2` reproduces the front end of `NonLinearProblemSimple.m`: upsample
by 2 through a 4097-tap half-band filter, form the error `PDdpd - PDout`, take the first
polyphase branch, band-limit with `BL`, and scale by `2^-15`.

**The two data sets mean different things by "dimension".** For `GeoData_TB` the `D`
envelopes are different carriers. For `DOV2` they are the two polyphase branches of a
2x-upsampled *single* carrier, i.e. the signal and a half-sample-delayed copy of it — a
fractional-delay trick that gives the envelope half-sample resolution. The same model code
runs on both, but a `D = 2` result means something different in each case.

## Splits

`dpd.data.Split.contiguous(n, train_frac=0.8, guard=512)` gives a contiguous 80/20 split with
a guard gap. The guard matters because memory kernels reach ~10 samples and `BL` another 16;
512 removes any leakage at negligible cost at `n = 221000`.

The MATLAB script reports an **in-sample** figure. On this data that turns out not to
flatter it — the A+B fit scores -24.123 dB on train and -24.139 dB on validation, so 1215
coefficients over 221000 samples is not overfitting. At `D = 3` the gap opens to ~1 dB with
the pseudo-inverse, which is what motivates the ridge default.

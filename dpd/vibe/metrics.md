# Metrics: how to read the numbers

## NMSE is negative here, and lower is better

Every NMSE figure in this project is the one `SimpleNonLinearModel_ML.m` prints:

```
NMSE_dB  =  10 * log10( sum|e|^2 / sum|xRef|^2 )
```

where `e` is the model residual and `xRef` is the **input carrier** of the band being
modelled. The residual is far weaker than the carrier, so the ratio is well under 1 and the
logarithm is negative. `-24.7 dB` means the leftover distortion sits 24.7 dB below the
carrier.

**Lower is better.** Going from `-14.3 dB` to `-24.7 dB` is a 10 dB improvement, even though
the number got smaller. Landmark values on `GeoData_TB`, band A, worst first:

| | NMSE | what it is |
|---|---|---|
| no model (predict zero) | -10.892 dB | the raw distortion, nothing removed |
| band A envelope only | -14.341 dB | 1-D nonlinearity |
| bands A + B | -24.121 dB | the headline result |
| bands A + B + C, ridge | -24.473 dB | 0.33 dB more for 2.8x the coefficients |
| vendor's own model (`eRefA`) | -33.451 dB | shipped with the capture; still ahead |

Informally, PA and DPD practitioners usually quote the magnitude — "24.7 dB NMSE" means
`-24.7 dB`. Both refer to the same quantity; only the sign convention differs.

## Two normalisations, and how to convert

The convention above divides by the **carrier** power. Most of the published PA-modelling
literature divides by the **desired signal** power instead:

```
NMSE_vs_d  =  10 * log10( sum|d - y|^2 / sum|d|^2 )
```

That asks "what fraction of the distortion did the model capture?", whereas the carrier-
referred version asks "how far below the carrier does the residual sit?". They differ by
exactly the no-model floor:

```
NMSE_vs_d  =  NMSE_vs_x  -  NMSE_no_model
           =  -24.121  -  (-10.892)
           =  -13.229 dB
```

So the A+B fit is "-24.1 dB" in this project's convention and "-13.2 dB" in the other one.
Neither is wrong; quote the second if you are comparing against a paper.

`dpd.metrics.summarise` returns both:

```python
{'nmse_model_db': -24.121,      # carrier-referred, what MATLAB prints
 'nmse_no_model_db': -10.892,   # the floor
 'nmse_reference_db': -33.451,  # the vendor's model, when eRef is available
 'improvement_db': 13.229}      # = nmse_no_model_db - nmse_model_db
```

`improvement_db` is how far the model moved off the floor, so the desired-referred figure is
just its negation:

```
NMSE_vs_d  =  nmse_model_db - nmse_no_model_db  =  -improvement_db  =  -13.229 dB
```

## A trap worth knowing about

The rows of `x` must be scaled to unit peak before the model sees them, because the
Chebyshev basis is indexed by `round(|x| * absScale * 2^15)` and would otherwise run off the
end of its lookup table. But `xRef` is captured **before** that scaling — `NonLinearProblemSimple.m`
does `xRef = x(1,:)` above the normalisation loop.

Normalising divides band A by its peak of 0.4669, so using the normalised row as the NMSE
reference inflates the reference power by `(1/0.4669)^2` and shifts every figure by 6.6 dB.
This is easy to do by accident, and it is why `dpd.training.train` takes `x_ref` as a
separate argument rather than reading it off `x[0]`.

## NMSE is not sufficient on its own

NMSE is a total-power metric. It says nothing about *where* the residual sits in frequency,
and for a PA that is usually the question that matters — out-of-band shoulders are what
spectral masks regulate. Measured on band A with a 75 MHz occupied bandwidth and a 90 MHz
offset:

| signal | ACPR |
|---|---|
| input carrier `xA` | -74.91 dB |
| desired distortion `dA` | +1.93 dB |
| residual after the A+B model | +4.17 dB |
| vendor reference residual `eRefA` | +2.63 dB |

The input is spectrally clean. The distortion is not: it has slightly more power in the
adjacent band than in-band, which is what distortion products look like. And the model's
residual is *more* adjacent-band-dominated than the distortion it started from — the model
removes 13 dB of total error power but preferentially removes the in-band part.

That is a real limitation of fitting to NMSE alone, and it is visible only spectrally. Use
`dpd.metrics.welch_psd` and `acpr_db` alongside the NMSE figure:

```python
from dpd.metrics import welch_psd, acpr_db

f, psd_db = welch_psd(residual, nfft=2048, fs=geo.fs_hz)
acpr = acpr_db(residual, occupied=75e6, offset=90e6, nfft=2048, fs=geo.fs_hz)
```

`acpr_db` is defined as adjacent-band power over in-band power, so a *negative* value means
the adjacent band is quieter — the opposite direction from NMSE. The original MATLAB
equivalent is `plot_psd4.m`, kept in `matlab/`.

## Occupied bandwidth of the three carriers

Measured from the Welch PSD at both -30 dB and -40 dB below the peak (the two agree to
0.4 MHz, so the carriers have sharp edges):

| band | occupied bandwidth | as a fraction of `FsLow` |
|---|---|---|
| A | 74.9 MHz | 0.203 |
| B | 59.9 MHz | 0.163 |
| C | 69.8 MHz | 0.189 |

`BL`, the band-limitation filter applied to every basis term, is `fir1(32, 0.36*2, 'low')` —
a cutoff at `0.36 * FsLow = 132.7 MHz`. So the model is allowed to generate distortion
roughly 1.8x wider than the band A carrier, but no wider.

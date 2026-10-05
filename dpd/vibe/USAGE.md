# dpd -- setup and running

> Part of [`vibe/`](.): written by Claude, not by hand. `dpd/README.md` is the
> human-authored idea board.

A PyTorch interface for the supplied MATLAB PA behavioural model, a verified port of it,
and a first geometric-algebra model built on the same data, metrics and splits. The idea
board and current scores are in [`README.md`](README.md).

Two things live here:

1. **A faithful port of the reference model** (`SimpleNonLinearModel_ML.m`) — a Chebyshev
   memory polynomial identified by least squares. Verified against the original `.m` code
   running in Octave: same NMSE to 4e-3 dB, same coefficients to 0.5%.
2. **A Clifford-equivariant model** (`dpd.clifford_model`) that keeps the outer structure
   of the reference model but replaces its hand-designed envelope basis with a
   Clifford-group-equivariant network. Working, tested, and **not yet competitive** — see
   [Status](#status-of-the-clifford-model).

## Install

Needs **Python 3.11–3.14**. Like every experiment folder here this is a self-contained Poetry
project, but it installs with plain pip just as well. From `dpd/`:

```bash
poetry install --extras dev          # or: poetry install --all-extras
poetry run pytest                    # 42 tests, ~5 s, no data needed
```

or, with pip (needs **pip >= 21.3**, see troubleshooting below):

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # any of 3.11-3.14
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

Runtime dependencies are just `numpy`, `scipy` and `torch`; `[dev]` adds `pytest`. Three more
extras are optional and nothing requires them: `[logging]` for Weights & Biases (`--wandb`),
`[plots]` for matplotlib, `[notebooks]` for the tour below. Install several with
`pip install -e ".[dev,notebooks]"`.

Note this project uses plain setuptools, not Poetry — unlike `3D Pose experiemtns/` in the
same repo. There is no shared environment between the two.

**The data is not in git.** The repo's `.gitignore` excludes `data/` and there is no LFS, so
the two `.mat` files have to be placed by hand:

```
dpd/data/GeoData_TB.mat     # 23 MB — every result below uses this
dpd/data/DOV2.mat           # 15 MB — only for dpd.data.preprocess_dov2
```

Or pass `--data /path/to/GeoData_TB.mat`. The test suite generates its own signals, so
`pytest` passing confirms the port is intact before you have the data.

Then the one command that reproduces the headline number (~30 s, prints `-24.121 dB`):

```bash
python scripts/reproduce_matlab.py --band A --bands AB --n-basis 8,8 --holdout
```

Octave is needed only to re-run the cross-check against the original `.m` files; see
[`reference-model.md`](reference-model.md#verification-against-the-original-code).

### Troubleshooting the install

Check both versions before anything else — most install failures here are one of these two:

```bash
python -c "import sys; print(sys.version)"
pip --version
```

**`File "setup.py" or "setup.cfg" not found. Directory cannot be installed in editable mode`**
Your pip predates PEP 660, which is what lets a `pyproject.toml`-only project be installed
with `-e`. That landed in pip 21.3 (October 2021). Fix with
`python -m pip install --upgrade pip`.

**`Requires-Python >=3.11,<3.15`, or the above on macOS**
pip 21.2.4 in particular is what macOS bundles with its system Python 3.9.6, which is below
this project's floor — so upgrading pip there just exposes the Python version as the next
error. The venv has to be built on a newer interpreter, not the system one:

```bash
brew install python@3.11              # or pyenv install 3.11, or use uv
rm -rf .venv
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

`uv venv --python 3.11 && uv pip install -e ".[dev]"` does the same thing and brings its own
modern resolver, if you would rather not touch the system Python at all.

**Not wanting to install at all.** Nothing here needs the package on the path — everything
runs from source with `numpy`, `scipy` and `torch` present:

```bash
PYTHONPATH=src python scripts/reproduce_matlab.py --band A --bands AB --n-basis 8,8
PYTHONPATH=src python -m pytest tests/
```

---

## Documentation

| | |
|---|---|
| [`metrics.md`](metrics.md) | Why NMSE is negative, which normalisation is in use, and what NMSE does not tell you |
| [`reference-model.md`](reference-model.md) | The MATLAB model line by line, the port's exact reorganisations, conditioning, porting assumptions |
| [`clifford-model.md`](clifford-model.md) | The geometric-algebra formulation, the equivariance argument, current status and next steps |
| [`data.md`](data.md) | What is in the two `.mat` files, and the splits |
| [notebooks/01_explore.ipynb](../notebooks/01_explore.ipynb) | **Start here** — a runnable tour of the data, metrics, reference model and Clifford model, with outputs already in the file |

---

## The data

`GeoData_TB.mat` is a tri-band concurrent PA capture, already reduced to model-ready form:
221000 complex baseband samples per carrier at `FsLow = 368.64 MHz`.

| band | carrier | no-model NMSE | vendor reference NMSE |
|------|---------|---------------|-----------------------|
| A    | 1.8423 GHz | -10.892 dB | -33.451 dB |
| B    | 2.1402 GHz | -15.800 dB | -33.475 dB |
| C    | 2.6550 GHz | -11.999 dB | -30.699 dB |

For each band the file carries `x*` (input envelope), `d*` (the band-limited nonlinear
residual to be fitted) and `eRef*` (the residual of the vendor's own model, for reference).
"No-model NMSE" is what you score by predicting zero; the vendor reference is the target to
beat.

> **Reading the numbers.** All NMSE figures are `10*log10(sum|e|^2 / sum|xRef|^2)` — residual
> power relative to the *carrier*. They are negative, and **lower is better**: `-24.7 dB` beats
> `-14.3 dB`. Most of the literature normalises by the desired signal instead, which shifts
> every figure by the no-model floor. [`metrics.md`](metrics.md) has the conversion and
> the reasons NMSE alone is not enough for a PA.

`DOV2.mat` is the single-band capture used by `NonLinearProblemSimple.m`. There the two
"dimensions" of the nonlinearity are half-sample fractional delays of one carrier, not
different carriers; `dpd.data.preprocess_dov2` reproduces that front end.

**The data files are not in git** (the repo's `.gitignore` excludes `data/`, and there is no
LFS). Put `GeoData_TB.mat` and `DOV2.mat` in `dpd/data/`, or pass `--data`.

---

## The reference model

```
y(k) = BL * sum_m sum_{n_1..n_D} c_{m,n} * x_1(k - s_m) * prod_p T_{n_p}(|x_p(k - l_m - u_{p,m})|)
```

`T_n` are Chebyshev polynomials of the first kind, evaluated through a 2^15-entry lookup
table indexed by quantised envelope magnitude. `BL` is a 33-tap lowpass that limits the
bandwidth of the nonlinearity. The delays `[s; l; u_1..u_D]` come from a *part model*: a
`(2+D, M)` integer matrix, `M = 15` hand-tuned columns in the supplied structure.
Coefficients are identified by least squares on the normal equations, `pinv(U'U) * U'd`.

### Results (band A, 15 parts, Chebyshev order 8 per dimension)

| modelled band | envelopes fed to the nonlinearity | coefficients | NMSE (in-sample) | train | val |
|---|---|---|---|---|---|
| A | A       |  135 | -14.341 dB | -14.297 | -14.517 |
| A | **A, B** | **1215** | **-24.121 dB** | -24.123 | -24.139 |
| A | A, C    | 1215 | -14.433 dB | -14.462 | -14.314 |
| A | A, B, C (order 8,4,4, ridge) | 3375 | -24.813 dB | -24.906 | -24.473 |
| B | B, A    | 1215 | -23.236 dB | -23.513 | -22.259 |
| C | C, A    | 1215 | -23.071 dB | -23.112 | -22.957 |
| C | C, B    | 1215 | -22.581 dB | -22.668 | -22.291 |

This is the ~20 dB figure: **-24.1 dB for band A**, from a -10.9 dB starting point.
`train`/`val` are an 80/20 contiguous split with a 512-sample guard; they agree with the
in-sample number, so the LS fit is not overfitting despite 1215 coefficients.

Verification against the original code (`matlab/run_geodata.m` under Octave):

| | NMSE | coefficient agreement |
|---|---|---|
| Octave, original `.m` files | -24.15657 dB | — |
| this port | -24.15318 dB | 0.5% relative |

### Three things worth knowing

**Band B matters for band A; band C does not.** Adding `|x_B|` moves band A from -14.3 dB to
-24.1 dB; adding `|x_C|` instead gains 0.1 dB. The carriers are 298 MHz (A–B) and 813 MHz
(A–C) apart, so this is cross-modulation falling off with separation, and it means a 2-D
nonlinearity on the right pair buys almost everything a 3-D one would.

**Most of the gain is at low polynomial order.** Sweeping the Chebyshev order on the A,B
model:

| order per dimension | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| coefficients | 60 | 135 | 240 | 375 | 735 | 1215 |
| NMSE (dB) | -18.809 | -22.243 | -23.084 | -23.682 | -24.028 | -24.153 |

Order 2 with 135 coefficients is within 1.9 dB of order 8 with 1215. The delay structure,
not the polynomial order, is doing most of the work.

**The normal equations are badly conditioned, and it costs real dB out of sample.** `U'U` has
condition number ~1e21 at D = 2 and ~4e21 at D = 3, with numerical rank 1168 of 1215 and 3269
of 3375. The prediction stays stable — that is what the pseudo-inverse is for — but the
coefficient vector is only defined up to the numerical null space, which is why reproducing
MATLAB's coefficients requires matching its `pinv` tolerance rule rather than NumPy's.

At D = 3 this costs 0.6 dB of held-out NMSE: `pinv` buys 0.08 dB in-sample and gives it back
with interest out of sample, with a coefficient vector two orders of magnitude larger in
norm. Prefer `--solver ridge` there. Full numbers and the reasoning in
[`reference-model.md`](reference-model.md#conditioning-and-the-solver).

Note that going from two carriers to three buys only 0.33 dB out of sample (-24.14 → -24.47)
for 2.8x the coefficients — consistent with band C being nearly irrelevant to band A.

---

## The Clifford model

A baseband PA model must be phase equivariant: rotating the input constellation rotates the
output the same way.

```
x(k) -> x(k) e^{j phi}   implies   y(k) -> y(k) e^{j phi}
```

Memory polynomials get this by *construction*. Every term is `x * f(|x|)` — one unpaired
carrier copy times a function of magnitudes only — which is also exactly what restricts them
to that shape.

In `Cl(2,0)` the I/Q plane is a genuine vector space, and that phase rotation is a rotor
sandwich `v -> R v R~` on `v = Re(z) e_1 + Im(z) e_2`. Two facts then do real work:

- The layers in `dpd.ga` (channel mixing with grade-wise scalar weights, the geometric
  product, gating by grade norms) are equivariant under that rotor action by construction.
- In `Cl(2,0)` the even subalgebra commutes with every rotor, so the even part of a
  multivector is **invariant** under the rotation — and it is isomorphic to `C`.

[`clifford-model.md`](clifford-model.md) derives both and gives the per-layer
equivariance argument.

So the even part of the network's output is precisely "a complex gain that does not change
when you rotate the constellation": the AM/AM + AM/PM characteristic, obtained as a symmetry
property rather than by restricting the basis to `|x|`. `CliffordPAModel` keeps the
reference model's outer form,

```
y(k) = BL * sum_i g_i(k) * x_1(k - s_i)
```

and computes the gains `g_i` with an equivariant trunk over the delayed envelopes of *all*
carriers as multivectors. It is strictly more expressive than the reference: `g_i` may
depend on relative phases between bands and taps, not just magnitudes, while the whole model
stays exactly phase-equivariant. `tests/test_ga.py` checks the equivariance of every layer
and of the assembled model to 1e-10.

### Status of the Clifford model

**It works and is exactly equivariant, but it does not yet beat the baseline.** Honest
numbers on band A from `x_A, x_B`, 80/20 split:

| model | parameters | val NMSE |
|---|---|---|
| no model | — | -10.892 dB |
| reference memory polynomial | 1215 complex | **-24.139 dB** |
| Clifford model, 600 steps | 26691 real | -12.221 dB |

A diagnostic narrows down where the gap is. The output is exactly *linear* in the readout
parameters, so the best possible readout on a given trunk can be solved in closed form:

| trunk | optimal-readout val NMSE |
|---|---|
| random, 16 channels / 2 blocks | -11.475 dB |
| random, 16 channels / 2 blocks, 13 linear taps | -11.495 dB |
| random, 32 channels / 3 blocks, 13 linear taps | -11.520 dB |

A random trunk supports -11.5 dB and training reaches -12.2 dB, so the trunk is producing
almost nothing the readout can use. This is a representation problem in the trunk, not an
optimisation or capacity problem — the reference reaches -18.8 dB with 60 coefficients.
Likely suspects, in order:

1. **Delay coverage.** The reference spends 15 hand-tuned delay triples with `s` in −2..10.
   The Clifford model uses a regular tap grid and 3 linear taps. The order sweep above says
   the delay structure carries most of the performance, so this is the first thing to fix:
   feed the model the reference part model's taps.
2. **Envelope dynamic range.** The invariants entering the gate are *squared* norms of
   unit-peak envelopes, so they concentrate near zero and the MLP sees a compressed input.
   Normalising the invariants, or feeding `|x|` rather than `|x|^2`, is worth trying.
3. **Depth vs. order.** Each geometric-product block roughly doubles polynomial degree; two
   blocks reach degree 4 against the reference's degree 8 per dimension.

The infrastructure to test all three is in place; none of it has been run.

---

## Layout

```
dpd/
├── dpd/                   # the package
│   ├── config.py          # the reference recipe and the feature flags
│   ├── matlab.py          # MATLAB-equivalent primitives (delay, conv 'same', fir1, gen_spl, nmse)
│   ├── data.py            # GeoData_TB / DOV2 loaders, the DOV2 front end, train/val split
│   ├── features.py        # part model + streaming Chebyshev regressor construction
│   ├── solvers.py         # matlab_pinv (bit-compatible) and ridge least squares
│   ├── models.py          # MemoryPolynomialPA: the reference model as an nn.Module
│   ├── ga.py              # Clifford algebra + equivariant layers (no external GA dependency)
│   ├── clifford_model.py  # CliffordPAModel
│   ├── training.py        # windowed SGD, NMSE loss, optional W&B logger, device selection
│   └── metrics.py         # NMSE, Welch PSD, ACPR
├── vibe/                  # AI-written write-ups (this file included)
├── notebooks/             # 01_explore.ipynb -- the tour
├── scripts/               # reproduce_matlab.py, train_clifford.py
├── matlab/                # the original .m files, plus delay/nmse/progress (see below)
├── tests/                 # 42 tests: MATLAB parity, algebra, equivariance, metrics
├── reports/               # one Typst report per finished experiment
├── awesome-reference/     # reading material
└── data/                  # not in git; put the .mat files here
```

### Assumptions about the MATLAB code

Four things in this port are inferred rather than supplied, and any of them could be wrong:

- **`delay.m`, `nmse.m`, `progress.m`** are not in the archive — they live on the original
  author's MATLAB path. The versions in `matlab/` are ours. `delay` as a circular shift is the
  least certain and the most worth confirming.
- **`DEFAULT_PART_MODEL` row 5**, the third band's delays. `GeoData_TB` is tri-band but the
  archive only ships a 2-dimensional structure, so any `D = 3` result depends on our choice.
  Rows 1–4 are exactly as supplied.

[`reference-model.md`](reference-model.md#helpers-the-archive-does-not-ship) states
each assumption and what it would cost if wrong.

---

## Usage

```bash
# Reproduce the reference result
python scripts/reproduce_matlab.py --band A --bands AB --n-basis 8,8 --holdout --rank
python scripts/reproduce_matlab.py --sweep --holdout --out results/matlab_sweep.json

# Three-band model (prefer the ridge solver, see above)
python scripts/reproduce_matlab.py --band A --bands ABC --n-basis 8,4,4 --solver ridge --holdout

# Clifford model
python scripts/train_clifford.py --bands AB --steps 3000 --baseline --device auto
python scripts/train_clifford.py --bands AB --steps 3000 --wandb   # optional W&B logging
```

Python:

```python
from dpd import load_geodata_tb, MemoryPolynomialPA, PartModel, summarise
from dpd.data import BL_DEFAULT
import torch

geo = load_geodata_tb("data/GeoData_TB.mat")
x, _ = geo.normalised_stack("AB")          # unit-peak rows, as the LUT indexing requires
band = geo.bands["A"]

model = MemoryPolynomialPA(PartModel.default(2), [8, 8], BL_DEFAULT, quantise=True)
model.fit_least_squares(x, band.d)         # closed form, the MATLAB solution
with torch.no_grad():
    y = model(torch.as_tensor(x)).numpy()
print(summarise(band.x, band.d, y, band.e_ref))
```

The same module is differentiable, so `fit_least_squares` can be followed by gradient
descent against any objective (`dpd.training.train`), and `quantise=False` makes it
differentiable with respect to the envelope as well as the coefficients.

### Re-running the Octave cross-check

```bash
apt-get install -y octave octave-signal
cd matlab && BAND=A BANDS=AB NB="[8 8]" octave --no-gui --quiet run_geodata.m
```

---

## Performance notes

Measured on 4 CPU cores, no GPU:

| operation | time |
|---|---|
| reference LS fit, D=2, 1215 coefficients | ~30 s |
| reference LS fit, D=3, 3375 coefficients | ~175 s |
| reference forward pass, full 221k record | 0.7 s |
| Clifford forward, 8192-sample window | 39 ms |
| Clifford training, 600 steps | ~30 s |

Two reorganisations of the reference algorithm make this practical:

- **Sum before filtering.** The reference convolves each of the 1215 basis terms with `BL`
  and then takes a weighted sum. Convolution is linear, so summing first and convolving once
  gives an identical signal for 1215x less filtering work.
- **Stream the Gram matrix.** The reference materialises `U` in full. At D = 3 that is 38 GB;
  `ChebyshevLUTFeatures.gram` accumulates `U'U` over time blocks instead, identical up to
  summation order.

# vibe

Everything in this folder was written by Claude, not by hand. It is kept separate from
`dpd/README.md`, which is the human-authored idea board and the thing to trust on intent and
priorities.

Treat these as working notes: they are detailed and the numbers in them were measured, but the
prose, the framing and the judgement calls are a model's.

| | |
|---|---|
| [`USAGE.md`](USAGE.md) | Install, run, troubleshoot. The layout of the package and what each module does. |
| [`metrics.md`](metrics.md) | Why NMSE is negative and lower is better, the two normalisations and how to convert, and why NMSE alone is not enough for a PA. |
| [`reference-model.md`](reference-model.md) | The supplied MATLAB model line by line, the port's three exact reorganisations, conditioning and solver choice, the Octave cross-check, and every assumption the port makes. |
| [`clifford-model.md`](clifford-model.md) | The `Cl(2,0)` formulation with the rotor and even-subalgebra arguments, per-layer equivariance, current status and ranked next steps. |
| [`data.md`](data.md) | The two `.mat` files variable by variable, measured bandwidths and carrier spacings, and the splits. |

The runnable tour is [`../notebooks/01_explore.ipynb`](../notebooks/01_explore.ipynb); it is also
model-written, but it is committed with its outputs so every number in it can be checked against a
re-run.

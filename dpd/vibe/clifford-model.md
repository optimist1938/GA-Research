# The Clifford model

Why geometric algebra is a natural fit for this problem, what the model actually does, and an
honest account of where it currently stands.

## The symmetry a PA model must have

A baseband PA model has to be phase equivariant. Rotating the input constellation rotates the
output the same way:

```
x(k) -> x(k) e^{j phi}    implies    y(k) -> y(k) e^{j phi}
```

This is not a modelling choice; it follows from the amplifier having no phase reference of
its own. Memory polynomials satisfy it by **construction**: every term has the form
`x * f(|x|)` — one unpaired carrier copy multiplying a function of magnitudes only, which
picks up exactly one factor of `e^{j phi}`. That is also precisely what confines them to that
shape. The envelope nonlinearity can never see a phase.

## Cl(2,0) makes the symmetry structural

`Cl(2,0)` has four blades: `1`, `e1`, `e2`, `e12`, with `e1^2 = e2^2 = 1` and `e12^2 = -1`.
Embed a complex sample as a **vector**:

```
z = I + jQ    ->    v = I e1 + Q e2
```

Two facts do the work.

**A phase rotation is a rotor sandwich.** With `R = exp(-phi/2 * e12) = cos(phi/2) - sin(phi/2) e12`,

```
R v R~  =  the vector representing  z e^{j phi}
```

(`tests/test_ga.py::test_rotor_sandwich_is_a_phase_rotation`, exact to 1e-12.)

**The even part is invariant.** The even subalgebra of `Cl(2,0)` is spanned by `1` and `e12`,
is commutative, and commutes with every rotor. So for `a = alpha + beta e12`,

```
R a R~  =  a R R~  =  a
```

The even part does not move under phase rotation — and it is isomorphic to `C`, since
`e12^2 = -1`.

Put together: **the even part of a multivector is exactly "a complex number that does not
change when you rotate the constellation"**, which is exactly what a PA complex gain is. The
AM/AM and AM/PM characteristic falls out as a rotor invariant instead of being imposed by
restricting the basis to `|x|`.

## What the model computes

`CliffordPAModel` keeps the reference model's outer form,

```
y(k) = BL * sum_i g_i(k) * x_1(k - s_i)
```

and replaces the hand-designed gain with an equivariant network:

1. **Embed.** For every band `p` and every envelope tap `tau`, embed `x_p(k - tau)` as a
   `Cl(2,0)` vector. With 2 bands and 11 taps that is 22 multivector channels.
2. **Trunk.** `MVLinear` to the hidden width, then residual `MVBlock`s. Each block is
   `x + gate(geometric_product(x))`.
3. **Readout.** `MVLinear` down to one multivector per linear tap; take blades 0 and `e12`
   as a complex number. By the lemma above these are rotor-invariant, so they are gains.
4. **Apply and band-limit.** Multiply each gain by its delayed carrier copy, sum, convolve
   with `BL`.

The result is strictly more expressive than the reference: `g_i` may depend on relative
*phases* between bands and taps, not only magnitudes, while the whole model stays exactly
phase-equivariant. The reference is the special case where `g_i` is a separable polynomial in
the magnitudes.

## Why each layer is equivariant

| layer | why it commutes with the rotor action |
|---|---|
| `MVLinear` | grade projection commutes with the sandwich; weights are scalars. Bias is only on grade 0, which is invariant. |
| `MVGeometricProduct` | `R(ab)R~ = (RaR~)(RbR~)`; the product is covariant. |
| `MVGradeGate`, `MVInvariantGate` | gates are functions of grade norms, which are invariants; they rescale each grade. |
| `MVLayerNorm` | same, with the norm taken across channels. |

`tests/test_ga.py::test_layers_are_equivariant` checks all of them to 1e-10, with parameters
perturbed away from their initialisation so the test cannot pass trivially.
`test_clifford_pa_model_is_phase_equivariant` checks the assembled model, and
`test_clifford_gains_are_phase_invariant` checks the gains themselves.

Note the symmetry the model enforces is under a **common** rotation of all carriers. The
physically exact statement for concurrent multi-band is stronger — band A's distortion is
equivariant in `phi_A` and *invariant* in `phi_B` — so the model's constraint is weaker than
reality. It holds exactly; it just does not capture everything.

## Two design decisions worth knowing

**`MVLayerNorm` is off by default.** Normalising across channels divides out the overall
envelope scale. For most equivariant-network applications that is a nuisance variable; here
it is the signal — the AM/AM characteristic *is* the dependence on that scale.

**The gate is an MLP over invariants, not a per-grade sigmoid.** `MVGradeGate` squashes one
invariant per (channel, grade) and can only produce a saturating function of that one norm. A
PA's AM/AM is a high-order function of the envelope that mixes magnitudes across carriers and
delays — the reference spends Chebyshev order 8 on it. `MVInvariantGate` feeds the whole
invariant vector through a small MLP, which stays equivariant (invariants in, scalars out)
and is far more expressive.

## Current status: works, not yet competitive

Band A from `x_A, x_B`, 80/20 contiguous split with a 512-sample guard:

| model | parameters | val NMSE |
|---|---|---|
| no model | — | -10.892 dB |
| reference memory polynomial | 1215 complex | **-24.139 dB** |
| Clifford model, 600 steps | 26691 real | -12.221 dB |

That gap is large and it is not subtle. A diagnostic localises it. The output is exactly
*linear* in the readout parameters — `y = BL * sum_i g_i x_1(k - s_i)` with `g_i` a real-linear
function of `readout.weight` and `readout.bias` — so the **best possible readout on a given
trunk** can be solved in closed form by stacking real and imaginary parts:

| trunk | optimal-readout val NMSE |
|---|---|
| random, 16 channels / 2 blocks, 3 linear taps | -11.475 dB |
| random, 16 channels / 2 blocks, 13 linear taps | -11.495 dB |
| random, 32 channels / 3 blocks, 13 linear taps | -11.520 dB |

A random trunk supports -11.5 dB; training reaches -12.2 dB. The trunk is producing almost
nothing the readout can use. This is a **representation problem in the trunk**, not an
optimisation or capacity problem — the reference reaches -18.8 dB with 60 coefficients.

## Next steps, in order

1. **Delay coverage.** The reference spends 15 hand-tuned delay triples with `s` in -2..10 and
   independent `u_p` per band. The Clifford model uses a regular tap grid and 3 linear taps.
   The order sweep in [reference-model.md](reference-model.md) shows the delay structure
   carries most of the performance, so the first experiment is to feed the model the
   reference part model's taps directly.
2. **Envelope dynamic range.** The invariants entering the gate are *squared* norms of
   unit-peak envelopes, so they concentrate near zero and the MLP sees a compressed input.
   Feeding `|x|` rather than `|x|^2`, or standardising the invariants, is cheap to try.
3. **Depth versus order.** Each geometric-product block roughly doubles polynomial degree, so
   two blocks reach degree 4 against the reference's degree 8 per dimension.
4. **Warm start from the reference.** The reference model is the special case where the gain
   is a separable magnitude polynomial. Initialising the Clifford model to reproduce a fitted
   memory polynomial and fine-tuning from there would separate "can it represent the
   baseline" from "can it find the baseline".

The infrastructure for all four is in place. None of them has been run.

## Reproducing the diagnostic

The readout least-squares check is not currently a script; it is short enough to inline. Take
the trunk output blades 0 and `e12` per channel, build one regressor column per
(linear tap, channel, {real, imaginary}) plus one bias column per tap, convolve each with
`BL`, and solve the real-linear system by stacking `Re` and `Im`. See the table above for the
values it produced.

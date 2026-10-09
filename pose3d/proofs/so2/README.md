# Lean proofs for the SO(2)-equivariant image → multivector mappings

Lean 4.34.1 + Mathlib (tag `v4.34.1`). Build and audit:

```bash
cd pose3d/proofs/so2 && lake exe cache get && lake build && lake env lean Audit.lean
```

`Audit.lean` prints the axioms of every theorem the reports rely on. Each one depends on nothing
or only on Lean's standard `propext`, `Classical.choice` and `Quot.sound`. There is no `sorry`
and no new axiom.

| file | theorem | what it gives the design |
|---|---|---|
| `Symmetry.lean` | `reynolds_equivariant`, `reynolds_mean_equivariant` | Averaging any map over a finite group is equivariant. |
| | `canonicalize_equivariant`, `canonicalize_of_canonical` | Variant B: an equivariant canonicaliser makes any frozen model equivariant, and leaves it unchanged on images that are already canonical. |
| | `orbit_argmax_equivariant` | Variant B: argmax of any score over the four turns is an equivariant canonicaliser (away from ties). |
| | `euler_equivariant` | The Euler sampler on the rotor group with an invariant body velocity commutes with the roll at every step. |
| | `body_velocity_invariant` | The frame read-out `r̃ w r` is invariant when `r ↦ g r` and `w ↦ g w g̃`. |
| | `medoid_cost_invariant` | Medoid selection commutes with left translation. |
| | `equivariant_error_invariant` | The error on a turned test pair equals the upright error, sample by sample. |
| | `fixed_input_forces_symmetric_output` | The price of exact equivariance: a turn-invariant image gets a turn-invariant output. |
| `Cl3.lean` | `sandwich_rotZ`, `sandwich_roll` | Cl(3,0) in the repo's blade order: a roll about e3 fixes 1, e3, e12, e123 and rotates (e1,e2) and (e13,e23) by θ. |
| | `rotZ_unit`, `rotZ_mul` | Roll rotors are unit rotors and compose by adding their angles. |
| `Harmonics.lean` | `lift_equivariant` | Variant A: lifting by input rotation gives the regular representation for *any* backbone. |
| | `dft_shift` | Variant A: the k-th DFT coefficient over the views is a frequency-k field. |
| | `rot_commutant` | The rotation-commuting real 2×2 maps are the complex scalars, i.e. the head's complex weights. |
| | `freq2_to_freq1_zero`, `freq0_to_freq1_zero` | Frequency mismatch forces a zero linear map, so frequency-2 content needs the n̄·c₂ coupling. |
| | `phase_coupling` | `z₂ · z̄₁` carries frequency 1. |
| `Camera.lean` | `project_roll`, `roll_posed`, `roll_fixes_axis_translation` | A roll about the optical axis rotates the image about the principal point and acts on the label by `R ↦ R_z R`. |
| | `roll_moves_offaxis_translation` | Off-axis crops break the symmetry, so the warp's centring is required. |

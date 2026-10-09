import Mathlib

/-!
# Cl(3,0) restricted to rotations about the optical axis e₃

Multivectors are 8-tuples in the repository's blade order `1, e1, e2, e3, e12, e13, e23, e123`
(`pose3d` / `clifford` package order). `gp` is the geometric product of Cl(3,0) written out
component by component (generated from the blade algebra, checked against `clifford`), `rev` the
reversion, and `rotZ c s = c - s e12` the rotor of a roll by θ with `c = cos (θ/2)`,
`s = sin (θ/2)`.

`sandwich_rotZ` computes `R x R̃` for every component: the scalar, e3, e12 and e123 parts are
invariant (SO(2) frequency 0) and the pairs (e1, e2) and (e13, e23) rotate by θ (frequency 1).
So `Res^{SO(3)}_{SO(2)} Cl(3,0) = 4·ρ₀ ⊕ 2·ρ₁`, which is exactly the token layout of the
SO(2) condition head; no component carries frequency ≥ 2 (see `Harmonics.lean` for why such
features must then be coupled down before they can enter a token).
-/

namespace So2.Cl3

/-- A Cl(3,0) multivector, components in the order 1, e1, e2, e3, e12, e13, e23, e123. -/
abbrev MV := Fin 8 → ℝ

/-- The geometric product of Cl(3,0). -/
def gp (a b : MV) : MV := ![
  a 0*b 0 + a 1*b 1 + a 2*b 2 + a 3*b 3 - a 4*b 4 - a 5*b 5 - a 6*b 6 - a 7*b 7,
  a 0*b 1 + a 1*b 0 - a 2*b 4 - a 3*b 5 + a 4*b 2 + a 5*b 3 - a 6*b 7 - a 7*b 6,
  a 0*b 2 + a 1*b 4 + a 2*b 0 - a 3*b 6 - a 4*b 1 + a 5*b 7 + a 6*b 3 + a 7*b 5,
  a 0*b 3 + a 1*b 5 + a 2*b 6 + a 3*b 0 - a 4*b 7 - a 5*b 1 - a 6*b 2 - a 7*b 4,
  a 0*b 4 + a 1*b 2 - a 2*b 1 + a 3*b 7 + a 4*b 0 - a 5*b 6 + a 6*b 5 + a 7*b 3,
  a 0*b 5 + a 1*b 3 - a 2*b 7 - a 3*b 1 + a 4*b 6 + a 5*b 0 - a 6*b 4 - a 7*b 2,
  a 0*b 6 + a 1*b 7 + a 2*b 3 - a 3*b 2 - a 4*b 5 + a 5*b 4 + a 6*b 0 + a 7*b 1,
  a 0*b 7 + a 1*b 6 - a 2*b 5 + a 3*b 4 + a 4*b 3 - a 5*b 2 + a 6*b 1 + a 7*b 0]

/-- Reversion: grades 2 and 3 change sign. -/
def rev (a : MV) : MV := ![a 0, a 1, a 2, a 3, -a 4, -a 5, -a 6, -a 7]

/-- The rotor `c - s e12` (a roll about e3). -/
def rotZ (c s : ℝ) : MV := ![c, 0, 0, 0, -s, 0, 0, 0]

/-- The sandwich `R x R̃`, the action of the rotor on a multivector (GATr's and CGENN's action). -/
def sandwich (R x : MV) : MV := gp (gp R x) (rev R)

/-- `rotZ c s` is a unit rotor when `c² + s² = 1`: `R R̃ = 1`. -/
theorem rotZ_unit (c s : ℝ) (h : c ^ 2 + s ^ 2 = 1) : gp (rotZ c s) (rev (rotZ c s)) = ![1, 0, 0, 0, 0, 0, 0, 0] := by
  funext i
  fin_cases i <;> simp [gp, rev, rotZ] <;> first | ring1 | linear_combination h

/-- Component formulas of the roll, with `C = c² - s² = cos θ` and `S = 2 c s = sin θ`. -/
theorem sandwich_rotZ (c s : ℝ) (h : c ^ 2 + s ^ 2 = 1) (x : MV) :
    sandwich (rotZ c s) x = ![x 0,
      (c ^ 2 - s ^ 2) * x 1 - (2 * c * s) * x 2,
      (2 * c * s) * x 1 + (c ^ 2 - s ^ 2) * x 2,
      x 3,
      x 4,
      (c ^ 2 - s ^ 2) * x 5 - (2 * c * s) * x 6,
      (2 * c * s) * x 5 + (c ^ 2 - s ^ 2) * x 6,
      x 7] := by
  funext i
  fin_cases i <;> simp [sandwich, gp, rev, rotZ]
  · linear_combination x 0 * h
  · ring
  · ring
  · linear_combination x 3 * h
  · linear_combination x 4 * h
  · ring
  · ring
  · linear_combination x 7 * h

/-- The same with angles: `R = cos (θ/2) - sin (θ/2) e12` rotates (e1, e2) and (e13, e23) by θ
and fixes 1, e3, e12, e123. -/
theorem sandwich_roll (θ : ℝ) (x : MV) :
    sandwich (rotZ (Real.cos (θ / 2)) (Real.sin (θ / 2))) x = ![x 0,
      Real.cos θ * x 1 - Real.sin θ * x 2,
      Real.sin θ * x 1 + Real.cos θ * x 2,
      x 3,
      x 4,
      Real.cos θ * x 5 - Real.sin θ * x 6,
      Real.sin θ * x 5 + Real.cos θ * x 6,
      x 7] := by
  have hc : Real.cos θ = Real.cos (θ / 2) ^ 2 - Real.sin (θ / 2) ^ 2 := by
    have := Real.cos_two_mul (θ / 2)
    rw [show 2 * (θ / 2) = θ by ring] at this
    linear_combination this + Real.sin_sq_add_cos_sq (θ / 2)
  have hs : Real.sin θ = 2 * Real.cos (θ / 2) * Real.sin (θ / 2) := by
    have := Real.sin_two_mul (θ / 2)
    rw [show 2 * (θ / 2) = θ by ring] at this
    rw [this]; ring
  rw [sandwich_rotZ _ _ (Real.cos_sq_add_sin_sq (θ / 2)), hc, hs]

/-- Rolls compose: `rotZ(θ₁) rotZ(θ₂) = rotZ(θ₁ + θ₂)` (the half-angle rotors form the double
cover of SO(2) inside Spin(3)). -/
theorem rotZ_mul (a b : ℝ) :
    gp (rotZ (Real.cos (a / 2)) (Real.sin (a / 2))) (rotZ (Real.cos (b / 2)) (Real.sin (b / 2)))
      = rotZ (Real.cos ((a + b) / 2)) (Real.sin ((a + b) / 2)) := by
  funext i
  have h1 : (a + b) / 2 = a / 2 + b / 2 := by ring
  fin_cases i <;> simp [gp, rotZ, h1, Real.cos_add, Real.sin_add] 

end So2.Cl3

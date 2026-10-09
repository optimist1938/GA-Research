import Mathlib

/-!
# SO(2) / C₄ harmonics: what can enter a Cl(3,0) token, and how

* `rot_commutant`: a real 2×2 map that commutes with every rotation is `a·I + b·J` (a complex
  scalar). So the most general equivariant linear map between two frequency-1 fields is a learned
  gain + phase, which is what the head's complex weights implement.
* `freq_mismatch_zero`: an equivariant linear map from a frequency-`m` field to a frequency-`k`
  field vanishes when the rotation by `π` already separates them (e.g. `m = 2`, `k = 1`:
  `e^{2iπ} = 1` but `e^{iπ} = -1`). Cl(3,0) tokens only carry `k ∈ {0, 1}` (`Cl3.sandwich_roll`),
  so frequency-2 image content cannot be written into a token linearly.
* `phase_coupling`: `z₂ · conj z₁` has frequency 1 when `z₂`, `z₁` have frequencies 2, 1: the
  (bispectrum-style) product that carries frequency-2 content into a vector token.
* `lift_equivariant`: lifting by input rotation, `F x k = k • B (k⁻¹ • x)`, turns any map `B`
  into a field on the group that transforms by the regular representation.
* `dft_shift`: the discrete Fourier coefficient `∑ₖ ω^{-mk} F (k)` of a cyclically shifted
  sequence picks up the phase `ω^{-m}`, i.e. the coefficient is a frequency-`m` field.
-/

namespace So2.Harmonics

open Complex

/-- Rotation of the plane, as multiplication by `e^{iθ}` on `ℂ`. -/
noncomputable def rot (θ : ℝ) (z : ℂ) : ℂ := exp (θ * I) * z

/-- A real-linear map of the plane that commutes with all rotations is multiplication by a
complex number. (Schur's lemma for the frequency-1 representation of SO(2) over ℝ.) -/
theorem rot_commutant (L : ℂ →ₗ[ℝ] ℂ) (hL : ∀ θ : ℝ, ∀ z, L (rot θ z) = rot θ (L z)) :
    ∀ z, L z = L 1 * z := by
  intro z
  have hI : L I = I * L 1 := by
    have h := hL (Real.pi / 2) 1
    simp only [rot, mul_one] at h
    have e : exp (((Real.pi / 2 : ℝ) : ℂ) * I) = I := by
      rw [show ((Real.pi / 2 : ℝ) : ℂ) * I = ((Real.pi / 2 : ℝ) : ℂ) * I from rfl]
      rw [exp_mul_I]
      simp
    rw [e] at h
    exact h
  have hz : z = (z.re : ℝ) • (1 : ℂ) + (z.im : ℝ) • I := by
    apply Complex.ext <;> simp
  rw [hz, map_add, map_smul, map_smul, hI]
  simp only [Complex.real_smul]
  ring

/-- Frequency mismatch: if `L (e^{imθ} z) = e^{ikθ} L z` for all `θ`, and the half turn acts by
`+1` on the source but `-1` on the target, then `L = 0`. -/
theorem freq_mismatch_zero (L : ℂ →ₗ[ℝ] ℂ) (m k : ℤ)
    (hL : ∀ θ : ℝ, ∀ z, L (exp ((m * θ : ℝ) * I) * z) = exp ((k * θ : ℝ) * I) * L z)
    (hm : exp ((m * Real.pi : ℝ) * I) = 1) (hk : exp ((k * Real.pi : ℝ) * I) = -1) :
    ∀ z, L z = 0 := by
  intro z
  have h := hL Real.pi z
  rw [hm, one_mul, hk] at h
  have : (2 : ℂ) * L z = 0 := by linear_combination h
  simpa using this

/-- The concrete case used by the head: frequency 2 cannot feed a frequency-1 slot linearly. -/
theorem freq2_to_freq1_zero (L : ℂ →ₗ[ℝ] ℂ)
    (hL : ∀ θ : ℝ, ∀ z, L (exp ((2 * θ : ℝ) * I) * z) = exp ((1 * θ : ℝ) * I) * L z) :
    ∀ z, L z = 0 := by
  refine freq_mismatch_zero L 2 1 (by exact_mod_cast hL) ?_ ?_
  · push_cast
    exact Complex.exp_two_pi_mul_I
  · push_cast
    simp [exp_pi_mul_I]

/-- Frequency-1 slots cannot be filled from frequency-0 (invariant) content linearly either. -/
theorem freq0_to_freq1_zero (L : ℂ →ₗ[ℝ] ℂ)
    (hL : ∀ θ : ℝ, ∀ z, L z = exp ((1 * θ : ℝ) * I) * L z) : ∀ z, L z = 0 := by
  intro z
  have h := hL Real.pi z
  have hk : exp (((1 : ℝ) * Real.pi : ℝ) * I) = -1 := by push_cast; simp [exp_pi_mul_I]
  rw [hk] at h
  have : (2 : ℂ) * L z = 0 := by linear_combination h
  simpa using this

/-- Phase coupling: frequencies add under products and flip under conjugation, so
`z₂ · conj z₁` is a frequency-1 quantity when `z₂`, `z₁` have frequencies 2 and 1. -/
theorem phase_coupling (θ : ℝ) (z₁ z₂ : ℂ) :
    (exp ((2 * θ : ℝ) * I) * z₂) * (starRingEnd ℂ) (exp ((1 * θ : ℝ) * I) * z₁)
      = exp ((1 * θ : ℝ) * I) * (z₂ * (starRingEnd ℂ) z₁) := by
  have hconj : (starRingEnd ℂ) (exp ((1 * θ : ℝ) * I)) = exp (-((1 * θ : ℝ) * I)) := by
    rw [← Complex.exp_conj]; simp [Complex.conj_ofReal]
  rw [map_mul, hconj]
  have : exp ((2 * θ : ℝ) * I) = exp ((1 * θ : ℝ) * I) * exp ((1 * θ : ℝ) * I) := by
    rw [← Complex.exp_add]; push_cast; ring_nf
  rw [this]
  have hinv : exp ((1 * θ : ℝ) * I) * exp (-((1 * θ : ℝ) * I)) = 1 := by
    rw [← Complex.exp_add]; simp
  linear_combination (exp ((1 * θ : ℝ) * I) * z₂ * (starRingEnd ℂ) z₁) * hinv

section Lift

variable {G X Y : Type*} [Group G] [MulAction G X] [MulAction G Y]

/-- Lifting by input rotation: run the same (non-equivariant) backbone `B` on every rotated copy
`k⁻¹ • x` and rotate its output back. -/
def lift (B : X → Y) (x : X) (k : G) : Y := k • B (k⁻¹ • x)

/-- The lifted features transform by the regular representation: rotating the input rotates every
output and shifts the group index, `F (g • x) k = g • F x (g⁻¹ k)`. -/
theorem lift_equivariant (B : X → Y) (g : G) (x : X) (k : G) :
    lift B (g • x) k = g • lift B x (g⁻¹ * k) := by
  simp only [lift]
  rw [mul_inv_rev, inv_inv, ← smul_smul, mul_smul k⁻¹ g, smul_smul g g⁻¹, mul_inv_cancel, one_smul]

end Lift

section DFT

/-- Cyclic-shift theorem on `ZMod n` for an `n`-th root of unity: shifting the sequence by one
step multiplies the `m`-th coefficient by `ω^m`. Stated with the coefficient
`c m F = ∑ k, ω^(m * k.val) * F k` and the shift `F (k - 1)`. -/
theorem dft_shift (n : ℕ) [NeZero n] (ω : ℂ) (hω : ω ^ n = 1) (m : ℕ) (F : ZMod n → ℂ) :
    ∑ k : ZMod n, ω ^ (m * k.val) * F (k - 1) = ω ^ m * ∑ k : ZMod n, ω ^ (m * k.val) * F k := by
  rw [Finset.mul_sum]
  refine Fintype.sum_equiv (Equiv.subRight (1 : ZMod n)) _ _ ?_
  intro k
  simp only [Equiv.subRight_apply]
  rw [← mul_assoc, ← pow_add]
  congr 1
  -- ω^(m * k.val) = ω^(m * ((k-1).val + 1)) because k.val ≡ (k-1).val + 1 (mod n) and ω^n = 1
  have red : ∀ c : ℕ, ω ^ (m * c) = ω ^ (m * (c % n)) := by
    intro c
    conv_lhs => rw [← Nat.mod_add_div c n]
    rw [mul_add, pow_add, show m * (n * (c / n)) = n * (m * (c / n)) by ring, pow_mul ω n, hω,
      one_pow, mul_one]
  have key : ∀ a b : ℕ, a % n = b % n → ω ^ (m * a) = ω ^ (m * b) := by
    intro a b hab
    rw [red a, red b, hab]
  rw [show m + m * (k - 1).val = m * ((k - 1).val + 1) by ring]
  apply key
  have hk : k = (k - 1) + 1 := by ring
  conv_lhs => rw [hk]
  rw [ZMod.val_add, ZMod.val_one_eq_one_mod, Nat.mod_mod, Nat.add_mod_mod]

end DFT

end So2.Harmonics

import Mathlib

/-!
# Group-theoretic equivariance lemmas for the SO(2)-equivariant pose pipeline

Every statement is about an abstract group `G` (in the model: the in-plane camera rolls, either the
exact pixel-grid rotations `C₄` or `SO(2)` itself) acting on images `X`, on condition tokens `V`,
and on poses by left multiplication through a hom into the rotor monoid `M`.

* `reynolds_equivariant`: averaging any map over a finite group gives an equivariant map
  (variant A: the C₄-lifted backbone).
* `canonicalize_equivariant`: an equivariant canonicaliser turns any map into an equivariant one
  (variant B: learned in-plane canonicalisation).
* `euler_equivariant`: Euler integration on the rotor group with an invariant body velocity
  carries `φ g * r₀` to `φ g * r_N` (the flow sampler is exactly equivariant, step by step).
* `body_velocity_invariant`: `r̃ w r` is unchanged when `r ↦ g r`, `w ↦ g w g̃`, `g̃ g = 1`.
* `medoid_cost_invariant`: the medoid's selection cost is unchanged by a left translation.
* `fixed_input_forces_symmetric_output`: the price of exact equivariance (an image fixed by a roll
  must get an output fixed by it).
-/

namespace So2

open Finset

section Reynolds

variable {G X V : Type*} [Group G] [Fintype G] [MulAction G X] [AddCommMonoid V]
  [DistribMulAction G V]

/-- Group averaging (Reynolds operator, without the `1/|G|`): `Φ x = ∑_g g • ψ (g⁻¹ • x)`. -/
def reynolds (ψ : X → V) (x : X) : V := ∑ g : G, g • ψ (g⁻¹ • x)

/-- The group average of an arbitrary map is equivariant. -/
theorem reynolds_equivariant (ψ : X → V) (h : G) (x : X) :
    reynolds (G := G) ψ (h • x) = h • reynolds (G := G) ψ x := by
  unfold reynolds
  rw [Finset.smul_sum]
  refine (Fintype.sum_equiv (Equiv.mulLeft h) (fun k => h • (k • ψ (k⁻¹ • x)))
    (fun g => g • ψ (g⁻¹ • (h • x))) ?_).symm
  intro k
  simp only [Equiv.coe_mulLeft]
  rw [mul_inv_rev, mul_smul k⁻¹ h⁻¹, inv_smul_smul, mul_smul]

/-- With scalars that commute with the action, the normalised average `|G|⁻¹ • Φ` is equivariant
too (the form used in the network, a mean over the 4 rotated passes). -/
theorem reynolds_mean_equivariant {K : Type*} [Semiring K] [Module K V] [SMulCommClass K G V]
    (c : K) (ψ : X → V) (h : G) (x : X) :
    c • reynolds (G := G) ψ (h • x) = h • (c • reynolds (G := G) ψ x) := by
  rw [reynolds_equivariant, smul_comm]

end Reynolds

section Canonicalize

variable {G X Y : Type*} [Group G] [MulAction G X] [MulAction G Y]

/-- Canonicalisation: undo the predicted roll `c x`, run any map `f₀`, re-apply the roll. -/
def canonicalize (c : X → G) (f₀ : X → Y) (x : X) : Y := c x • f₀ ((c x)⁻¹ • x)

/-- If the canonicaliser is equivariant (`c (g • x) = g * c x`), the canonicalised map is
equivariant, whatever `f₀` is (e.g. the pretrained non-equivariant ResNet + heads). -/
theorem canonicalize_equivariant (c : X → G) (hc : ∀ (g : G) (x : X), c (g • x) = g * c x) (f₀ : X → Y)
    (g : G) (x : X) : canonicalize c f₀ (g • x) = g • canonicalize c f₀ x := by
  simp only [canonicalize, hc]
  rw [mul_inv_rev, mul_smul (c x)⁻¹ g⁻¹, inv_smul_smul, mul_smul]

/-- On inputs whose canonical roll is the identity (upright photos, if the canonicaliser learns
that), the canonicalised map *is* `f₀`: canonicalisation does not perturb the pretrained model
where the data already is canonical. -/
theorem canonicalize_of_canonical (c : X → G) (f₀ : X → Y) (x : X) (hx : c x = 1) :
    canonicalize c f₀ x = f₀ x := by
  simp [canonicalize, hx]

end Canonicalize

section Euler

variable {G M X : Type*} [Group G] [Monoid M] [MulAction G X]

/-- Euler steps on the rotor group: `r_{n+1} = r_n * inc r_n n x`, where `inc r n x` stands for
`exp (dt • v (r, t_n, x))` (body velocity, right multiplication as in `CliffordFlow._integrate`). -/
def euler (inc : M → ℕ → X → M) (x : X) (r₀ : M) : ℕ → M
  | 0 => r₀
  | n + 1 => euler inc x r₀ n * inc (euler inc x r₀ n) n x

/-- If the increment is invariant under `r ↦ φ g * r`, `x ↦ g • x` (an equivariant body-velocity
field), the sampler commutes with the roll at every step: rolled image + rolled noise give the
rolled trajectory. -/
theorem euler_equivariant (φ : G →* M) (inc : M → ℕ → X → M)
    (hinv : ∀ (g : G) (r : M) (n : ℕ) (x : X), inc (φ g * r) n (g • x) = inc r n x) (g : G) (x : X) (r₀ : M) :
    ∀ n, euler inc (g • x) (φ g * r₀) n = φ g * euler inc x r₀ n
  | 0 => rfl
  | n + 1 => by
    simp only [euler]
    rw [euler_equivariant φ inc hinv g x r₀ n, hinv, mul_assoc]

end Euler

section BodyVelocity

/-- The frame read-out of the GATr field: a spatial velocity `w` that transforms like a vector
(`w ↦ g w g̃`) and a pose `r ↦ g r` give the body velocity `r̃ w r`, which is then invariant. -/
theorem body_velocity_invariant {R : Type*} [Monoid R] [StarMul R] (g r w : R)
    (hg : star g * g = 1) :
    star (g * r) * (g * w * star g) * (g * r) = star r * w * r := by
  simp only [star_mul, mul_assoc]
  rw [← mul_assoc (star g) g, hg, one_mul, ← mul_assoc (star g) g, hg, one_mul]

end BodyVelocity

section Medoid

variable {G P : Type*} [Group G] [MulAction G P]

/-- The medoid picks the sample minimising `∑_j d(s_i, s_j)`. With a left-invariant distance the
cost of every candidate is unchanged by translating all samples, so the same index wins and the
medoid of the translated samples is the translated medoid. -/
theorem medoid_cost_invariant {n : ℕ} (d : P → P → ℝ) (hd : ∀ (g : G) (a b : P), d (g • a) (g • b) = d a b)
    (g : G) (s : Fin n → P) (i : Fin n) :
    ∑ j, d (g • s i) (g • s j) = ∑ j, d (s i) (s j) := by
  simp [hd]

end Medoid

section Price

variable {G X Y : Type*} [Group G] [MulAction G X] [MulAction G Y]

/-- The cost of exact equivariance: an input fixed by a roll (`g • x = x`, e.g. a rotationally
symmetric crop) must be mapped to an output fixed by the same roll. For the pose distribution this
means it is symmetric about the optical axis, so an upright-camera prior cannot be used on it. -/
theorem fixed_input_forces_symmetric_output (f : X → Y) (hf : ∀ (g : G) (x : X), f (g • x) = g • f x)
    (g : G) (x : X) (hx : g • x = x) : g • f x = f x := by
  rw [← hf, hx]

end Price

section OrbitArgmax

variable {G X : Type*} [Group G] [MulAction G X]

/-- The C₄ canonicaliser of variant B: score every turn of the image, `k ↦ s (k⁻¹ • x)`, and pick
the best. If `c` is the strict maximiser for `x`, then `g * c` is the strict maximiser for `g • x`,
so `c (g • x) = g * c x`: the argmax-over-the-orbit canonicaliser is equivariant for any scorer `s`
(away from ties). -/
theorem orbit_argmax_equivariant (s : X → ℝ) (x : X) (c : G)
    (hc : ∀ k, k ≠ c → s (k⁻¹ • x) < s (c⁻¹ • x)) (g : G) :
    ∀ k, k ≠ g * c → s (k⁻¹ • (g • x)) < s ((g * c)⁻¹ • (g • x)) := by
  intro k hk
  have hk' : g⁻¹ * k ≠ c := by
    intro h; apply hk; rw [← h, mul_inv_cancel_left]
  have e1 : k⁻¹ • (g • x) = (g⁻¹ * k)⁻¹ • x := by
    rw [mul_inv_rev, inv_inv, mul_smul]
  have e2 : (g * c)⁻¹ • (g • x) = c⁻¹ • x := by
    rw [mul_inv_rev, mul_smul, inv_smul_smul]
  rw [e1, e2]
  exact hc _ hk'

end OrbitArgmax

section ErrorInvariance

variable {G X P : Type*} [Group G] [MulAction G X] [MulAction G P]

/-- The robustness guarantee: for an equivariant predictor and a left-invariant error metric
(the geodesic angle on SO(3) is bi-invariant), the error on a turned test pair `(g • x, g • R)` is
the error on `(x, R)`. So the whole error distribution on a turned test set equals the upright one,
sample by sample. -/
theorem equivariant_error_invariant (f : X → P) (hf : ∀ (g : G) (x : X), f (g • x) = g • f x)
    (d : P → P → ℝ) (hd : ∀ (g : G) (a b : P), d (g • a) (g • b) = d a b) (g : G) (x : X) (R : P) :
    d (f (g • x)) (g • R) = d (f x) R := by
  rw [hf, hd]

end ErrorInvariance

end So2

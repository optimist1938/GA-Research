import Mathlib

/-!
# Why the in-plane roll is an exact symmetry of the warped crops

A pinhole camera with focal length `f` and principal point `p` projects a camera-frame point
`X = (X₁, X₂, X₃)`, `X₃ ≠ 0`, to `π X = (f X₁ / X₃ + p₁, f X₂ / X₃ + p₂)`.

* `project_roll`: rolling the scene about the optical axis by θ rotates the image about the
  principal point by the same θ (in the image axes that match the camera axes `e1`, `e2`).
* `roll_pose`: the roll is linear, so the rolled posed object `roll (R P + t)` is the object posed
  by `(R_z R, R_z t)`. The labels are rotations only, so the label change is `R ↦ R_z R` for **any**
  object translation; the translation rotating too is invisible to the label.
* What the symmetry does need is that the image is turned about the **principal point** `p`
  (`project_roll`). The Image2Sphere warp puts the principal point at the crop centre (112, 112) in
  skimage pixel-centre coordinates, while `torch.rot90` on a 224 grid turns about 111.5: a
  half-pixel offset, so the pixel-level symmetry is exact up to that shift.
* `roll_fixes_axis_translation`, `roll_moves_offaxis_translation`: the translation itself is fixed by
  a roll iff it lies on the optical axis (only relevant for full 6-DoF labels, not for ours).
-/

namespace So2.Camera

/-- Roll about the optical axis e₃ by `θ`, acting on camera-frame points. -/
noncomputable def roll (θ : ℝ) (X : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![Real.cos θ * X 0 - Real.sin θ * X 1, Real.sin θ * X 0 + Real.cos θ * X 1, X 2]

/-- Rotation of the image plane about the principal point `p` by `θ`. -/
noncomputable def rot2 (θ : ℝ) (p u : Fin 2 → ℝ) : Fin 2 → ℝ :=
  ![Real.cos θ * (u 0 - p 0) - Real.sin θ * (u 1 - p 1) + p 0,
    Real.sin θ * (u 0 - p 0) + Real.cos θ * (u 1 - p 1) + p 1]

/-- Pinhole projection. -/
noncomputable def project (f : ℝ) (p : Fin 2 → ℝ) (X : Fin 3 → ℝ) : Fin 2 → ℝ :=
  ![f * X 0 / X 2 + p 0, f * X 1 / X 2 + p 1]

/-- Projection commutes with the roll: `π (R_z X) = rot_p (π X)`. -/
theorem project_roll (f θ : ℝ) (p : Fin 2 → ℝ) (X : Fin 3 → ℝ) (hX : X 2 ≠ 0) :
    project f p (roll θ X) = rot2 θ p (project f p X) := by
  funext i
  fin_cases i <;> simp [project, roll, rot2] <;> field_simp <;> ring

/-- The roll is linear: rolling the posed point `Y + t` gives `roll Y + roll t`, i.e. the object
posed by `(R_z R, R_z t)`. -/
theorem roll_pose (θ : ℝ) (Y t : Fin 3 → ℝ) : roll θ (Y + t) = roll θ Y + roll θ t := by
  funext i
  fin_cases i <;> simp [roll] <;> ring

/-- A translation along the optical axis is fixed by the roll. -/
theorem roll_fixes_axis_translation (θ d : ℝ) : roll θ ![0, 0, d] = ![0, 0, d] := by
  funext i
  fin_cases i <;> simp [roll]

/-- The roll is linear, so with an on-axis translation it moves the posed point `Y + t` to
`roll Y + t`: the rolled image is the image of the object posed by `R_z R`. -/
theorem roll_posed (θ d : ℝ) (Y : Fin 3 → ℝ) :
    roll θ (Y + ![0, 0, d]) = roll θ Y + ![0, 0, d] := by
  funext i
  fin_cases i <;> simp [roll] <;> ring

/-- Off the optical axis the roll is not a symmetry of the translation: if `(t₁, t₂) ≠ 0` and the
roll fixes `t`, then `θ` is a whole turn (`cos θ = 1`). -/
theorem roll_moves_offaxis_translation (θ : ℝ) (t : Fin 3 → ℝ) (ht : t 0 ≠ 0 ∨ t 1 ≠ 0)
    (hfix : roll θ t = t) : Real.cos θ = 1 := by
  have h0 := congrFun hfix 0
  have h1 := congrFun hfix 1
  simp [roll] at h0 h1
  have hsq := Real.sin_sq_add_cos_sq θ
  -- (cos θ - 1) t₀ = sin θ t₁ and sin θ t₀ = (1 - cos θ) t₁ give (1 - cos θ)(t₀² + t₁²) = 0
  have key : (1 - Real.cos θ) * (t 0 ^ 2 + t 1 ^ 2) = 0 := by
    nlinarith [h0, h1, hsq, sq_nonneg (t 0), sq_nonneg (t 1)]
  have hpos : t 0 ^ 2 + t 1 ^ 2 ≠ 0 := by
    rcases ht with h | h
    · have := pow_pos (abs_pos.mpr h) 2; rw [sq_abs] at this; positivity
    · have := pow_pos (abs_pos.mpr h) 2; rw [sq_abs] at this; positivity
  have := (mul_eq_zero.mp key).resolve_right hpos
  linarith

end So2.Camera

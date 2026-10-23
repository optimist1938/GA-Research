"""Is the learned flow even in the rotor, and where do the trajectories go: to q or to -q?

The flow-matching target is exactly even, B(-r, r1) = B(r, r1) (log_map takes the w >= 0 representative), but nothing forces the network to be:
its input is the rotor multivector, which changes sign under r -> -r. This measures, on a trained CliffordFlow:

  evenness   |v(r, t) - v(-r, t)| / |v(r, t)| (and the absolute difference) at the states the flow actually visits, binned by the rotation
             error of that state, restricted to trajectories that end within `--near` degrees of the truth;
  mirror     integrate from -r0 instead of r0 (same image, same noise up to sign): an even field gives exactly the mirrored trajectory, so the final
             rotations agree and the final rotors are negatives; the angle between the two final rotations is reported;
  split      how many trajectories end at +q_gt (w >= 0 representative of the truth) and how many at -q_gt, and how much of that the sign of the
             starting rotor (q_gt . r0) explains.

    python -m pose3d.analyze_rotor_symmetry --checkpoint a.pth --val_cache pascal_val.pt --n_images 100 --k 32 --out sym.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pose3d.engine.rotor_viz import gt_rotor
from pose3d.geometry.flow import exp_map, rotor_multiply
from pose3d.geometry.rotor import random_rotor

BINS = (0, 5, 15, 30, 60, 90, 181)          # rotation error of the visited state (deg): near-correct first


def rot_err(rotors, q):
    """Rotation angle (deg) between rotors (..., 4) and the rotation of q: twice the S^3 angle to the nearer of +-q."""
    return np.degrees(2 * np.arccos(np.clip(np.abs(rotors @ q), 0, 1)))


@torch.no_grad()
def integrate(model, r0, cond, steps, keep_velocity=False):
    """Euler flow from r0; returns the rotors (steps + 1, k, 4) and, if asked, v(r_t, t) and v(-r_t, t) at every step (steps, k, 3)."""
    dev = r0.device
    k = r0.shape[0]
    rotor, path, v_pos, v_neg = r0, [r0], [], []
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((k,), i * dt, device=dev)
        v = model.velocity(rotor, t, cond)
        if keep_velocity:
            v_pos.append(v); v_neg.append(model.velocity(-rotor, t, cond))
        rotor = rotor_multiply(rotor, exp_map(dt * v), model.algebra)
        path.append(rotor)
    out = torch.stack(path).cpu().numpy().astype(np.float64)
    return (out, torch.stack(v_pos).cpu().numpy(), torch.stack(v_neg).cpu().numpy()) if keep_velocity else out


@torch.no_grad()
def analyze(model, imgs, rots, idx, k=32, steps=20, seed=0, near=15.0, log=print):
    dev = next(model.parameters()).device
    rows = []                                                   # one row per (image, sample, step) of an evenness measurement
    mirror, split = [], []
    for n, i in enumerate(idx):
        q = gt_rotor(rots[i])
        cond = model.condition(imgs[i][None].to(dev)).repeat(k, 1, 1)
        g = torch.Generator().manual_seed(seed + int(i))
        r0 = torch.nn.functional.normalize(torch.randn(k, 4, generator=g), dim=-1).to(dev)
        P, vp, vn = integrate(model, r0, cond, steps, keep_velocity=True)
        Q = integrate(model, -r0, cond, steps)
        end_err = rot_err(P[-1], q)
        ok = end_err < near
        sign_end = np.sign(P[-1] @ q); sign_start = np.sign(P[0] @ q)
        for j in range(k):
            split.append((int(i), j, float(sign_start[j]), float(sign_end[j]), float(end_err[j]), float(np.sign(P[-1, j, 0]))))
            ang = float(np.degrees(2 * np.arccos(np.clip(abs(P[-1, j] @ Q[-1, j]), 0, 1))))   # rotation difference of the two final states
            mirror.append((int(i), j, bool(ok[j]), float(end_err[j]), float(rot_err(Q[-1, j], q)), ang, float(np.sign(P[-1, j] @ Q[-1, j]))))
        dn = np.linalg.norm(vp - vn, axis=-1)                    # (steps, k)
        nv = np.linalg.norm(vp, axis=-1)
        state_err = np.stack([rot_err(P[s], q) for s in range(steps)])          # error of the state the velocity was evaluated at
        for s in range(steps):
            for j in range(k):
                rows.append((s / steps, float(state_err[s, j]), float(dn[s, j] / max(nv[s, j], 1e-9)), float(dn[s, j]), float(nv[s, j]), bool(ok[j])))
        if (n + 1) % 10 == 0:
            log(f"{n + 1}/{len(idx)} images")
    return np.array(rows, dtype=np.float64), mirror, split


def summarize(rows, mirror, split, near=15.0):
    t, err, rel, absd, nv, ok = rows.T
    ok = ok.astype(bool)
    out = {"n_trajectories": len(split), "near_deg": near, "n_near_correct": int(sum(m[2] for m in mirror))}
    # 1) evenness, near-correct trajectories only, by the error of the visited state
    bins = {}
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = ok & (err >= lo) & (err < hi)
        if m.sum():
            bins[f"{lo}-{hi}"] = dict(n=int(m.sum()), rel_mean=float(rel[m].mean()), rel_median=float(np.median(rel[m])), abs_mean=float(absd[m].mean()), speed_mean=float(nv[m].mean()))
    out["evenness_by_state_error_deg_near_correct_trajectories"] = bins
    out["evenness_by_time_near_correct_trajectories"] = {f"{x:.2f}": float(np.median(rel[ok & np.isclose(t, x)])) for x in np.unique(t)[::4]}
    out["evenness_all_trajectories_median"] = float(np.median(rel)); out["evenness_near_correct_median"] = float(np.median(rel[ok])) if ok.any() else None
    # 2) mirror consistency
    mi = np.array([(m[2], m[3], m[4], m[5], m[6]) for m in mirror], dtype=np.float64)
    mok = mi[:, 0].astype(bool)
    out["mirror"] = dict(n=int(mok.sum()))
    if mok.any():
        out["mirror"].update(
            final_rotation_diff_deg_median=float(np.median(mi[mok, 3])), final_rotation_diff_deg_p90=float(np.percentile(mi[mok, 3], 90)),
            minus_r0_also_near_correct=float((mi[mok, 2] < near).mean()),
            final_rotors_are_negatives=float((mi[mok, 4] < 0).mean()))
    # 3) where trajectories go
    sp = np.array([(s[2], s[3], s[4], s[5]) for s in split])
    ends_ok = sp[:, 2] < near
    plus = ends_ok & (sp[:, 1] > 0); minus = ends_ok & (sp[:, 1] < 0)
    out["split"] = dict(to_plus_q=int(plus.sum()), to_minus_q=int(minus.sum()), miss=int((~ends_ok).sum()),
                        frac_plus_among_near_correct=float(plus.sum() / max(ends_ok.sum(), 1)),
                        start_sign_predicts_destination=float((sp[ends_ok, 0] == sp[ends_ok, 1]).mean()),
                        frac_final_w_positive_among_near_correct=float((sp[ends_ok, 3] > 0).mean()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True); p.add_argument("--val_cache", required=True)
    p.add_argument("--n_images", type=int, default=100); p.add_argument("--k", type=int, default=32); p.add_argument("--steps", type=int, default=20)
    p.add_argument("--near", type=float, default=15.0); p.add_argument("--seed", type=int, default=0); p.add_argument("--out", default="rotor_symmetry.json")
    a = p.parse_args()
    from pose3d.visualize_flow import load
    torch.set_num_threads(4)
    blob = torch.load(a.val_cache, map_location="cpu", weights_only=True)
    imgs, rots = blob["imgs"].float() / 255, blob["targets"].float().numpy()
    idx = np.unique(np.linspace(0, len(imgs) - 1, a.n_images).astype(int))
    rows, mirror, split = analyze(load(a.checkpoint), imgs, rots, idx, a.k, a.steps, a.seed, a.near, log=lambda s: print(s, flush=True))
    res = summarize(rows, mirror, split, a.near)
    Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

"""How multimodal is the flow's distribution over poses? Plus an interactive (rotatable) version of the trajectory plot.

For N validation images each model's flow is run from K random rotations. The K final rotations are grouped into modes (greedy clusters
of radius --mode_deg around the most crowded rotation); then, per image: the mass of the dominant mode, how many modes carry at least
--min_mass of the samples, which mode is closest to the truth, and how widely the samples are spread. Writes
  modes_analysis.png   mass of the dominant mode, share of multi-modal images per class, spread vs error, angle between the two top modes
  bimodal_gallery.png  images where the flow splits, with the competing poses and the truth drawn as coordinate frames
  interactive_paths.html  the paths-in-the-rotation-ball plot, rotatable, one image / model per dropdown entry

    python -m pose3d.analyze_flow_modes --checkpoint a.pth --label GATr --checkpoint b.pth --label MLP --val_cache pascal_val.pt --out viz/

Class names: the Pascal3D+ validation set is ordered by category, so a class is known from the per-class image counts of the test set.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from pose3d.engine.flow_viz import CLASS_NAMES, angle_deg, relative_vectors, triad, trajectories
from pose3d.visualize_flow import load

PASCAL_VAL_COUNTS = (228, 88, 130, 91, 125, 185, 96, 7, 90, 27, 100, 142)     # aeroplane ... tvmonitor, 1309 test images


def modes_of(finals, thr):
    """Greedy clusters of (k, 3, 3) rotations within `thr` degrees: [{center, mass, members}], dominant first."""
    k = len(finals)
    dist = angle_deg(finals[:, None], finals[None])
    free, out = np.ones(k, bool), []
    while free.any():
        c = int(np.argmax(((dist < thr) & free[None, :]).sum(1) * free))
        members = np.where((dist[c] < thr) & free)[0]
        center = finals[members[int(dist[np.ix_(members, members)].sum(1).argmin())]]
        out.append(dict(center=center, mass=len(members) / k, members=members))
        free[members] = False
    return sorted(out, key=lambda m: -m["mass"])


def summarize(finals, gt, thr, min_mass):
    """Per-image statistics from the (k, 3, 3) final rotations and the true rotation."""
    ms = modes_of(finals, thr)
    big = [m for m in ms if m["mass"] >= min_mass] or ms[:1]       # the dominant group always counts, even when the samples are scattered
    dist = angle_deg(finals[:, None], finals[None])
    mode_err = [float(angle_deg(m["center"], gt)) for m in big]
    return dict(top_mass=ms[0]["mass"], n_modes=len(big), top_err=mode_err[0], best_err=min(mode_err),
                medoid_err=float(angle_deg(finals[int(dist.sum(1).argmin())], gt)),
                spread=float(np.median(dist[np.triu_indices(len(finals), 1)])),
                mode_gap=float(angle_deg(big[0]["center"], big[1]["center"])) if len(big) > 1 else np.nan, modes=big)


def run_model(model, imgs, rots, k, steps, seed, thr, min_mass, label, cache=None):
    """Per-image summaries; the sampled final rotations are cached in `cache` (an .npy path) so a rerun only redoes the statistics."""
    if cache is not None and Path(cache).exists() and np.load(cache).shape[:2] == (len(imgs), k):
        finals = np.load(cache)
    else:
        finals = []
        for i in range(len(imgs)):
            finals.append(trajectories(model, imgs[i], k, steps, seed)[-1])
            if (i + 1) % 50 == 0:
                print(f"  {label}: {i + 1}/{len(imgs)}", flush=True)
        finals = np.stack(finals)
        if cache is not None:
            np.save(cache, finals)
    return [summarize(finals[i], rots[i], thr, min_mass) for i in range(len(imgs))]


def figure_analysis(results, cls, min_mass, out):
    labels = list(results)
    cols = {l: c for l, c in zip(labels, ("tab:orange", "tab:blue", "tab:purple"))}
    fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
    for l in labels:
        ax[0, 0].hist([r["top_mass"] for r in results[l]], bins=np.linspace(0, 1, 17), alpha=0.55, color=cols[l], label=l)
    ax[0, 0].set(xlabel="share of samples in the dominant mode", ylabel="validation images", title="How concentrated is the flow's distribution?")
    ax[0, 0].legend()
    width = 0.8 / len(labels)
    for j, l in enumerate(labels):
        share = [np.mean([r["n_modes"] >= 2 for r, c in zip(results[l], cls) if c == k]) if (cls == k).any() else 0 for k in range(12)]
        ax[0, 1].bar(np.arange(12) + j * width, share, width, color=cols[l], label=l)
    ax[0, 1].set_xticks(np.arange(12) + 0.4 - width / 2); ax[0, 1].set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax[0, 1].set(ylabel=f"share with 2+ modes (each >= {min_mass:.0%} of samples)", title="Which classes split the flow?"); ax[0, 1].legend()
    for l in labels:
        sp, er = np.array([r["spread"] for r in results[l]]), np.array([r["medoid_err"] for r in results[l]])
        rho = spearmanr(sp, er)[0]
        ax[1, 0].scatter(sp, er, s=12, alpha=0.5, color=cols[l], label=f"{l} (Spearman {rho:.2f})")
    ax[1, 0].set(xlabel="spread of the samples (median pairwise angle, deg)", ylabel="error of the medoid prediction (deg)", yscale="log",
                 title="Does the flow know when it is unsure?"); ax[1, 0].legend(); ax[1, 0].grid(alpha=0.3)
    for l in labels:
        gaps = [r["mode_gap"] for r in results[l] if not np.isnan(r["mode_gap"])]
        ax[1, 1].hist(gaps, bins=np.linspace(0, 180, 19), alpha=0.55, color=cols[l], label=f"{l} ({len(gaps)} images)")
    ax[1, 1].set(xlabel="angle between the two strongest modes (deg)", ylabel="images", title="How far apart are competing poses?"); ax[1, 1].legend()
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def figure_gallery(label, rows, imgs, rots, cls, out, n=6):
    cand = [i for i, r in enumerate(rows) if r["n_modes"] >= 2]
    cand.sort(key=lambda i: -min(rows[i]["modes"][0]["mass"], rows[i]["modes"][1]["mass"]))
    pick, per_class = [], {}
    for i in cand:                                      # at most two images per class
        if per_class.get(cls[i], 0) < 2:
            pick.append(i); per_class[cls[i]] = per_class.get(cls[i], 0) + 1
        if len(pick) == n: break
    if not pick: return
    fig = plt.figure(figsize=(7.2, 3.4 * len(pick)))
    for r, i in enumerate(pick):
        ax = fig.add_subplot(len(pick), 2, 2 * r + 1); ax.imshow(imgs[i].permute(1, 2, 0).numpy()); ax.axis("off")
        ms = rows[i]["modes"]
        masses = ", ".join("%.0f%%" % (100 * m["mass"]) for m in ms[:3])
        ax.set_title(f"{CLASS_NAMES[cls[i]]} (val #{i}): modes {masses}", fontsize=9)
        a3 = fig.add_subplot(len(pick), 2, 2 * r + 2, projection="3d")
        triad(a3, rots[i], "--", 6, 0.25, "truth")
        for m, (st, lw) in zip(ms[:2], (("-", 2.6), (":", 2.6))):
            triad(a3, m["center"], st, lw, 1.0)
        a3.set_xlim(-1, 1); a3.set_ylim(-1, 1); a3.set_zlim(-1, 1); a3.set_box_aspect((1, 1, 1)); a3.view_init(22, 40)
        a3.set_xticks([]); a3.set_yticks([]); a3.set_zticks([])
        errs = [angle_deg(m["center"], rots[i]) for m in ms[:2]]
        a3.set_title(f"solid: top mode ({errs[0]:.0f}° off), dotted: 2nd ({errs[1]:.0f}° off)", fontsize=8)
    fig.suptitle(f"{label}: images where the flow splits (wide faint = truth)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out, dpi=105); plt.close(fig)


def interactive_html(models, imgs, rots, picks, cls, k, steps, seed, out):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, column_widths=[0.28, 0.72], specs=[[{"type": "xy"}, {"type": "scene"}]], horizontal_spacing=0.02)
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    groups, labels = [], []
    for ml, model in models.items():
        for i in picks:
            vec = relative_vectors(trajectories(model, imgs[i], k, steps, seed), rots[i])
            t = np.linspace(0, 1, vec.shape[0])
            start = len(fig.data)
            fig.add_trace(go.Image(z=(imgs[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8), hoverinfo="skip"), row=1, col=1)
            fig.add_trace(go.Surface(x=np.pi * np.cos(u) * np.sin(v), y=np.pi * np.sin(u) * np.sin(v), z=np.pi * np.cos(v), opacity=0.07,
                                     showscale=False, colorscale=[[0, "#999"], [1, "#999"]], hoverinfo="skip"), row=1, col=2)
            fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker=dict(size=9, color="gold", symbol="diamond", line=dict(color="black", width=2)),
                                       name="true pose", hovertext="true pose"), row=1, col=2)
            for j in range(vec.shape[1]):
                seg = vec[:, j].copy(); ang = np.degrees(np.linalg.norm(seg, axis=1))
                jump = np.r_[False, np.linalg.norm(np.diff(seg, axis=0), axis=1) > np.pi]       # antipodal wrap: leave a gap
                seg[jump] = np.nan
                fig.add_trace(go.Scatter3d(x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines", showlegend=False,
                                           line=dict(width=4, color=t, colorscale="Viridis", cmin=0, cmax=1),
                                           customdata=np.stack([t, ang], 1), hovertemplate="t=%{customdata[0]:.2f}<br>%{customdata[1]:.1f}° from the truth<extra></extra>"), row=1, col=2)
            ok = np.degrees(np.linalg.norm(vec[-1], axis=1)) < 15
            fig.add_trace(go.Scatter3d(x=vec[0, :, 0], y=vec[0, :, 1], z=vec[0, :, 2], mode="markers", name="noise",
                                       marker=dict(size=3, color="white", line=dict(color="gray", width=1)), hoverinfo="skip"), row=1, col=2)
            fig.add_trace(go.Scatter3d(x=vec[-1, :, 0], y=vec[-1, :, 1], z=vec[-1, :, 2], mode="markers", name="final",
                                       marker=dict(size=5, color=np.where(ok, "green", "red"), line=dict(color="black", width=1)),
                                       hovertemplate="final: %{customdata:.1f}° from the truth<extra></extra>", customdata=np.degrees(np.linalg.norm(vec[-1], axis=1))), row=1, col=2)
            groups.append((start, len(fig.data)))
            labels.append(f"{ml}: val #{i} ({CLASS_NAMES[cls[i]]}), {int(ok.sum())}/{vec.shape[1]} samples within 15°")
    for gi, (a, b) in enumerate(groups):
        for tr in fig.data[a:b]:
            tr.visible = gi == 0
    buttons = [dict(label=lab, method="update", args=[{"visible": [a <= n < b for n in range(len(fig.data))]}, {"title": lab}]) for (a, b), lab in zip(groups, labels)]
    r = np.pi * 1.05
    fig.update_layout(title=labels[0], updatemenus=[dict(buttons=buttons, direction="down", x=0.0, y=1.12, xanchor="left", showactive=True)],
                      scene=dict(xaxis=dict(visible=False, range=[-r, r]), yaxis=dict(visible=False, range=[-r, r]), zaxis=dict(visible=False, range=[-r, r]), aspectmode="cube"),
                      margin=dict(l=10, r=10, t=90, b=10), height=640)
    fig.update_xaxes(visible=False, row=1, col=1); fig.update_yaxes(visible=False, row=1, col=1)
    fig.write_html(out, include_plotlyjs=True, full_html=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", action="append", required=True); p.add_argument("--label", action="append", required=True)
    p.add_argument("--val_cache", required=True); p.add_argument("--out", default="viz")
    p.add_argument("--n_images", type=int, default=300); p.add_argument("--k", type=int, default=32); p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--mode_deg", type=float, default=30.0); p.add_argument("--min_mass", type=float, default=0.15)
    p.add_argument("--examples", type=int, nargs="+", default=[1170, 922, 454, 908], help="validation indices for the interactive plot")
    a = p.parse_args(); assert len(a.checkpoint) == len(a.label)
    torch.set_num_threads(4); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    blob = torch.load(a.val_cache, map_location="cpu", weights_only=True)
    imgs, rots = blob["imgs"].float() / 255, blob["targets"].float().numpy()
    cls_all = np.repeat(np.arange(12), PASCAL_VAL_COUNTS) if len(imgs) == sum(PASCAL_VAL_COUNTS) else np.full(len(imgs), -1)
    idx = np.unique(np.linspace(0, len(imgs) - 1, a.n_images).astype(int))
    models = {lab: load(ck) for ck, lab in zip(a.checkpoint, a.label)}
    results = {}
    for lab, m in models.items():
        results[lab] = run_model(m, imgs[idx], rots[idx], a.k, a.steps, a.seed, a.mode_deg, a.min_mass, lab, cache=out / f"finals_{lab}.npy")
        r = results[lab]
        print(f"{lab}: {len(r)} images | 2+ modes in {np.mean([x['n_modes'] >= 2 for x in r]):.1%} | dominant-mode mass median {np.median([x['top_mass'] for x in r]):.2f} | "
              f"medoid err median {np.median([x['medoid_err'] for x in r]):.2f} | top-mode wrong (>15) but another mode right: "
              f"{np.mean([(x['top_err'] > 15) and (x['best_err'] <= 15) for x in r]):.1%} | oracle-mode err median {np.median([x['best_err'] for x in r]):.2f}", flush=True)
    cls = cls_all[idx]
    figure_analysis(results, cls, a.min_mass, out / "modes_analysis.png")
    for lab, m in models.items():
        figure_gallery(lab, results[lab], imgs[idx], rots[idx], cls, out / f"bimodal_gallery_{lab}.png")
    interactive_html(models, imgs, rots, a.examples, cls_all, min(a.k, 24), a.steps, a.seed, out / "interactive_paths.html")
    np.savez(out / "modes_summary.npz", idx=idx, **{f"{l}_{k}": np.array([r[k] for r in rs]) for l, rs in results.items() for k in ("top_mass", "n_modes", "top_err", "best_err", "medoid_err", "spread", "mode_gap")})


if __name__ == "__main__":
    main()

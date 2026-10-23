"""Flow-matching trajectories of a trained CliffordFlow on SO(3), and how to log them to Weights & Biases.

The flow is integrated from several random rotations (the Euler loop of CliffordFlow.predict, but every intermediate rotation is kept).
Each rotation R_t is shown relative to the ground truth, R_gt^T R_t, as an axis-angle vector inside the ball of radius pi: the true pose is
the origin, the distance from the origin is the geodesic error, and a path that ends near the origin has found the pose. Opposite points
of the ball's surface are the same rotation, so a path can jump across it (drawn as a break).
"""
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation

from pose3d.geometry.flow import exp_map, rotor_multiply
from pose3d.geometry.rotor import random_rotor, rotor_to_matrix

CLASS_NAMES = ("aeroplane", "bicycle", "boat", "bottle", "bus", "car", "chair",
               "diningtable", "motorbike", "sofa", "train", "tvmonitor")


def angle_deg(r_a, r_b):
    """Geodesic angle between rotation matrices (..., 3, 3), in degrees."""
    tr = np.einsum("...ji,...ji->...", r_a, r_b)          # trace(A^T B)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


@torch.no_grad()
def rotor_trajectories(model, img, k=24, steps=20, seed=0):
    """(steps + 1, k, 4): the flow's unit rotors (scalar, e12, e13, e23), k noise rotors integrated to the model's pose, every Euler step kept."""
    dev = next(model.parameters()).device
    cond = model.condition(img[None].to(dev)).repeat(k, 1, 1)
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)                      # the same noise for every model and every call
    rotor = random_rotor(k).to(dev)
    torch.random.set_rng_state(state)            # ... without disturbing the training run's random stream
    path = [rotor]
    dt = 1.0 / steps
    for i in range(steps):
        v = model.velocity(rotor, torch.full((k,), i * dt, device=dev), cond)
        rotor = rotor_multiply(rotor, exp_map(dt * v), model.algebra)
        path.append(rotor)
    return torch.stack(path).cpu().numpy()


@torch.no_grad()
def trajectories(model, img, k=24, steps=20, seed=0):
    """(steps + 1, k, 3, 3): k noise rotations integrated to the model's pose, every Euler step kept."""
    rotors = torch.from_numpy(rotor_trajectories(model, img, k, steps, seed))
    return torch.stack([rotor_to_matrix(r, model.algebra) for r in rotors]).numpy()


def _finish(fig, out):
    """Save and close when a path is given, else hand the figure back (e.g. for wandb.Image)."""
    if out:
        fig.savefig(out, dpi=110)
        plt.close(fig)
        return None
    return fig


@torch.no_grad()
def errors(model, imgs, rots, n_samples=8, batch=16):
    out = []
    for i in range(0, len(imgs), batch):
        pred = model.predict(imgs[i:i + batch].to(next(model.parameters()).device), n_samples=n_samples, steps=20).cpu().numpy()
        out.append(angle_deg(pred, rots[i:i + batch]))
    return np.concatenate(out)


def relative_vectors(path, gt):
    """Axis-angle vectors of R_gt^T R_t, shape (steps + 1, k, 3); the norm is the angle, at most pi."""
    rel = np.einsum("ji,tkjl->tkil", gt, path)
    return Rotation.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec().reshape(path.shape[:2] + (3,))


def draw_ball(ax):
    u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
    ax.plot_wireframe(np.pi * np.cos(u) * np.sin(v), np.pi * np.sin(u) * np.sin(v), np.pi * np.cos(v), color="0.8", lw=0.3, alpha=0.5)
    ax.scatter([0], [0], [0], marker="*", s=130, c="gold", edgecolors="k", zorder=10, label="true pose")
    ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi); ax.set_zlim(-np.pi, np.pi)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()


def draw_paths(ax, vec, cmap="viridis"):
    """Time-coloured paths; a jump across the ball's surface (antipodal identification) is left as a gap."""
    n_t, k, _ = vec.shape
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, n_t - 1))
    for j in range(k):
        seg = np.stack([vec[:-1, j], vec[1:, j]], axis=1)
        keep = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1) < np.pi
        ax.add_collection3d(Line3DCollection(seg[keep], colors=colors[keep], linewidths=1.2, alpha=0.85))
    ax.scatter(*vec[0].T, s=14, facecolors="none", edgecolors="0.35", linewidths=0.6)                      # noise
    final = np.linalg.norm(vec[-1], axis=1)
    ax.scatter(*vec[-1].T, s=22, c=np.where(np.degrees(final) < 15, "tab:green", "tab:red"), edgecolors="k", linewidths=0.3)


def triad(ax, rot, style, lw, alpha, label=None):
    for c, col in enumerate(("tab:red", "tab:green", "tab:blue")):
        ax.plot([0, rot[0, c]], [0, rot[1, c]], [0, rot[2, c]], color=col, ls=style, lw=lw, alpha=alpha, label=label if c == 0 else None)


def pick_images(model, imgs, rots, n_pool, quantiles):
    idx = np.linspace(0, len(imgs) - 1, n_pool).astype(int)
    err = errors(model, imgs[idx], rots[idx])
    order = np.argsort(err)
    chosen = [int(order[min(int(q * len(order)), len(order) - 1)]) for q in quantiles]
    return [int(idx[c]) for c in chosen], [float(err[c]) for c in chosen]


def figure_paths(label, model, imgs, rots, picks, errs, k, steps, seed, out=None):
    fig = plt.figure(figsize=(14, 3.7 * len(picks)))
    for r, (i, e) in enumerate(zip(picks, errs)):
        path = trajectories(model, imgs[i], k, steps, seed)
        vec = relative_vectors(path, rots[i])
        dist = np.linalg.norm(vec, axis=-1) * 180 / np.pi
        ax = fig.add_subplot(len(picks), 3, 3 * r + 1); ax.imshow(imgs[i].permute(1, 2, 0).numpy()); ax.axis("off")
        ax.set_title(f"val #{i}: medoid error {e:.1f}°", fontsize=10)
        ax3 = fig.add_subplot(len(picks), 3, 3 * r + 2, projection="3d"); draw_ball(ax3); draw_paths(ax3, vec)
        ax3.set_title("chart centred on the truth (radius = angle to it)", fontsize=10)
        axd = fig.add_subplot(len(picks), 3, 3 * r + 3)
        t = np.linspace(0, 1, dist.shape[0])
        axd.plot(t, dist, color="tab:blue", alpha=0.25, lw=0.9); axd.plot(t, np.median(dist, axis=1), color="k", lw=2, label="median of samples")
        axd.axhline(15, color="tab:green", ls="--", lw=1, label="15°"); axd.set_ylim(0, 180); axd.set_xlabel("flow time t"); axd.set_ylabel("angle to true pose (°)")
        axd.grid(alpha=0.3); axd.legend(fontsize=8, loc="upper right")
        axd.set_title(f"{(dist[-1] < 15).sum()}/{k} samples end within 15°", fontsize=10)
    fig.suptitle(f"{label}: flow from noise to pose on SO(3), {k} noise samples, {steps} Euler steps", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); return _finish(fig, out)


AXIS_COLORS = ("tab:red", "tab:green", "tab:blue")


def draw_unit_sphere(ax):
    u, v = np.mgrid[0:2 * np.pi:36j, 0:np.pi:18j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="0.82", lw=0.3, alpha=0.6)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1); ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()


def draw_frame_paths(ax, path, gt, n_show=6):
    """A rotation is a frame of three perpendicular unit vectors; their tips (columns of R) always lie on the unit sphere.

    Thin curves: the tips of the x (red), y (green) and z (blue) axes along n_show noise samples, hollow circle = noise, dot = where the
    flow ends. Stars: the tips of the true frame. This is the absolute motion, not a chart centred on the answer.
    """
    draw_unit_sphere(ax)
    for j in range(min(n_show, path.shape[1])):
        for a, col in enumerate(AXIS_COLORS):
            tip = path[:, j, :, a]
            ax.plot(*tip.T, color=col, lw=1.1, alpha=0.65)
            ax.scatter(*tip[0], s=14, facecolors="none", edgecolors=col, linewidths=0.8)
            ax.scatter(*tip[-1], s=16, color=col, edgecolors="k", linewidths=0.3)
    for a, col in enumerate(AXIS_COLORS):
        ax.scatter(*gt[:, a], marker="*", s=170, color=col, edgecolors="k", linewidths=0.9, zorder=10)


_HOPF_REF = np.array([0.267, 0.535, 0.802])        # arbitrary generic direction: the spin colour is undefined only where the axis points exactly along it


def hopf_coords(rot):
    """Hopf-style coordinates of rotations (..., 3, 3): where the z axis points (a point on the unit sphere) and the spin of the frame
    around that axis, in degrees [0, 360). Together they identify the rotation uniquely, so one sphere plus one colour shows all of SO(3).
    """
    rot = np.asarray(rot, dtype=np.float64)
    p, x = rot[..., :, 2], rot[..., :, 0]
    a = _HOPF_REF - (p @ _HOPF_REF)[..., None] * p                                  # reference tangent direction at p
    a /= np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12
    b = np.cross(p, a)
    return p, np.degrees(np.arctan2((x * b).sum(-1), (x * a).sum(-1))) % 360.0


def figure_hopf(label, model, imgs, rots, picks, errs, k, steps, seed, out=None, view=(20, 35)):
    """Per image: photo | ONE sphere. The position is where the z axis points, the colour is the spin around it (cyclic hue, see the bar).

    Hollow circle = noise, curve = the flow, dot = its end, star = the true pose: a path has found the pose when it ends at the star AND has the same colour.
    """
    cmap = plt.get_cmap("hsv")
    fig = plt.figure(figsize=(9.5, 4.2 * len(picks)))
    for r, (i, e) in enumerate(zip(picks, errs)):
        path = trajectories(model, imgs[i], k, steps, seed)
        dist = angle_deg(path, rots[i]); pos, spin = hopf_coords(path); gp, gs = hopf_coords(rots[i])
        ax = fig.add_subplot(len(picks), 2, 2 * r + 1); ax.imshow(imgs[i].permute(1, 2, 0).numpy()); ax.axis("off")
        ax.set_title(f"val #{i}: medoid error {e:.1f}°\n{(dist[-1] < 15).sum()}/{k} samples end within 15°", fontsize=10)
        a3 = fig.add_subplot(len(picks), 2, 2 * r + 2, projection="3d"); draw_unit_sphere(a3); a3.view_init(*view)
        for j in range(path.shape[1]):
            pts = pos[:, j]; seg = np.stack([pts[:-1], pts[1:]], 1)
            a3.add_collection3d(Line3DCollection(seg, colors=cmap(((spin[:-1, j] + spin[1:, j]) / 2) / 360.0), linewidths=2.0))
            a3.scatter(*pts[0], s=26, c=[cmap(spin[0, j] / 360.0)], edgecolors="white", linewidths=1.2)
            a3.scatter(*pts[-1], s=30, c=[cmap(spin[-1, j] / 360.0)], edgecolors="k", linewidths=0.8)
        a3.scatter(*gp, marker="*", s=380, c=[cmap(gs / 360.0)], edgecolors="k", linewidths=1.4, zorder=10)
        a3.set_title("z-axis direction on the sphere, colour = spin around it", fontsize=10)
        if r == 0:
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 360), cmap=cmap), ax=a3, shrink=0.55, pad=0.0); cb.set_label("spin (°)")
    fig.suptitle(f"{label}: noise (○) to the true pose (★); each rotation = a point + a colour", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); return _finish(fig, out)


def figure_frames(label, model, imgs, rots, picks, errs, k, steps, seed, out=None, view=(20, 35)):
    """Per image: photo | one unit sphere per frame axis (x, y, z). Hollow circle = noise start, curve = flow, dot = end, star = truth.

    A rotation is a frame of three perpendicular unit vectors, so each axis tip lives on a sphere and the flow never leaves it.
    """
    fig = plt.figure(figsize=(15, 3.8 * len(picks)))
    cmap = plt.get_cmap("tab10")
    for r, (i, e) in enumerate(zip(picks, errs)):
        path = trajectories(model, imgs[i], k, steps, seed)
        dist = angle_deg(path, rots[i])
        ax = fig.add_subplot(len(picks), 4, 4 * r + 1); ax.imshow(imgs[i].permute(1, 2, 0).numpy()); ax.axis("off")
        ax.set_title(f"val #{i}: medoid error {e:.1f}°\n{(dist[-1] < 15).sum()}/{k} samples end within 15°", fontsize=10)
        for a, name in enumerate("xyz"):
            a3 = fig.add_subplot(len(picks), 4, 4 * r + 2 + a, projection="3d"); draw_unit_sphere(a3); a3.view_init(*view)
            for j in range(path.shape[1]):
                tip = path[:, j, :, a]; col = cmap(j % 10)
                a3.plot(*tip.T, color=col, lw=1.3, alpha=0.8)
                a3.scatter(*tip[0], s=22, facecolors="white", edgecolors=col, linewidths=1.0)
                a3.scatter(*tip[-1], s=18, color=col, edgecolors="k", linewidths=0.3)
            a3.scatter(*rots[i][:, a], marker="*", s=260, color="gold", edgecolors="k", linewidths=1.0, zorder=10)
            a3.set_title(f"tip of the {name}-axis on the unit sphere", fontsize=10)
    fig.suptitle(f"{label}: noise (○) flowing to the true pose (★); every point stays on the sphere", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); return _finish(fig, out)


def interactive_frames(models, imgs, rots, picks, k, steps, seed, out, cls=None, errs=None):
    """Self-contained plotly HTML, Hopf view: ONE unit sphere you can drag; the position is where the z axis points, the colour (cyclic) is the spin
    around it. Hover a curve for flow time, error and spin. Hollow = noise, dot = end (outline green within 15 deg, red otherwise), diamond = truth.
    The dropdown picks (model, image).
    """
    import plotly.graph_objects as go
    import torch.nn.functional as F
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, column_widths=[0.28, 0.72], horizontal_spacing=0.01, specs=[[{"type": "xy"}, {"type": "scene"}]])
    u, v = np.mgrid[0:2 * np.pi:48j, 0:np.pi:24j]
    groups, labels = [], []
    for ml, model in models.items():
        for n, i in enumerate(picks):
            path = trajectories(model, imgs[i], k, steps, seed)
            dist = angle_deg(path, rots[i]); t = np.linspace(0, 1, path.shape[0]); ok = dist[-1] < 15
            pos, spin = hopf_coords(path); gp, gs = hopf_coords(rots[i])
            small = F.interpolate(imgs[i][None].float(), size=(160, 160), mode="bilinear", align_corners=False)[0]
            start = len(fig.data)
            fig.add_trace(go.Image(z=(small.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8), hoverinfo="skip"), row=1, col=1)
            fig.add_trace(go.Surface(x=np.cos(u) * np.sin(v), y=np.sin(u) * np.sin(v), z=np.cos(v), opacity=0.1, showscale=False, hoverinfo="skip",
                                     colorscale=[[0, "#aab"], [1, "#aab"]]), row=1, col=2)
            for j in range(path.shape[1]):
                fig.add_trace(go.Scatter3d(x=pos[:, j, 0], y=pos[:, j, 1], z=pos[:, j, 2], mode="lines", showlegend=False,
                                           line=dict(width=6, color=spin[:, j], colorscale="HSV", cmin=0, cmax=360),
                                           customdata=np.stack([t, dist[:, j], spin[:, j]], 1),
                                           hovertemplate=f"sample {j}<br>t=%{{customdata[0]:.2f}}<br>%{{customdata[1]:.1f}}° from the truth<br>spin %{{customdata[2]:.0f}}°<extra></extra>"), row=1, col=2)
            fig.add_trace(go.Scatter3d(x=pos[0, :, 0], y=pos[0, :, 1], z=pos[0, :, 2], mode="markers", showlegend=False, hoverinfo="skip",
                                       marker=dict(size=5, color=spin[0], colorscale="HSV", cmin=0, cmax=360, line=dict(color="white", width=3))), row=1, col=2)
            fig.add_trace(go.Scatter3d(x=pos[-1, :, 0], y=pos[-1, :, 1], z=pos[-1, :, 2], mode="markers", showlegend=False,
                                       marker=dict(size=6, color=spin[-1], colorscale="HSV", cmin=0, cmax=360, line=dict(color=np.where(ok, "green", "red"), width=3)),
                                       customdata=np.stack([dist[-1], spin[-1]], 1), hovertemplate="final: %{customdata[0]:.1f}° from the truth<br>spin %{customdata[1]:.0f}°<extra></extra>"), row=1, col=2)
            fig.add_trace(go.Scatter3d(x=[gp[0]], y=[gp[1]], z=[gp[2]], mode="markers", showlegend=False, hovertext=f"true pose (spin {gs:.0f}°)",
                                       marker=dict(size=11, color=[gs], colorscale="HSV", cmin=0, cmax=360, symbol="diamond", line=dict(color="black", width=3),
                                                   colorbar=dict(title="spin (°)", len=0.6, x=1.0))), row=1, col=2)
            groups.append((start, len(fig.data)))
            name = f" ({CLASS_NAMES[cls[i]]})" if cls is not None and 0 <= cls[i] < len(CLASS_NAMES) else ""
            err = f", medoid error {errs[n]:.1f}°" if errs is not None else ""
            labels.append(f"{ml}: val #{i}{name}{err}; {int(ok.sum())}/{path.shape[1]} samples end within 15°")
    for gi, (a0, b0) in enumerate(groups):
        for tr in fig.data[a0:b0]:
            tr.visible = gi == 0
    buttons = [dict(label=lab, method="update", args=[{"visible": [a0 <= n < b0 for n in range(len(fig.data))]}, {"title.text": lab}]) for (a0, b0), lab in zip(groups, labels)]
    ax = dict(visible=False, range=[-1.05, 1.05])
    fig.update_layout(title=dict(text=labels[0], x=0.0, xanchor="left", y=0.97), updatemenus=[dict(buttons=buttons, direction="down", x=0.0, y=1.0, xanchor="left", yanchor="bottom", pad=dict(b=6), showactive=True)],
                      scene=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="cube", camera=dict(eye=dict(x=1.5, y=1.5, z=0.9))),
                      margin=dict(l=5, r=5, t=130, b=5), height=640)
    fig.update_xaxes(visible=False, row=1, col=1); fig.update_yaxes(visible=False, row=1, col=1)
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    return out


def figure_filmstrip(label, model, imgs, rots, i, steps, seed, out=None, sample=0):
    path = trajectories(model, imgs[i], 8, steps, seed)
    times = [0, steps // 4, steps // 2, 3 * steps // 4, steps]
    fig = plt.figure(figsize=(3 * (len(times) + 1), 3.6))
    ax = fig.add_subplot(1, len(times) + 1, 1); ax.imshow(imgs[i].permute(1, 2, 0).numpy()); ax.axis("off"); ax.set_title(f"val #{i}", fontsize=10)
    gt = rots[i]
    for c, s in enumerate(times):
        a3 = fig.add_subplot(1, len(times) + 1, c + 2, projection="3d")
        triad(a3, gt, "--", 5, 0.25); triad(a3, path[s, sample], "-", 2.2, 1.0)
        a3.set_xlim(-1, 1); a3.set_ylim(-1, 1); a3.set_zlim(-1, 1); a3.set_box_aspect((1, 1, 1)); a3.view_init(22, 40)
        a3.set_xticks([]); a3.set_yticks([]); a3.set_zticks([])
        a3.set_title(f"t = {s / steps:.2f}  ({angle_deg(path[s, sample], gt):.0f}° off)", fontsize=10)
    fig.suptitle(f"{label}: one noise sample, rotating its frame (solid R,G,B axes) into the true pose (faint, wide axes)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); return _finish(fig, out)


def figure_compare(models, imgs, rots, picks, errs, k, steps, seed, out):
    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 3.8), sharey=True)
    t = np.linspace(0, 1, steps + 1)
    for c, (i, ax) in enumerate(zip(picks, np.atleast_1d(axes))):
        for (label, model), col in zip(models.items(), ("tab:orange", "tab:blue", "tab:purple")):
            vec = relative_vectors(trajectories(model, imgs[i], k, steps, seed), rots[i])      # same seed = same noise for every model
            d = np.linalg.norm(vec, axis=-1) * 180 / np.pi
            ax.fill_between(t, np.percentile(d, 25, axis=1), np.percentile(d, 75, axis=1), color=col, alpha=0.2)
            ax.plot(t, np.median(d, axis=1), color=col, lw=2.2, label=f"{label} (median, IQR)")
        ax.axhline(15, color="tab:green", ls="--", lw=1); ax.set_xlabel("flow time t"); ax.set_title(f"val #{i}", fontsize=10); ax.grid(alpha=0.3); ax.set_ylim(0, 180)
        if c == 0: ax.set_ylabel("angle to true pose (°)"); ax.legend(fontsize=8)
    fig.suptitle(f"Same images, same {k} noise rotations: how fast each model's flow converges", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); return _finish(fig, out)


def animation(label, model, imgs, rots, i, k, steps, seed, out):
    vec = relative_vectors(trajectories(model, imgs[i], k, steps, seed), rots[i])
    fig = plt.figure(figsize=(8, 4.3)); ax = fig.add_subplot(1, 2, 1); ax.imshow(imgs[i].permute(1, 2, 0).numpy()); ax.axis("off")
    a3 = fig.add_subplot(1, 2, 2, projection="3d"); draw_ball(a3)
    sc = a3.scatter(*vec[0].T, s=26, c="tab:blue", edgecolors="k", linewidths=0.3); lines = [a3.plot([], [], [], color="tab:blue", lw=0.8, alpha=0.5)[0] for _ in range(k)]
    title = fig.suptitle("", fontsize=11)
    def frame(f):
        sc._offsets3d = tuple(vec[f].T)
        sc.set_color(np.where(np.linalg.norm(vec[f], axis=1) < np.radians(15), "tab:green", "tab:blue"))
        for j, ln in enumerate(lines):
            seg = vec[:f + 1, j]
            ok = np.r_[True, np.linalg.norm(np.diff(seg, axis=0), axis=1) < np.pi]
            seg = np.where(ok[:, None], seg, np.nan); ln.set_data(seg[:, 0], seg[:, 1]); ln.set_3d_properties(seg[:, 2])
        a3.view_init(18, 30 + 90 * f / steps)
        title.set_text(f"{label}: t = {f / steps:.2f}, {int((np.linalg.norm(vec[f], axis=1) < np.radians(15)).sum())}/{k} within 15°")
        return [sc, *lines]
    FuncAnimation(fig, frame, frames=steps + 1, blit=False).save(out, writer=PillowWriter(fps=6)); plt.close(fig)




# ------------------------------------------------------------------------------------------------------------ Weights & Biases
def collect_examples(dataset, n):
    """n evenly spaced validation samples (the same ones at every call, so epochs are comparable): images, rotations, class ids, indices."""
    idx = np.unique(np.linspace(0, len(dataset) - 1, n).astype(int))
    items = [dataset[int(i)] for i in idx]
    imgs = torch.stack([torch.as_tensor(it["img"]).float() for it in items])
    rots = torch.stack([torch.as_tensor(it["rot"]).float() for it in items]).numpy()
    cls = [int(torch.as_tensor(it.get("cls_eval", it.get("cls", -1))).view(-1)[0]) for it in items]
    return imgs, rots, cls, [int(i) for i in idx]


def _medoid(final_rots):
    """The sample closest to all the others (what predict(n_samples>1) returns), from (k, 3, 3) rotation matrices."""
    d = angle_deg(final_rots[:, None], final_rots[None])
    return final_rots[int(d.sum(1).argmin())]


def wandb_media(model, dataset, n=6, k=16, steps=20, seed=0, final=False):
    """Everything worth looking at, as a dict for run.log: a predictions table, the path figure and, at the end, a film strip and an animation.

    The model is put in eval mode and back, and the global random state is left alone, so this can be called in the middle of training.
    """
    import wandb

    was_training = model.training
    model.eval()
    try:
        imgs, rots, cls, idx = collect_examples(dataset, n)
        table = wandb.Table(columns=["val_index", "image", "class", "medoid_error_deg", "median_sample_error_deg",
                                     "frac_samples_within_15deg", "true_rotvec", "predicted_rotvec"])
        medoid_err = []
        for j in range(len(idx)):
            final_rots = trajectories(model, imgs[j], k, steps, seed)[-1]                       # (k, 3, 3)
            sample_err = angle_deg(final_rots, rots[j])
            pred = _medoid(final_rots); medoid_err.append(float(angle_deg(pred, rots[j])))
            name = CLASS_NAMES[cls[j]] if 0 <= cls[j] < len(CLASS_NAMES) else ""
            table.add_data(idx[j], wandb.Image(imgs[j].permute(1, 2, 0).numpy()), name, medoid_err[-1], float(np.median(sample_err)),
                           float((sample_err < 15).mean()), str(np.round(Rotation.from_matrix(rots[j]).as_rotvec(), 3).tolist()),
                           str(np.round(Rotation.from_matrix(pred).as_rotvec(), 3).tolist()))
        picks = list(range(len(idx)))
        fig = figure_hopf("flow", model, imgs, rots, picks, medoid_err, k, steps, seed)
        media = {"viz/predictions": table, "viz/paths": wandb.Image(fig)}
        plt.close(fig)
        if final:
            fig = figure_filmstrip("flow", model, imgs, rots, picks[min(1, len(picks) - 1)], steps, seed)
            media["viz/filmstrip"] = wandb.Image(fig); plt.close(fig)
            gif = Path(tempfile.mkdtemp(prefix="flow_viz_")) / "flow.gif"
            animation("flow", model, imgs, rots, picks[min(1, len(picks) - 1)], k, steps, seed, str(gif))
            media["viz/flow_animation"] = wandb.Video(str(gif), fps=6, format="gif")
            html = gif.with_name("interactive.html")
            interactive_frames({"flow": model}, imgs, rots, picks, min(k, 12), steps, seed, str(html), cls=cls, errs=medoid_err)
            media["viz/interactive"] = wandb.Html(str(html), inject=False)
        return media
    finally:
        model.train(was_training)

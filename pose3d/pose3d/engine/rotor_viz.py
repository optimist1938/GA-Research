"""The flow in rotor (unit quaternion) space, where it actually lives: Spin(3) = S^3, the double cover of SO(3).

Both q and -q are the same rotation, so a ground-truth rotation R has TWO rotors, +-q. The geodesic interpolation of the flow-matching loss
(geometry/flow.py: log_map flips the sign so w >= 0) goes to whichever of the two is nearer to the noise rotor, so the learned field has two
attractors, one per hemisphere of S^3. This module draws that: S^3 cannot be embedded in 3D, so the rotors are shown

  * by stereographic projection from a pole P orthogonal to q_gt, y = (q.e0, q.e1, q.e2) / (1 - q.P) with e0 = q_gt: conformal, every point stays on
    S^3, and q_gt, -q_gt land on (+1, 0, 0) and (-1, 0, 0) (the pole is chosen away from the paths, so nothing is sent to infinity);
  * by the signed alignment c(t) = q(t) . q_gt, which is +1 at q_gt and -1 at -q_gt, a 1-D view that shows which of the two a sample picks;
  * by the four rotor components of one sample against +-q_gt.
"""

import numpy as np
import torch

from pose3d.engine.flow_viz import _finish, angle_deg, plt, rotor_trajectories
from pose3d.geometry.rotor import matrix_to_rotor, rotor_to_matrix

PLUS, MINUS = "tab:blue", "tab:orange"
COMPONENTS = ("1", "e12", "e13", "e23")


def gt_rotor(rot):
    """The ground-truth rotor q with w >= 0 (any of +-q is the same rotation), as float64 (4,)."""
    q = matrix_to_rotor(torch.as_tensor(rot, dtype=torch.float32)).double().numpy()
    return q if q[0] >= 0 else -q


def stereo_frame(q_gt, path, n_candidates=512, seed=0):
    """Orthonormal (e0 = q_gt, e1, e2, P): P is the stereographic pole, chosen orthogonal to q_gt as far from every path point as possible."""
    q_gt = np.asarray(q_gt, dtype=np.float64)
    basis = np.linalg.qr(np.column_stack([q_gt, np.eye(4)]))[0][:, 1:4]          # an orthonormal basis of q_gt's complement
    if np.dot(basis[:, 0], q_gt) > 0.5:                                           # qr may return q_gt's own direction first
        basis = np.linalg.qr(np.column_stack([q_gt, np.random.default_rng(seed).normal(size=(4, 3))]))[0][:, 1:4]
    cand = np.random.default_rng(seed).normal(size=(n_candidates, 3))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dirs = cand @ basis.T                                                         # (n, 4), all orthogonal to q_gt
    pts = path.reshape(-1, 4)
    worst = (dirs @ pts.T).max(1)                                                 # largest cosine to any path point: smaller = farther away
    P = dirs[int(worst.argmin())]
    rest = np.linalg.qr(np.column_stack([q_gt, P, np.random.default_rng(seed + 1).normal(size=(4, 2))]))[0][:, 2:4]
    return q_gt, rest[:, 0], rest[:, 1], P


def stereographic(x, frame):
    """Points (..., 4) of S^3 -> (..., 3): the projection from frame's pole P, with q_gt on the first axis."""
    e0, e1, e2, P = frame
    return np.stack([x @ e0, x @ e1, x @ e2], -1) / (1.0 - x @ P)[..., None]


def destinations(path, q_gt, deg=15.0):
    """+1 / -1: which of +-q_gt each sample ends nearest to; and whether it ends within `deg` degrees of that rotation."""
    c = path[-1] @ q_gt
    sign = np.where(c >= 0, 1, -1)
    err = np.degrees(2 * np.arccos(np.clip(np.abs(c), 0, 1)))                    # rotation error = twice the S^3 angle
    return sign, err < deg


def figure_rotors(label, model, imgs, rots, picks, errs, k, steps, seed, out=None, view=(18, 40)):
    """Per image: photo | S^3 by stereographic projection | signed alignment with q_gt over flow time | the 4 components of sample 0."""
    fig = plt.figure(figsize=(17, 3.9 * len(picks)))
    t = np.linspace(0, 1, steps + 1)
    for r, (i, e) in enumerate(zip(picks, errs)):
        path = rotor_trajectories(model, imgs[i], k, steps, seed).astype(np.float64)
        q = gt_rotor(rots[i])
        assert np.allclose(rotor_to_matrix(torch.from_numpy(np.stack([q, -q])).float(), model.algebra).numpy(), rots[i], atol=1e-4), "q and -q must both give the truth"
        sign, ok = destinations(path, q)
        frame = stereo_frame(q, path)
        y = stereographic(path, frame)
        n_plus, n_minus, n_miss = int(((sign > 0) & ok).sum()), int(((sign < 0) & ok).sum()), int((~ok).sum())

        ax = fig.add_subplot(len(picks), 4, 4 * r + 1); ax.imshow(imgs[i].permute(1, 2, 0).numpy()); ax.axis("off")
        ax.set_title(f"val #{i}: medoid error {e:.1f}°\n{n_plus} to +q, {n_minus} to −q, {n_miss} miss (>15°)", fontsize=10)

        a3 = fig.add_subplot(len(picks), 4, 4 * r + 2, projection="3d")
        for j in range(path.shape[1]):
            col = PLUS if sign[j] > 0 else MINUS
            a3.plot(*y[:, j].T, color=col, lw=1.2, alpha=0.8 if ok[j] else 0.35)
            a3.scatter(*y[0, j], s=16, facecolors="white", edgecolors=col, linewidths=1.0)
            a3.scatter(*y[-1, j], s=20, color=col, edgecolors="k" if ok[j] else "r", linewidths=0.8)
        for sgn, name in ((1, "q"), (-1, "−q")):
            a3.scatter(sgn, 0, 0, marker="*", s=420, c="gold", edgecolors="k", linewidths=1.2, zorder=10)
            a3.text(sgn * 1.15, 0, 0.18, name, fontsize=11, weight="bold")
        lim = 2.0
        a3.set_xlim(-lim, lim); a3.set_ylim(-lim, lim); a3.set_zlim(-lim, lim); a3.set_box_aspect((1, 1, 1)); a3.view_init(*view)
        a3.set_title("rotors on S³ (stereographic), ★ = q and −q", fontsize=10)

        ac = fig.add_subplot(len(picks), 4, 4 * r + 3)
        c = path @ q
        for j in range(path.shape[1]):
            ac.plot(t, c[:, j], color=PLUS if sign[j] > 0 else MINUS, lw=1.2, alpha=0.8 if ok[j] else 0.35)
        ac.axhline(1, color="gold", ls="--"); ac.axhline(-1, color="gold", ls="--"); ac.axhline(0, color="0.8", lw=0.8)
        ac.set_ylim(-1.08, 1.08); ac.set_xlabel("flow time t"); ac.set_ylabel("q(t) · q_gt  (+1 = q, −1 = −q)"); ac.grid(alpha=0.3)
        ac.set_title("which of the two rotors a sample reaches", fontsize=10)

        ak = fig.add_subplot(len(picks), 4, 4 * r + 4)
        for comp in range(4):
            line, = ak.plot(t, path[:, 0, comp], lw=1.6, label=COMPONENTS[comp])
            ak.axhline(q[comp], color=line.get_color(), ls=":", lw=1); ak.axhline(-q[comp], color=line.get_color(), ls="--", lw=0.8, alpha=0.6)
        ak.set_xlabel("flow time t"); ak.set_ylim(-1.05, 1.05); ak.grid(alpha=0.3); ak.legend(fontsize=7, ncol=4, loc="upper center")
        ak.set_title("sample 0: rotor components (dotted +q, dashed −q)", fontsize=10)
    fig.suptitle(f"{label}: the flow on rotors (unit quaternions); both q and −q are the ground truth", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); return _finish(fig, out)


def interactive_rotors(models, imgs, rots, picks, k, steps, seed, out, cls=None, errs=None):
    """Plotly HTML of the stereographic S^3 view: drag to rotate; hover a curve for flow time, alignment with q_gt and the rotation error; dropdown picks (model, image)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from pose3d.engine.flow_viz import CLASS_NAMES

    fig = make_subplots(rows=1, cols=2, column_widths=[0.28, 0.72], horizontal_spacing=0.01, specs=[[{"type": "xy"}, {"type": "scene"}]])
    groups, labels = [], []
    t = np.linspace(0, 1, steps + 1)
    for ml, model in models.items():
        for n, i in enumerate(picks):
            path = rotor_trajectories(model, imgs[i], k, steps, seed).astype(np.float64)
            q = gt_rotor(rots[i]); sign, ok = destinations(path, q); y = stereographic(path, stereo_frame(q, path))
            err = angle_deg(np.stack([rotor_to_matrix(torch.from_numpy(p).float(), model.algebra).numpy() for p in path]), rots[i])
            small = torch.nn.functional.interpolate(imgs[i][None].float(), size=(160, 160), mode="bilinear", align_corners=False)[0]
            start = len(fig.data)
            fig.add_trace(go.Image(z=(small.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8), hoverinfo="skip"), row=1, col=1)
            for j in range(path.shape[1]):
                col = "#1f77b4" if sign[j] > 0 else "#ff7f0e"
                fig.add_trace(go.Scatter3d(x=y[:, j, 0], y=y[:, j, 1], z=y[:, j, 2], mode="lines", showlegend=False, line=dict(width=5, color=col),
                                           opacity=0.9 if ok[j] else 0.4, customdata=np.stack([t, path[:, j] @ q, err[:, j]], 1),
                                           hovertemplate=f"sample {j} → {'+q' if sign[j] > 0 else '−q'}<br>t=%{{customdata[0]:.2f}}<br>q·q_gt=%{{customdata[1]:.2f}}<br>%{{customdata[2]:.1f}}° from the truth<extra></extra>"), row=1, col=2)
            fig.add_trace(go.Scatter3d(x=y[0, :, 0], y=y[0, :, 1], z=y[0, :, 2], mode="markers", showlegend=False, hoverinfo="skip",
                                       marker=dict(size=4, color="white", line=dict(color="gray", width=2))), row=1, col=2)
            fig.add_trace(go.Scatter3d(x=y[-1, :, 0], y=y[-1, :, 1], z=y[-1, :, 2], mode="markers", showlegend=False,
                                       marker=dict(size=5, color=np.where(sign > 0, "#1f77b4", "#ff7f0e"), line=dict(color=np.where(ok, "black", "red"), width=2)),
                                       customdata=err[-1], hovertemplate="final: %{customdata:.1f}° from the truth<extra></extra>"), row=1, col=2)
            fig.add_trace(go.Scatter3d(x=[1, -1], y=[0, 0], z=[0, 0], mode="markers+text", text=["q", "−q"], textposition="top center", showlegend=False,
                                       marker=dict(size=11, color="gold", symbol="diamond", line=dict(color="black", width=3))), row=1, col=2)
            groups.append((start, len(fig.data)))
            name = f" ({CLASS_NAMES[cls[i]]})" if cls is not None and 0 <= cls[i] < len(CLASS_NAMES) else ""
            e = f", medoid error {errs[n]:.1f}°" if errs is not None else ""
            labels.append(f"{ml}: val #{i}{name}{e}; {int(((sign > 0) & ok).sum())} → +q, {int(((sign < 0) & ok).sum())} → −q, {int((~ok).sum())} miss")
    for gi, (a0, b0) in enumerate(groups):
        for tr in fig.data[a0:b0]:
            tr.visible = gi == 0
    buttons = [dict(label=lab, method="update", args=[{"visible": [a0 <= m < b0 for m in range(len(fig.data))]}, {"title.text": lab}]) for (a0, b0), lab in zip(groups, labels)]
    ax = dict(visible=False, range=[-2, 2])
    fig.update_layout(title=dict(text=labels[0], x=0.0, xanchor="left", y=0.97), updatemenus=[dict(buttons=buttons, direction="down", x=0.0, y=1.0, xanchor="left", yanchor="bottom", pad=dict(b=6), showactive=True)],
                      scene=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="cube", camera=dict(eye=dict(x=1.5, y=1.5, z=0.9))), margin=dict(l=5, r=5, t=130, b=5), height=640)
    fig.update_xaxes(visible=False, row=1, col=1); fig.update_yaxes(visible=False, row=1, col=1)
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    return out

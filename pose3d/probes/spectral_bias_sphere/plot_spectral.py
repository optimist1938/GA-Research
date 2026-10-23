"""Heatmaps (Rahaman et al. Fig. 1 style) and learning-speed summary from the spectral-bias results.

usage: plot_spectral.py results_equal.json results_increasing.json out.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES = {"mlp": "MLP (6 layers, ~98k)", "mlp256": "MLP 6x256 (paper's, 330k)", "cgenn": "CGENN (~99k)", "gatr": "GATr (~98k)"}


def load(path):
    return json.load(open(path))


def first_step(steps, rho, thr):
    """First checkpoint at which the seed-mean recovered fraction reaches thr (nan if never)."""
    idx = np.where(rho >= thr)[0]
    return float(steps[idx[0]]) if len(idx) else float("nan")


def main(paths, out):
    fig, axes = plt.subplots(len(paths) * 2, 4, figsize=(17, 4.6 * len(paths) * 2 / 2 + 3), squeeze=False)
    summary = {}
    for r, path in enumerate(paths):
        res = load(path)
        amps = res["args"]["amps"]
        models = [m for m in NAMES if m in res and res[m]]
        for c, m in enumerate(models):
            runs = res[m]
            steps = np.array(runs[0]["steps"])
            rho = np.mean([np.array(x["rho"]) for x in runs], 0)             # (ckpt, L)
            ax = axes[2 * r][c]
            L = rho.shape[1]
            im = ax.imshow(np.clip(rho, 0, 1), aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1,
                           extent=[0.5, L + 0.5, 0, len(steps)])
            ax.set_yticks(np.arange(0, len(steps), 6) + 0.5)
            ax.set_yticklabels([str(int(s)) for s in steps[::6]], fontsize=7)
            ax.set_xlabel("target degree l"); ax.set_ylabel("training step")
            ax.set_title(f"{NAMES[m]} | {amps}", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046) if c == len(models) - 1 else None
            summary.setdefault(amps, {})[m] = {
                "step_to_50pct": [first_step(steps, rho[:, l], 0.5) for l in range(L)],
                "step_to_90pct": [first_step(steps, rho[:, l], 0.9) for l in range(L)],
                "final_rho": [float(x) for x in rho[-1]],
                "final_test_mse": [float(x["test_mse"][-1]) for x in runs],
                "params": runs[0]["params"],
            }
        ax = axes[2 * r + 1][0]
        for m in models:
            s50 = summary[amps][m]["step_to_90pct"]
            ax.plot(range(1, len(s50) + 1), s50, marker="o", label=NAMES[m])
        ax.set_yscale("log"); ax.set_xlabel("target degree l"); ax.set_ylabel("steps to recover 90 %")
        ax.set_title(f"learning speed | {amps}", fontsize=10); ax.legend(fontsize=7)
        ax = axes[2 * r + 1][1]
        for m in models:
            runs = res[m]
            steps = np.array(runs[0]["steps"])[1:]
            ax.plot(steps, np.mean([x["test_mse"][1:] for x in runs], 0), label=NAMES[m])
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("held-out MSE")
        ax.set_title(f"held-out MSE | {amps}", fontsize=10)
        for k in (2, 3):
            axes[2 * r + 1][k].axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    json.dump(summary, open(out.replace(".png", ".json"), "w"), indent=1)
    return summary


if __name__ == "__main__":
    s = main(sys.argv[1:-1], sys.argv[-1])
    for amps, d in s.items():
        print("==", amps)
        for m, v in d.items():
            print(f"{m:7s} params {v['params']:6d} | steps to 90 %: {[None if np.isnan(x) else int(x) for x in v['step_to_90pct']]} | final rho {[round(x, 2) for x in v['final_rho']]} | test mse {[round(x, 4) for x in v['final_test_mse']]}")

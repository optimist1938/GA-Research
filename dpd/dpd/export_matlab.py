"""Export a trained `gated_tcn` model to a MATLAB .mat for `matlab_model/dpd_gated_tcn.m`.

    python -m dpd.export_matlab runs/tcn16x4_model.pt matlab_model/dpd_tcn16x4.mat

The checkpoint comes from `python -m dpd --model gated_tcn ... --save_model`. The .mat holds one
struct `model` with every weight in double precision and in the layout `dpd_gated_tcn.m` indexes
(W1(:,:,j,i) is the out x in matrix of tap j of block i), plus a 4096-sample check vector
(`x_check`, `d_check`) so the MATLAB side can verify itself with `dpd_selfcheck.m`.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import scipy.io as sio
import torch
from torch import nn

from dpd.data import load, split_index
from dpd.features import carrier_taps, local_invariants
from dpd.models import GatedTCN


def load_checkpoint(path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)   # our own file: config dict + numpy scale
    if ck["kind"] != "gated_tcn":
        raise ValueError(f"only gated_tcn models can be exported, got {ck['kind']}")
    return ck


def build_net(ck: dict) -> GatedTCN:
    net_cfg, model_cfg = ck["config"]["net"], ck["config"]["model"]
    n_inv = 3 * (1 + 2 * net_cfg["max_lag_diff"])
    net = GatedTCN(n_inv, 2 * model_cfg["memory"] + 1, net_cfg["tcn_channels"], net_cfg["tcn_levels"])
    net.load_state_dict(ck["state_dict"])
    return net.eval()


@torch.no_grad()
def predict(ck: dict, net: GatedTCN, x_raw: np.ndarray) -> np.ndarray:
    """The correction d for raw band envelopes x_raw (3, N), exactly as training evaluated it."""
    scale = np.asarray(ck["scale"])[:, None]
    x = x_raw * scale
    inv = torch.from_numpy(local_invariants(x, ck["config"]["net"]["max_lag_diff"]))
    inv = (inv - ck["mu"]) / ck["sd"]
    taps = torch.from_numpy(carrier_taps(x, ck["config"]["model"]["memory"]))
    y = net(inv[None], taps[None])[0].numpy().T.astype(complex)
    return y / scale


def macs_per_sample(net: GatedTCN, n_inv: int, n_taps: int) -> int:
    """Real multiply-accumulates per output sample (all three bands)."""
    conv = sum(m.in_channels * m.out_channels * m.kernel_size[0] for m in net.modules() if isinstance(m, nn.Conv1d))
    features = 2 * n_inv          # each |x|^2 or Re/Im of x conj(x(k-j)): ~2 real MAC
    gating = 3 * n_taps * 4       # one complex multiply per band and carrier tap
    return conv + features + gating


def matlab_struct(ck: dict, net: GatedTCN) -> dict:
    sd = {k: v.detach().double().numpy() for k, v in net.state_dict().items()}
    net_cfg, model_cfg = ck["config"]["net"], ck["config"]["model"]
    levels, n_inv, n_taps = net_cfg["tcn_levels"], 3 * (1 + 2 * net_cfg["max_lag_diff"]), 2 * model_cfg["memory"] + 1
    return {
        "kind": "gated_tcn",
        "memory": float(model_cfg["memory"]),
        "max_lag_diff": float(net_cfg["max_lag_diff"]),
        "levels": float(levels),
        "channels": float(net_cfg["tcn_channels"]),
        "receptive": float(net.receptive),
        "n_params": float(ck["n_params"]),
        "macs_per_sample": float(macs_per_sample(net, n_inv, n_taps)),
        "scale": np.asarray(ck["scale"], dtype=float).reshape(3, 1),
        "mu": ck["mu"].double().numpy().reshape(-1, 1),
        "sd": ck["sd"].double().numpy().reshape(-1, 1),
        "W_in": sd["inp.weight"][:, :, 0],
        "b_in": sd["inp.bias"].reshape(-1, 1),
        "W1": np.stack([sd[f"blocks.{i}.1.weight"] for i in range(levels)], axis=-1),
        "b1": np.stack([sd[f"blocks.{i}.1.bias"] for i in range(levels)], axis=-1),
        "W2": np.stack([sd[f"blocks.{i}.3.weight"][:, :, 0] for i in range(levels)], axis=-1),
        "b2": np.stack([sd[f"blocks.{i}.3.bias"] for i in range(levels)], axis=-1),
        "W_out": sd["out.weight"][:, :, 0],
        "b_out": sd["out.bias"].reshape(-1, 1),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint")
    p.add_argument("out")
    p.add_argument("--data_path", default="data/GeoData_TB.mat")
    p.add_argument("--check_len", type=int, default=4096)
    a = p.parse_args()

    ck = load_checkpoint(a.checkpoint)
    net = build_net(ck)
    sig = load(a.data_path)
    _, test = split_index(sig.n, ck["config"]["run"]["train_frac"])

    # The saved predictions, if present, must match: same weights, same features, same normalization.
    pred_path = pathlib.Path(a.checkpoint).with_name(pathlib.Path(a.checkpoint).name.replace("_model.pt", "_pred.npy"))
    if pred_path.exists():
        rel = np.abs(predict(ck, net, sig.x) - np.load(pred_path)).max() / np.abs(sig.d).max()
        print(f"full-series prediction vs saved {pred_path.name}: max rel diff {rel:.2e}")

    start = test.start + 1000
    x_check = sig.x[:, start:start + a.check_len]
    model = matlab_struct(ck, net)
    model["x_check"] = x_check
    model["d_check"] = predict(ck, net, x_check)
    model["nmse_test_vs_d"] = np.array([ck["bands"][b]["nmse_vs_d"] for b in "ABC"]).reshape(3, 1)
    model["nmse_test_vs_x"] = np.array([ck["bands"][b]["nmse_vs_x"] for b in "ABC"]).reshape(3, 1)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(out, {"model": model}, do_compression=True)
    print(f"wrote {out}: {int(model['n_params'])} params, {int(model['macs_per_sample'])} MAC/sample, "
          f"latency +-{int(model['receptive'] + model['memory'])} samples")


if __name__ == "__main__":
    main()

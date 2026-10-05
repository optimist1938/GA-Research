# pose3d

**Best score**: Clifford Flow, ResNet-101, **9.46 MRE** on Pascal3D+ (W&B `6te3pvqa`) — now the default recipe in `pose3d/config.py`.

Setup and how to run: [`USAGE.md`](USAGE.md). Results: [`reports/`](reports/). Reading material:
[`awesome-reference/`](awesome-reference/README.md).

| Status | Idea | Description |
|---|---|---|
| Experimental | Endpoint (x1) parametrisation of Clifford Flow | `--flow_param x1 [--x1_loss tangent\|geodesic]` (branch `flow-x1-prediction`): the vector field predicts the remaining displacement log(rt~ r1), bounded by pi, instead of the constant velocity; the sampler divides by (1-t) and its last Euler step lands on the predicted endpoint (the x0-prediction of diffusion models). Run `clifford_flow_x1_warp_synth_ema_b64` on the best data recipe (warp + synth pack + EMA + batch 64): result pending, vs 7.88 deg median / 9.82 deg class-mean (`clifford_flow_cgenn_warp_synth_ema_b64`). |

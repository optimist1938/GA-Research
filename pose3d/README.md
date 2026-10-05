# pose3d

**Best score**: Clifford Flow with a GATr vector field, ResNet-101, trained on Image2Sphere's data (warp + synthetic renders) with EMA weights: **8.95°** mean of per-class median errors, **7.85°** median over all test images, on Pascal3D+ (`clifford_flow_gatr_warp_synth_ema_b64`, Kaggle kernel `syfry5suvzovvakmuj/clifford-gatr-ema-b64-rtx`) — now the default recipe in `pose3d/config.py`.

Setup and how to run: [`USAGE.md`](USAGE.md). Results: [`reports/`](reports/). Reading material:
[`awesome-reference/`](awesome-reference/README.md).

| Status | Idea | Description |
|---|---|---|

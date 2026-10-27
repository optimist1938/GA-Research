# pose3d

**Best score**: Clifford Flow with a GATr vector field, ResNet-101, trained on Image2Sphere's data (warp + synthetic renders) with EMA weights: **8.95°** mean of per-class median errors, **7.85°** median over all test images, on Pascal3D+ (`clifford_flow_gatr_warp_synth_ema_b64`, Kaggle kernel `syfry5suvzovvakmuj/clifford-gatr-ema-b64-rtx`) — now the default recipe in `pose3d/config.py`.

Setup and how to run: [`USAGE.md`](USAGE.md). Results: [`reports/`](reports/). Reading material:
[`awesome-reference/`](awesome-reference/README.md).

| Status | Idea | Description |
|---|---|---|
| done | Two-tower Clifford Flow: STF tensor track on SYMSOL I (10 %) | `--stf_track --stf_lmax 3\|6`: a second feature track of symmetric traceless Cartesian tensors (ranks 2..L, the degree-l irreps that multivectors cannot hold; CliffordSTF, arXiv 2606.29584) built from the condition vectors and injected into the GATr vector field as invariants of the current rotor, against the unchanged reference model under the `symsol1_10pct_*_i2s_adamw` protocol, 40 epochs. The reference flow never leaves the zero-velocity stall (median 47.3 deg, flat for 40 epochs); both STF arms leave it after ~20 epochs and end at **acc@15 0.046** (L = 3) / 0.037 (L = 6), median 43.7 / 44.9 deg (32 samples). Small in absolute terms (Image2Sphere: acc@15 1.00), but a qualitative difference from the control. Next: `--symsol_nearest_coupling` for both arms. [report](reports/symsol_stf/README.md) |
| done | Image2Sphere baseline on SYMSOL I (10 %) | `--model i2s_real` under the `symsol1_10pct_*` protocol: **3.2 deg** mean of per-shape nearest-equivalent medians (AdamW/cosine; 4.5 deg with the paper's SGD recipe), log-likelihood 3.48, against 19.9 deg / 3.24 for the best flow run. The paper itself gives no rotation error on SYMSOL. [report](reports/i2s_symsol1_10pct.md) |

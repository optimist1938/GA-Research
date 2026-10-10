"""analyze_so2.py: where does c4lift lose to the control? Per-sample errors at 0 deg (same test order)."""
import numpy as np

S = "/tmp/claude-1000/-home-bebra-GA-Research/79577114-4a69-4d0d-932d-b82986259792/scratchpad"
C = ["aeroplane", "bicycle", "boat", "bottle", "bus", "car", "chair", "diningtable", "motorbike", "sofa", "train", "tvmonitor"]
ctrl = np.load(f"{S}/out_so2-ctrl-pooled-framech/clifford_flow_gatr_pooled_framech_e60_rot_noprior_per_sample.npz")
c4 = np.load(f"{S}/out_so2-c4lift-framech/clifford_flow_gatr_c4lift_framech_e60_rot_noprior_per_sample.npz")
c4p = np.load(f"{S}/out_so2-c4lift-framech/clifford_flow_gatr_c4lift_framech_e60_rot_prior_per_sample.npz")
e_c, e_l, e_lp, cls = ctrl["err_0"], c4["err_0"], c4p["err_0"], ctrl["cls"]
assert (cls == c4["cls"]).all()
print("n", len(cls))
print("exact C4: max |err_90 - err_0| =", float(np.abs(c4["err_90"] - c4["err_0"]).max()),
      " |err_180 - err_0| =", float(np.abs(c4["err_180"] - c4["err_0"]).max()))
bins = [0, 5, 10, 15, 30, 60, 120, 181]
for name, e in (("ctrl", e_c), ("c4lift", e_l), ("c4lift+prior", e_lp)):
    h = np.histogram(e, bins)[0] / len(e)
    print(f"{name:13s} " + "  ".join(f"[{bins[i]},{bins[i + 1]}):{h[i]:.3f}" for i in range(len(h))))
# Large errors: are they ~90 / ~180 (turn flips) or spread out?
for name, e in (("ctrl", e_c), ("c4lift", e_l)):
    big = e[e > 60]
    print(f"{name:7s} errors > 60 deg: {len(big)}  near 90 (75-105): {int(((big > 75) & (big < 105)).sum())}"
          f"  near 180 (>150): {int((big > 150).sum())}")
# paired: images good in ctrl (<15) but bad in c4lift (>30), and vice versa
a = (e_c < 15) & (e_l > 30)
b = (e_l < 15) & (e_c > 30)
print(f"ctrl good / c4lift bad: {int(a.sum())}   c4lift good / ctrl bad: {int(b.sum())}")
print("per class  n  med_ctrl med_c4 med_c4prior  frac>30 ctrl/c4  ctrl-good&c4-bad")
for i, n in enumerate(C):
    m = cls == i
    print(f"{n:12s} {m.sum():4d} {np.median(e_c[m]):7.2f} {np.median(e_l[m]):7.2f} {np.median(e_lp[m]):7.2f}"
          f"    {(e_c[m] > 30).mean():.2f}/{(e_l[m] > 30).mean():.2f}    {int(a[m].sum())}")
# diningtable: errors of the c4lift failures
m = (cls == 7) & (e_l > 30)
print("diningtable c4lift errors > 30:", np.round(np.sort(e_l[m]), 1))
print("same images, ctrl errors     :", np.round(e_c[m][np.argsort(e_l[m])], 1))

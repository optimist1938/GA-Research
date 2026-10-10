"""compare_so2.py: final metrics (per class) and the turned-test results of the two Stage-1 arms."""
import json
import re

S = "/tmp/claude-1000/-home-bebra-GA-Research/79577114-4a69-4d0d-932d-b82986259792/scratchpad"
CLASSES = ["aeroplane", "bicycle", "boat", "bottle", "bus", "car", "chair", "diningtable", "motorbike", "sofa",
           "train", "tvmonitor"]
arms = {"ctrl": "so2-ctrl-pooled-framech", "c4lift": "so2-c4lift-framech"}
final, rot = {}, {}
for a, k in arms.items():
    raw = open(f"{S}/follow_{k}.txt").read()
    m = re.search(r"^WANDB_SYNC_FINAL (\{.*\})\s*$", raw, re.M) or re.search(
        r'^WANDB_SYNC (\{.*"final": true.*\})\s*$', raw, re.M)
    final[a] = json.loads(m.group(1))["metrics"] if m else {}
    rot[a] = {mm.group(1): json.loads(mm.group(2)) for mm in re.finditer(r"EVAL_ROTATIONS (\S+) (\{.*)$", raw, re.M)}
    if re.search(r"Traceback", raw):
        print(a, "has a Traceback:", re.findall(r"^.*Error.*$", raw, re.M)[-2:])

print(f"{'':22s} {'ctrl':>8s} {'c4lift':>8s} {'diff':>7s}")
for key in ("final_median_rotation_error", "final_class_mean_median_error", "final_acc@15", "final_acc@30",
            "final_median_rotation_error_raw", "final_class_mean_median_error_raw"):
    c, l = final["ctrl"].get(key, float("nan")), final["c4lift"].get(key, float("nan"))
    print(f"{key.replace('final_', ''):22s} {c:8.3f} {l:8.3f} {l - c:+7.3f}")
print()
for i, name in enumerate(CLASSES):
    c, l = final["ctrl"].get(f"final_median_error_class{i}"), final["c4lift"].get(f"final_median_error_class{i}")
    ca, la = final["ctrl"].get(f"final_acc@30_class{i}"), final["c4lift"].get(f"final_acc@30_class{i}")
    if c is not None and l is not None:
        print(f"{name:12s} median {c:6.2f} -> {l:6.2f} ({l - c:+5.2f})   acc@30 {ca:.3f} -> {la:.3f}")
print()
for a in arms:
    for name, d in rot[a].items():
        print(f"== {a} {name}", d.get("roll_prior", ""))
        for ang, row in d["angles"].items():
            print(f"   {ang:>4} deg: median {row['median']:7.2f}  cls-mean {row['class_mean_median']:7.2f}  "
                  f"acc15 {row['acc15']:.3f}  acc30 {row['acc30']:.3f}")

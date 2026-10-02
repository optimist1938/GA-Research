# GA-Research

Geometric-algebra (Clifford) research experiments, one folder per experiment family.

| Folder | Experiment | Status |
|---|---|---|
| [`pose3d/`](pose3d/) | Image-conditioned 3D pose (SO(3)) estimation: Clifford Flow vs Image2Sphere, IPDF, matrix Fisher | active |
| [`dpd/`](dpd/) | Huawei multi-band digital predistortion: U(1)³-equivariant models vs LUT | active |

Each experiment folder is a self-contained Poetry project. `main` always holds the reference recipe
of every experiment; experiments in progress live on branches until they are proven.

## Inside an experiment folder

| Path | What it is |
|---|---|
| `README.md` | the **idea board**: a table of experiment ideas. Anyone adds an idea, a developer claims it, and the row is updated with the result. |
| `reports/` | one Typst (`.typ`) report per finished experiment. |
| `awesome-reference/` | articles, code and websites worth reading, one line each. |
| `config.py` (in the package) | the reference recipe and the feature flags. |

## Working on `main`

* **Adopted change** (measured better): merged with its flag switched **on** in the experiment's
  `config.py`, or its value made the new default.
* **Not yet proven**: merged with its flag switched **off**, so it stays one `--flag` away but does
  not change the reference recipe.
* **Turned out worse**: not merged. It is written up in a report so nobody repeats it,
  and the branch stays available.

## Adding an experiment family

Create a sibling folder next to `pose3d/` with its own `pyproject.toml`, an idea-board `README.md`,
`reports/` and `awesome-reference/`, and a `config.py` whose defaults are that
experiment's reference recipe. Add a row to the table above.

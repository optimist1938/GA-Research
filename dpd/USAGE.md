# dpd: usage

```bash
cd dpd
poetry install
mkdir -p data && unzip ../GeoData_TB.zip -d data   # Huawei data, git-ignored: never commit or upload it
poetry run pytest tests                           # U(1)^3 equivariance of every model
poetry run python -m dpd                          # reference recipe: lut_ls
poetry run python -m dpd --model gated --epochs 100
poetry run python -m dpd --help                   # every config field is a flag
```

Each run prints per-band test NMSE vs d and vs x, and writes `runs/<run_name>.json` plus
`runs/<run_name>_psd_{A,B,C}.png` (Welch PSD of x, d, the model error and Huawei's eRef).
Everything runs on CPU: `lut_ls` ~5 min, `gated` 100 epochs ~4 min on 22 cores.

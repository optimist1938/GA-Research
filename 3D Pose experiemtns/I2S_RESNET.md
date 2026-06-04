# I2S_ResNet

`I2S_ResNet` is implemented in `src/model.py` and registered in `src/main.py` + `src/config.py`.

## Shape flow

1. Input image: `[B, 3, H, W]`
2. Frozen ResNet50 feature map: `[B, 2048, H_backbone, W_backbone]`
3. Conv adapter + adaptive pool: `[B, 8, 8, 8]`
4. Flatten to tokens: `[B, 64, 8]`
5. Add learned positional embeddings: `[B, 64, 8]`
6. GA blocks (`TralaleroTralala`) consume the token sequence.

## Output modes

`--model i2s_resnet` supports both existing training/evaluation modes:

- Direct rotation matrix prediction (`[B, 3, 3]`):
  - Use `--loss mse`
  - Optional explicit override: `--i2s_resnet_output_mode rotation_matrix`
- Fourier/PDF decomposition logits on the SO(3) grid:
  - Use `--loss prob`
  - Optional explicit override: `--i2s_resnet_output_mode fourier`

With `--i2s_resnet_output_mode auto` (default), mode is selected from `--loss` in `instantiate()`.

## Example runs

### Rotation matrix mode

```bash
python -m src.main \
  --path_to_datasets /path/to/data \
  --model i2s_resnet \
  --loss mse
```

### Fourier/PDF mode

```bash
python -m src.main \
  --path_to_datasets /path/to/data \
  --model i2s_resnet \
  --loss prob
```

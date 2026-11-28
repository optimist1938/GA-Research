import json, glob, base64
geo = json.load(open(glob.glob('nb_geo/*.ipynb')[0]))
gatr_cell = ''.join(geo['cells'][2]['source'])
gatr_line = [l for l in gatr_cell.split('\n') if l.startswith('GATR_WHEELS')][0]
deps_src = '\n'.join(l for l in gatr_cell.split('\n') if not l.startswith('GATR_WHEELS'))
deps_src = deps_src.replace(
    "shutil.copy(os.path.join(DEPS_DIR, 'resnet101-cd907fc2.pth'), os.path.join(ckpt_dir, 'resnet101-cd907fc2.pth'))",
    "for w in ('resnet50-11ad3fa6.pth', 'resnet101-cd907fc2.pth'):   # ImageNet weights (--encoder resnet50 loads the first)\n"
    "    shutil.copy(os.path.join(DEPS_DIR, w), os.path.join(ckpt_dir, w))")
assert 'resnet50-11ad3fa6.pth' in deps_src
code_b64 = base64.b64encode(open('build/code.tgz', 'rb').read()).decode()

RUN = 'chile3d_pyr7_dit_geobias_grid14_nside8_s100_e180_l4x4'
ARGS = ['--model', 'clifford_flow', '--encoder', 'resnet50', '--dataset', 'symsol', '--vector_field', 'dit_geobias',
        '--cond_tokens', 'spherecloud', '--pose_tokens', 'frame_pin', '--condition_head', 'none',
        '--dircloud_scalars', '64', '--dircloud_nside', '8', '--dircloud_grid', '14',
        '--gatr_mv_channels', '8', '--gatr_s_channels', '32', '--no-gatr_zero_readout', '--gatr_heads', '8',
        '--dit_dim', '21', '--dit_heads', '8', '--dit_blocks', '4', '--dit_mlp_ratio', '4',
        '--symsol_set', '14', '--symsol_train_views', '4500', '--symsol_test_views', '1000',
        '--no-use_warp', '--no-use_synth', '--num_workers', '8', '--no-ema', '--fixed_val_noise', '--batch_size', '64',
        '--optimizer', 'adamw', '--lr', '1e-4', '--lr_scaling', 'sqrt', '--lr_schedule', 'step', '--lr_step_gamma', '1.0',
        '--warmup_epochs', '0', '--n_epochs', '180', '--no-symsol_likelihood', '--ddp', '--num_gpus', '4', '--sync_bn',
        '--seed', '0']

c0 = f'''# 3D SYMSOL-CHILE 7-GON (pyr7, --symsol_set 14), DiT + explicit geometric attention bias (dit_geobias), seed 0, 4xL4.
# Question (requested by a human, 2026-11-28): does DiT-geobias, trained with the recipe under which cosine-attention GATr
# learned the bottom-view 7-gon (14x14 map, 368 sphere points, constant lr), get past the ~13 deg plateau that GATr-cosdir
# reached on full 3D pyr7 (cheesewaffler/chile3d-pyr7-cosdir-grid14-s100-e180-seed0-t4x2: 12.7 deg; seed 1: 13.1 deg)?
# Identical to that GATr run except: vector field dit_geobias (width 21, 8 heads, 4 blocks, ~97k field parameters as in
# stefanychaik/symsol-dit-spherecloud-geobias-h8-step30-l4x4) with a FIXED bias gain s = 100 (GATr's s_h started at 100 and
# was learnable), 4xL4 instead of 2xT4 (same global batch 64), num_workers 8.
# Code: chagrygoris/GA-Research@claude/symsol-symsol2-differences-2dukos (SYMSOL-CHILE renderer, resume) + the pyr7 kernel's
# --dircloud_nside / --dircloud_grid / --symsol_set 14 patches + dit_geobias.py from the geobias kernel (gain from
# POSE3D_GEOBIAS_SCALE), shipped below as CODE_B64. ETA ~2-2.5 h on 4xL4 (approved by a human before launch).
RUN_NAME = {RUN!r}
ARGS = {ARGS!r}
GEOBIAS_SCALE = '100'
WATCHDOG_MINUTES = 330   # ~2x the estimate; a stop keeps the per-epoch resume checkpoint
'''

c1 = f'''CODE_B64 = {code_b64!r}
import base64, glob, io, os, shutil, subprocess, sys, tarfile
if os.path.exists('/kaggle/working/GA-Research'):
    shutil.rmtree('/kaggle/working/GA-Research')
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_B64)), mode='r:gz') as t:
    t.extractall('/kaggle/working')
P = '/kaggle/working/GA-Research/pose3d'
for path, marker in {{'pose3d/models/dit_geobias.py': 'POSE3D_GEOBIAS_SCALE', 'pose3d/datasets/symsol_chile.py': 'pyr7',
                     'pose3d/datasets/symsol.py': '14: "symsol_chile"', 'pose3d/config.py': 'dircloud_grid: int = 7',
                     'pose3d/engine/resume.py': 'RESUMABLE_EXIT', 'pose3d/models/dit_flow.py': 'attention=None'}}.items():
    assert marker in open(os.path.join(P, path)).read(), f'stale code: {{marker!r}} not in {{path}}'
print('code unpacked', flush=True)
'''

c2 = gatr_line + '\n' + deps_src + '\n'

c3 = '''import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print('Accelerators:', names, flush=True)
assert len(names) == 4 and all('L4' in n for n in names), names
os.chdir(P)
DATA = '/kaggle/working/data'
# Render SYMSOL-CHILE pyr7 (seeded: the same images as the GATr pyr7 kernels).
if not os.path.exists(f'{DATA}/symsol_chile/train/rotations.npz'):
    subprocess.run([sys.executable, '-u', '-m', 'pose3d.datasets.symsol_chile', f'--root={DATA}', '--train', '4500',
                    '--test', '1000', '--device', 'cuda', '--view', 'random', '--shapes', 'pyr7'], check=True)
print('SYMSOL-CHILE root:', DATA, flush=True)
r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_symsol_chile.py',
                    'tests/test_resume.py', 'tests/test_spherecloud.py', 'tests/test_dircloud.py'],
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4'}, capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-2000:], flush=True)
assert r.returncode == 0, 'CPU preflight failed'
'''

c4 = '''# GPU preflight: the exact model of the run, before any training.
os.environ['POSE3D_GEOBIAS_SCALE'] = GEOBIAS_SCALE
os.environ['SYMSOL_SHARED_CACHE'] = '/kaggle/temp/symsol-shared-pixels'
from pose3d.config import parse_args
from pose3d.models import build_model
from pose3d.train import make_algebra
cfg = parse_args([f'--path_to_datasets={DATA}', '--platform=kaggle', f'--run_name={RUN_NAME}', *ARGS, '--no-pretrained_backbone'])
assert cfg.run.dataset == 'symsol' and cfg.data.symsol_set == 14
model = build_model(cfg, make_algebra(cfg.model.algebra_dim)).cuda().eval()
assert type(model.vector_field).__name__ == 'DiTGeometricBiasField', type(model.vector_field)
field_params = sum(p.numel() for p in model.vector_field.parameters())
assert 94_000 <= field_params <= 98_000, field_params
assert all(float(b.attn.scale) == float(GEOBIAS_SCALE) for b in model.vector_field.blocks)
with torch.no_grad():
    img = torch.rand(2, 3, 224, 224, device='cuda')
    cond = model.condition(img)
    print('condition tokens:', tuple(cond.shape), flush=True)
    assert cond.shape[1] == 369, cond.shape     # 368 HEALPix nside-8 points + the optical-axis token
    assert torch.isfinite(model.predict(img, steps=2)).all()
print('trainable params:', sum(p.numel() for p in model.parameters() if p.requires_grad), '| field:', field_params, flush=True)
del model, img, cond
torch.cuda.empty_cache()
print('DiT geobias pyr7 GPU preflight passed', flush=True)
'''

c5 = '''import threading, time
name = RUN_NAME
resume = sorted(glob.glob(f'/kaggle/input/**/{name}_resume.pth', recursive=True))
resume_args = []
if resume:
    shutil.copy(resume[0], f'/kaggle/working/{name}_resume.pth')
    resume_args = [f'--resume=/kaggle/working/{name}_resume.pth']
    print('resuming from', resume[0], flush=True)
cmd = [sys.executable, '-u', '-m', 'pose3d', f'--path_to_datasets={DATA}', '--platform=kaggle', f'--run_name={name}',
       '--resume_dir=/kaggle/working', *resume_args, *ARGS]
print(' '.join(cmd), flush=True)
print(f'watchdog {WATCHDOG_MINUTES} min, geobias s = {GEOBIAS_SCALE}', flush=True)
with open(f'/kaggle/working/{name}.log', 'w') as fh:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env={**os.environ})
    def _watchdog():
        time.sleep(WATCHDOG_MINUTES * 60)
        if proc.poll() is None:
            print(f'WATCHDOG: stopping the run after {WATCHDOG_MINUTES} minutes (resumable)', flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
    threading.Thread(target=_watchdog, daemon=True).start()
    for line in proc.stdout:
        print(line, end='', flush=True)
        fh.write(line); fh.flush()
    rc = proc.wait()
if rc == 75:
    print(f'STOPPED, RESUMABLE: /kaggle/working/{name}_resume.pth; relaunch with this kernel as --kernel-source', flush=True)
else:
    assert rc == 0, f'pose3d exited with code {rc}'
'''

cells = [c0, c1, c2, c3, c4, c5]
nb = {'cells': [{'cell_type': 'code', 'execution_count': None, 'id': f'c{i}', 'metadata': {}, 'outputs': [],
                 'source': c.splitlines(keepends=True)} for i, c in enumerate(cells)],
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 5}
json.dump(nb, open('launch/chile3d-pyr7-dit-geobias-grid14-s100-e180-l4x4.ipynb', 'w'))
for i, c in enumerate(cells):
    compile(c, f'cell{i}', 'exec')
print('notebook written, cells compile')

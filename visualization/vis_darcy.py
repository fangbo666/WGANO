"""
Darcy Flow Visualization: True Solution | SAOT Error | WGANO Error
Usage:
    python vis_darcy.py --data_path /path/to/darcy --sample_idx 0
"""
import os, sys, argparse
import numpy as np
import scipy.io as scio
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Fix fonts for AAAI 2027 (avoid Identity-H encoding) ─────────────────────
plt.rcParams['pdf.fonttype'] = 42       # TrueType (avoids Type 3)
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'  # Helvetica recommended by AAAI
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.serif'] = ['Times', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'     # Computer Modern for math (allowed)
plt.rcParams['text.usetex'] = False         # Avoid LaTeX fonts

# ── path setup ──────────────────────────────────────────────────────────────
VIS_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(VIS_DIR, '..'))          # C:\WGANO
SAOT_ROOT = os.path.join(REPO_ROOT, 'baseline', 'SAOT_NO')

sys.path.insert(0, SAOT_ROOT)
sys.path.insert(0, os.path.join(SAOT_ROOT, 'model'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'WGANO', 'n1'))

from utils.normalizer import UnitTransformer
from model_dict import get_model as saot_get_model

# ── argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser('Darcy Visualization')
parser.add_argument('--data_path',  type=str,
                    default='/root/autodl-tmp/datasets/darcy')
parser.add_argument('--saot_ckpt',  type=str,
                    default=os.path.join(REPO_ROOT, 'experiment', 'saot_repro',
                                         'darcy', 'checkpoints', 'darcy_SAOT_best.pt'))
parser.add_argument('--wgano_ckpt', type=str,
                    default=os.path.join(REPO_ROOT, 'experiment', 'table1',
                                         'f1darcy', 'checkpoints', 'f1darcy_n1v6.pt'))
parser.add_argument('--sample_idx', type=int, default=0,
                    help='which test sample to visualize (0-indexed)')
parser.add_argument('--out_dir',    type=str, default=VIS_DIR)
parser.add_argument('--gpu',        type=str, default='0')
parser.add_argument('--downsample', type=int, default=5)
parser.add_argument('--ntest',      type=int, default=200)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cpu')  # Force CPU for darcy (normalizer has device mismatch issue)
print(f'Using device: {device}')

# ── data ─────────────────────────────────────────────────────────────────────
r  = args.downsample
s  = int(((421 - 1) / r) + 1)   # 85
nt = args.ntest

train_path = os.path.join(args.data_path, 'piececonst_r421_N1024_smooth1.mat')
test_path  = os.path.join(args.data_path, 'piececonst_r421_N1024_smooth2.mat')

print('Loading data …')
td = scio.loadmat(train_path)
x_tr = torch.from_numpy(td['coeff'][:1000, ::r, ::r][:, :s, :s]).float()
y_tr = torch.from_numpy(td['sol']  [:1000, ::r, ::r][:, :s, :s]).float()

vd = scio.loadmat(test_path)
x_te = torch.from_numpy(vd['coeff'][:nt, ::r, ::r][:, :s, :s]).float()
y_te = torch.from_numpy(vd['sol']  [:nt, ::r, ::r][:, :s, :s]).float()

# normalizers built on train set (flat)
x_norm = UnitTransformer(x_tr.reshape(1000, -1))
y_norm = UnitTransformer(y_tr.reshape(1000, -1))
x_norm.to(device); y_norm.to(device)

x_te_norm = x_norm.encode(x_te.reshape(nt, -1)).reshape(nt, s, s, 1)
y_true     = y_te.reshape(nt, s, s)                       # raw (unnormalized) ground truth

# grid positions
gx, gy = torch.meshgrid(torch.linspace(0, 1, s),
                         torch.linspace(0, 1, s), indexing='ij')
pos = torch.stack([gx, gy], dim=-1).unsqueeze(0).to(device)

# ── SAOT model ───────────────────────────────────────────────────────────────
class _SAOTArgs:
    model       = 'SAOT_Structured_Mesh_2D'
    n_hidden    = 128
    n_layers    = 8
    n_heads     = 8
    mlp_ratio   = 2
    dropout     = 0.0
    unified_pos = 1
    ref         = 8
    slice_num   = 64
    fun_dim     = 1
    out_dim     = 1

print('Loading SAOT …')
_saot_mod = saot_get_model(_SAOTArgs())   # returns the module (SAOT_Structured_Mesh_2D)
_saot_cls = _saot_mod.Model if hasattr(_saot_mod, 'Model') else _saot_mod
saot_model = _saot_cls(
    space_dim=2,
    n_layers=_SAOTArgs.n_layers,
    n_hidden=_SAOTArgs.n_hidden,
    dropout=_SAOTArgs.dropout,
    n_head=_SAOTArgs.n_heads,
    Time_Input=False,
    mlp_ratio=_SAOTArgs.mlp_ratio,
    fun_dim=1, out_dim=1,
    slice_num=_SAOTArgs.slice_num,
    ref=_SAOTArgs.ref,
    unified_pos=_SAOTArgs.unified_pos,
    H=s, W=s,
    is_filter=True,
).to(device)
saot_model.load_state_dict(
    torch.load(args.saot_ckpt, map_location=device), strict=False)
saot_model.eval()
print(f'  SAOT params: {sum(p.numel() for p in saot_model.parameters()):,}')

# ── WGANO model ──────────────────────────────────────────────────────────────
print('Loading WGANO …')
# import the N1 model via sys.path to avoid circular import issues
n1_dir = os.path.join(REPO_ROOT, 'WGANO', 'n1')
if n1_dir not in sys.path:
    sys.path.insert(0, n1_dir)
if 'model_n1qk' in sys.modules:
    del sys.modules['model_n1qk']
if 'model_n1' in sys.modules:
    del sys.modules['model_n1']
_wgano_imported = False
for _name in ('model_n1', 'model_n1qk'):
    _path = os.path.join(n1_dir, _name + '.py')
    if os.path.exists(_path):
        import importlib
        _mod = importlib.import_module(_name)
        N1Model = _mod.Model
        print(f'  Imported WGANO from {_name}.py')
        _wgano_imported = True
        break
if not _wgano_imported:
    raise RuntimeError(f'No model_n1qk.py or model_n1.py found in {n1_dir}')

wgano_model = N1Model(
    space_dim=2, n_layers=9, n_hidden=156,
    n_head=6, mlp_ratio=2, fun_dim=1, out_dim=1,
    ref=8, unified_pos=1, H=s, W=s,
).to(device)
wgano_model.load_state_dict(
    torch.load(args.wgano_ckpt, map_location=device), strict=False)
wgano_model.eval()
print(f'  WGANO params: {sum(p.numel() for p in wgano_model.parameters()):,}')

# ── inference on one sample ───────────────────────────────────────────────────
idx = args.sample_idx
print(f'Inferring sample {idx} …')

fx_i = x_te_norm[idx:idx+1].to(device)   # [1,s,s,1]
pos_i = pos                                # [1,s,s,2]
y_true_i = y_true[idx].numpy()            # [s,s]

with torch.no_grad():
    # SAOT
    out_saot = saot_model(pos_i, fx=fx_i).squeeze(-1)   # [1,s,s]
    out_saot = y_norm.decode(out_saot).squeeze(0).cpu().numpy()   # [s,s]

    # WGANO
    out_wgano = wgano_model(pos_i, fx=fx_i)[0].squeeze(-1)
    out_wgano = y_norm.decode(out_wgano).squeeze(0).cpu().numpy()

err_saot  = out_saot  - y_true_i
err_wgano = out_wgano - y_true_i

# ── plotting ──────────────────────────────────────────────────────────────────
print('Plotting …')
fig = plt.figure(figsize=(15, 4))
gs  = GridSpec(1, 3, figure=fig, wspace=0.35)

vmin_sol = y_true_i.min(); vmax_sol = y_true_i.max()
# Both error maps share the same vmax for unified magnitude
all_err = np.concatenate([np.abs(err_saot).flatten(), np.abs(err_wgano).flatten()])
err_vmax_abs = np.percentile(all_err, 99)
err_magnitude = int(np.floor(np.log10(err_vmax_abs)))
err_scale = 10**err_magnitude
err_vmax = err_vmax_abs / err_scale  # 归一化到数量级范围

def _add_panel(ax, data, title, vmin, vmax, cmap='viridis', is_error=False):
    im = ax.imshow(data, origin='lower', cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    if is_error:
        # Colorbar with real error values and magnitude label
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        ticks=[-vmax, 0, vmax])
        cb.ax.set_yticklabels([f'{-vmax:.1f}', '0.0', f'{vmax:.1f}'], fontsize=20)
        cb.ax.set_title(f'1e{err_magnitude:d}', fontsize=20, pad=8)
        cb.ax.tick_params(labelsize=20, size=0)
        # Nudge top and bottom labels inward
        fig.canvas.draw()
        labels = cb.ax.yaxis.get_majorticklabels()
        from matplotlib.transforms import ScaledTranslation
        offset_up = ScaledTranslation(0, 10/72., fig.dpi_scale_trans)   # 10pt up
        offset_down = ScaledTranslation(0, -10/72., fig.dpi_scale_trans) # 10pt down
        labels[0].set_transform(labels[0].get_transform() + offset_up)
        labels[-1].set_transform(labels[-1].get_transform() + offset_down)
    else:
        # True solution: actual value ticks with magnitude on top
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        ticks=[vmin, (vmin+vmax)/2, vmax])
        mag_true = int(np.floor(np.log10(vmax)))
        scale = 10**(-mag_true)
        cb.ax.set_yticklabels([f'{vmin*scale:.1f}', f'{(vmin+vmax)/2*scale:.1f}', f'{vmax*scale:.1f}'], fontsize=20)
        cb.ax.set_title(f'1e{mag_true:d}', fontsize=20, pad=8)
        cb.ax.tick_params(labelsize=20, size=0)
        # Nudge top and bottom labels inward
        fig.canvas.draw()
        labels = cb.ax.yaxis.get_majorticklabels()
        from matplotlib.transforms import ScaledTranslation
        offset_up = ScaledTranslation(0, 10/72., fig.dpi_scale_trans)   # 10pt up
        offset_down = ScaledTranslation(0, -10/72., fig.dpi_scale_trans) # 10pt down
        labels[0].set_transform(labels[0].get_transform() + offset_up)
        labels[-1].set_transform(labels[-1].get_transform() + offset_down)
    return im

_add_panel(fig.add_subplot(gs[0]), y_true_i,
           'True Solution', vmin_sol, vmax_sol, cmap='RdBu_r', is_error=False)
_add_panel(fig.add_subplot(gs[1]), err_saot / err_scale,
           'SAOT Error', -err_vmax, err_vmax, cmap='RdBu_r', is_error=True)
_add_panel(fig.add_subplot(gs[2]), err_wgano / err_scale,
           'WGANO Error', -err_vmax, err_vmax, cmap='RdBu_r', is_error=True)

plt.tight_layout()

out_path = os.path.join(args.out_dir, 'figure2.pdf')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved → {out_path}')
plt.close(fig)

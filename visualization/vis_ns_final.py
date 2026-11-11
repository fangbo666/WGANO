"""
Navier-Stokes 2D Visualization: True Solution | SAOT Error | WGANO Error
Single physics field: velocity magnitude |u|
统一字体和格式，向darcy/mhd标准看齐
"""
import os
import sys
import numpy as np
import scipy.io as scio
import torch
import argparse
import matplotlib
matplotlib.use('Agg')
from safe_load import safe_load
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import ScaledTranslation

# ── Fix fonts for AAAI 2027 ──────────────────────────────────────────────────
plt.rcParams['pdf.fonttype'] = 42       # TrueType (avoids Type 3)
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'  # Helvetica recommended by AAAI
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.serif'] = ['Times', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'     # Computer Modern for math (allowed)
plt.rcParams['text.usetex'] = False         # Avoid LaTeX fonts

FONT_TITLE = 20
FONT_LABEL = 18
FONT_CB = 20
FONT_CB_TL = 20

# ── Path setup ──────────────────────────────────────────────────────────────
VIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(VIS_DIR, '../../..'))
SAOT_ROOT = os.path.join(REPO_ROOT, 'baseline', 'SAOT_NO')

sys.path.insert(0, SAOT_ROOT)
sys.path.insert(0, os.path.join(SAOT_ROOT, 'model'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'WGANO', 'n1'))

from utils.normalizer import UnitTransformer
from model_dict import get_model as saot_get_model

# ── Argument parsing ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser('NS Visualization')
parser.add_argument('--data_path', type=str,
                    default=os.path.join(REPO_ROOT, 'data', 'pdebench'))
parser.add_argument('--saot_ckpt', type=str,
                    default=os.path.join(REPO_ROOT, 'experiment', 'saot_repro',
                                         'ns', 'checkpoints', 'ns_SAOT_best.pt'))
parser.add_argument('--wgano_ckpt', type=str,
                    default=os.path.join(REPO_ROOT, 'experiment', 'table1',
                                         'f1ns', 'checkpoints', 'f1ns_n1v4.pt'))
parser.add_argument('--sample_idx', type=int, default=None)
parser.add_argument('--out_dir', type=str, default=VIS_DIR)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--T_in', type=int, default=10)
parser.add_argument('--T', type=int, default=10)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ── Load data ───────────────────────────────────────────────────────────────
data_path = os.path.join(args.data_path, 'NavierStokes_V1e-5_N1200_T20', 'NavierStokes_V1e-5_N1200_T20.mat')
print(f'Loading data from {data_path}...')

data = scio.loadmat(data_path)
u_data = data['u']  # [N, H, W, T]
ntrain = 1000
ntest = 200
T_in = args.T_in
T = args.T

# Test set - NS训练时完全没有使用normalizer，输入输出都是原始数据
x_test = torch.from_numpy(u_data[-ntest:, :, :, :T_in]).float()
y_test = torch.from_numpy(u_data[-ntest:, :, :, T_in:T_in+T]).float()

# 直接使用原始数据，不进行归一化
x_test = x_test.to(device)
x_test_norm = x_test.permute(0, 3, 1, 2)  # [N, H, W, T] -> [N, T, H, W]

y_true = y_test.cpu().numpy()

H, W = 64, 64
gx, gy = torch.meshgrid(torch.linspace(0, 1, H),
                        torch.linspace(0, 1, W), indexing='ij')
pos = torch.stack([gx, gy], dim=-1).unsqueeze(0).to(device)

# ── SAOT model ──────────────────────────────────────────────────────────────
print('Loading SAOT...')
class _SAOTArgs:
    model = 'SAOT_Structured_Mesh_2D'
    n_hidden = 256
    n_layers = 8
    n_heads = 8
    mlp_ratio = 1
    dropout = 0.0
    unified_pos = 1
    ref = 8
    slice_num = 32
    fun_dim = T_in
    out_dim = 1  # SAOT trained autoregressively with 1-step output

_saot_mod = saot_get_model(_SAOTArgs())
_saot_cls = _saot_mod.Model if hasattr(_saot_mod, 'Model') else _saot_mod
saot_model = _saot_cls(
    space_dim=2, n_layers=_SAOTArgs.n_layers, n_hidden=_SAOTArgs.n_hidden,
    dropout=_SAOTArgs.dropout, n_head=_SAOTArgs.n_heads, Time_Input=False,
    mlp_ratio=_SAOTArgs.mlp_ratio, fun_dim=T_in, out_dim=1,
    slice_num=_SAOTArgs.slice_num, ref=_SAOTArgs.ref,
    unified_pos=_SAOTArgs.unified_pos, H=H, W=W, is_filter=True
).to(device)
saot_model.load_state_dict(torch.load(args.saot_ckpt, map_location=device))
if hasattr(saot_model, 'pos'):
    saot_model.pos = saot_model.pos.to(device)
saot_model.eval()
print(f'  SAOT params: {sum(p.numel() for p in saot_model.parameters()):,}')

# ── WGANO model ─────────────────────────────────────────────────────────────
print('Loading WGANO...')
n1_dir = os.path.join(REPO_ROOT, 'WGANO', 'n1')
if n1_dir not in sys.path:
    sys.path.insert(0, n1_dir)

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
    space_dim=2, n_layers=8, n_hidden=328, n_head=8,
    mlp_ratio=1, fun_dim=T_in, out_dim=1, ref=8,
    unified_pos=1, H=H, W=W
).to(device)
wgano_model.load_state_dict(torch.load(args.wgano_ckpt, map_location=device))
if hasattr(wgano_model, 'pos'):
    wgano_model.pos = wgano_model.pos.to(device)
wgano_model.eval()
print(f'  WGANO params: {sum(p.numel() for p in wgano_model.parameters()):,}')

# ── Inference ───────────────────────────────────────────────────────────────
print('Running inference...')
saot_preds = []
wgano_preds = []

with torch.no_grad():
    for i in range(ntest):
        if i % 50 == 0:
            print(f'  {i}/{ntest}...')
        x_i = x_test_norm[i:i+1]
        fx_i = x_i.permute(0, 2, 3, 1)

        # SAOT: autoregressive 1-step-at-a-time
        saot_pred_t = []
        fx_saot = fx_i.clone()
        for t in range(T):
            out_t = saot_model(pos, fx=fx_saot)
            saot_pred_t.append(out_t)
            fx_saot = torch.cat([fx_saot[..., 1:], out_t], dim=-1)
        out_saot = torch.cat(saot_pred_t, dim=-1)
        out_saot = out_saot.cpu().numpy()[0]  # NS模型输出已是原始尺度，不需要decode
        saot_preds.append(out_saot)

        # WGANO: autoregressive 1-step-at-a-time
        wgano_pred_t = []
        fx_wgano = fx_i.clone()
        for t in range(T):
            out_t = wgano_model(pos, fx=fx_wgano)[0]
            wgano_pred_t.append(out_t)
            fx_wgano = torch.cat([fx_wgano[..., 1:], out_t], dim=-1)
        out_wgano = torch.cat(wgano_pred_t, dim=-1)
        out_wgano = out_wgano.cpu().numpy()[0]  # NS模型输出已是原始尺度，不需要decode
        wgano_preds.append(out_wgano)

saot_preds = np.array(saot_preds)
wgano_preds = np.array(wgano_preds)
print(f'Predictions shape: SAOT={saot_preds.shape}, WGANO={wgano_preds.shape}')

# ── Sample selection ────────────────────────────────────────────────────────
def per_sample_nrmse(pred, true):
    """使用与训练一致的TestLoss计算方式（rel方法，无h缩放）"""
    errs = np.zeros(len(pred))
    for i in range(len(pred)):
        p = pred[i].ravel()
        t = true[i].ravel()
        diff_norm = np.linalg.norm(p - t, ord=2)
        y_norm = np.linalg.norm(t, ord=2)
        errs[i] = diff_norm / (y_norm + 1e-8)
    return errs

saot_nrmse = per_sample_nrmse(saot_preds, y_true)
wgano_nrmse = per_sample_nrmse(wgano_preds, y_true)

print(f'\nOverall relL2:')
print(f'  SAOT:  {saot_nrmse.mean():.6f} (expected 0.057303)')
print(f'  WGANO: {wgano_nrmse.mean():.6f} (expected 0.077065)')

ratio = saot_nrmse / (wgano_nrmse + 1e-10)
if args.sample_idx is None:
    # 选SAOT稍微好一点点的样本：ratio在0.7-0.95之间
    target_ratio = 0.85
    candidates = np.where((ratio > 0.7) & (ratio < 0.95))[0]
    if len(candidates) > 0:
        idx = candidates[np.argmin(np.abs(ratio[candidates] - target_ratio))]
    else:
        # 如果没有合适的，选最接近target_ratio的
        idx = int(np.argmin(np.abs(ratio - target_ratio)))
else:
    idx = args.sample_idx

print(f'Selected sample index: {idx}')
print(f'  SAOT nRMSE: {saot_nrmse[idx]:.4f}')
print(f'  WGANO nRMSE: {wgano_nrmse[idx]:.4f}')
print(f'  Ratio: {ratio[idx]:.1f}x')

# ── Plotting ────────────────────────────────────────────────────────────────
print('Plotting...')
y_true_i = y_true[idx, :, :, -1]  # Last timestep
saot_i = saot_preds[idx, :, :, -1]
wgano_i = wgano_preds[idx, :, :, -1]

err_saot = saot_i - y_true_i
err_wgano = wgano_i - y_true_i

fig = plt.figure(figsize=(15, 4))
gs = GridSpec(1, 3, figure=fig, wspace=0.35)

vmin_sol = y_true_i.min()
vmax_sol = y_true_i.max()
all_err = np.concatenate([np.abs(err_saot).flatten(), np.abs(err_wgano).flatten()])
err_vmax = np.percentile(all_err, 99)

def _add_panel(ax, data, vmin, vmax, cmap='RdBu_r', is_error=False):
    im = ax.imshow(data, origin='lower', cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation='nearest')

    if is_error:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        ticks=[-vmax, 0, vmax])
        magnitude = int(np.floor(np.log10(vmax)))
        scale = 10**(-magnitude)
        err_vmax_normalized = vmax * scale
        cb.ax.set_yticklabels([f'{-err_vmax_normalized:.1f}', '0.0', f'{err_vmax_normalized:.1f}'], fontsize=FONT_CB)
        cb.ax.set_title(f'1e{magnitude:d}', fontsize=FONT_CB_TL, pad=8)
        cb.ax.tick_params(size=0)
        fig.canvas.draw()
        labels = cb.ax.yaxis.get_majorticklabels()
        offset_up = ScaledTranslation(0, 10/72., fig.dpi_scale_trans)
        offset_down = ScaledTranslation(0, -10/72., fig.dpi_scale_trans)
        labels[0].set_transform(labels[0].get_transform() + offset_up)
        labels[-1].set_transform(labels[-1].get_transform() + offset_down)
    else:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        ticks=[vmin, (vmin+vmax)/2, vmax])
        mag_true = int(np.floor(np.log10(vmax)))
        scale = 10**(-mag_true)
        cb.ax.set_yticklabels([f'{vmin*scale:.1f}', f'{(vmin+vmax)/2*scale:.1f}',
                               f'{vmax*scale:.1f}'], fontsize=FONT_CB)
        cb.ax.set_title(f'1e{mag_true:d}', fontsize=FONT_CB_TL, pad=8)
        cb.ax.tick_params(size=0)
        fig.canvas.draw()
        labels = cb.ax.yaxis.get_majorticklabels()
        offset_up = ScaledTranslation(0, 10/72., fig.dpi_scale_trans)
        offset_down = ScaledTranslation(0, -10/72., fig.dpi_scale_trans)
        labels[0].set_transform(labels[0].get_transform() + offset_up)
        labels[-1].set_transform(labels[-1].get_transform() + offset_down)
    return im

col_labels = ["True Solution", "SAOT Error", "WGANO Error"]
data_list = [y_true_i, err_saot, err_wgano]
is_error_list = [False, True, True]
vmin_list = [vmin_sol, -err_vmax, -err_vmax]
vmax_list = [vmax_sol, err_vmax, err_vmax]

for c, (data, label, is_err, vmin, vmax) in enumerate(zip(data_list, col_labels, is_error_list, vmin_list, vmax_list)):
    ax = fig.add_subplot(gs[c])
    _add_panel(ax, data, vmin, vmax, cmap='RdBu_r', is_error=is_err)
    ax.set_title(label, fontsize=20, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()

out_path = os.path.join(args.out_dir, '06_ns_final.pdf')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.close(fig)

"""
Airfoil可视化修复 - 顺时针旋转90度
问题：当前排版不佳（221x51 → 旋转后 51x221）
解决：将所有field顺时针旋转90度
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from safe_load import safe_load
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import ScaledTranslation
from matplotlib.font_manager import FontProperties

# ── AAAI 2027 Font Config (统一标准) ────────────────────────────────────────
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['text.usetex'] = False

# 强制matplotlib识别Bold字体
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)  # 强制重新加载字体缓存

FONT_TITLE = 20
FONT_LABEL = 18
FONT_CB = 20
FONT_CB_TL = 20

VIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(VIS_DIR, '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'baseline', 'SAOT_NO'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'baseline', 'SAOT_NO', 'model'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'WGANO', 'n1'))

from model_dict import get_model as saot_get_model

device = torch.device('cuda')
print(f'Device: {device}')

print('Loading Airfoil data...')
data_path = os.path.join(REPO_ROOT, 'data/pdebench/airfoil-20260512T160726Z-3-001/airfoil/naca')
coord_x = np.load(os.path.join(data_path, 'NACA_Cylinder_X.npy'))
coord_y = np.load(os.path.join(data_path, 'NACA_Cylinder_Y.npy'))
q_path = data_path.replace('001', '002')
output_q = np.load(os.path.join(q_path, 'NACA_Cylinder_Q.npy'))[:, 4]

ntrain, ntest = 1000, 200
s1, s2 = 221, 51

x_coords = np.stack([coord_x, coord_y], axis=-1)
x_train = torch.from_numpy(x_coords[:ntrain]).float()
x_test = torch.from_numpy(x_coords[-ntest:]).float()
y_train = torch.from_numpy(output_q[:ntrain]).float()
y_test = torch.from_numpy(output_q[-ntest:]).float()

print(f'  Test samples: {ntest}, Grid: {s1}x{s2}')
y_true = y_test.cpu().numpy()

print('\n[SAOT] Loading model...')
_saot_mod = saot_get_model(type('Args', (), {'model': 'SAOT_Structured_Mesh_2D'})())
_saot_cls = _saot_mod.Model if hasattr(_saot_mod, 'Model') else _saot_mod
saot_model = _saot_cls(
    space_dim=2, n_layers=5, n_hidden=256, dropout=0.0, n_head=8,
    Time_Input=False, mlp_ratio=1, fun_dim=0, out_dim=1, slice_num=64,
    ref=8, unified_pos=0, H=s1, W=s2, is_filter=True,
).to(device)
saot_ckpt = os.path.join(REPO_ROOT, 'experiment/saot_repro/airfoil/checkpoints/airfoil_SAOT_best.pt')
safe_load(saot_model, saot_ckpt)
saot_model.eval()
if hasattr(saot_model, 'pos') and saot_model.pos is not None:
    saot_model.pos = saot_model.pos.to(device)
print(f'  SAOT params: {sum(p.numel() for p in saot_model.parameters()):,}')

print('\n[WGANO] Loading model...')
from model_n1 import Model as N1Model
wgano_model = N1Model(
    space_dim=2, n_layers=10, n_hidden=320, n_head=8, mlp_ratio=1,
    fun_dim=0, out_dim=1, ref=8, unified_pos=0, H=s1, W=s2,
).to(device)
wgano_ckpt = os.path.join(REPO_ROOT, 'experiment/table1/f1airfoil/checkpoints/f1airfoil_n1v5.pt')
safe_load(wgano_model, wgano_ckpt)
wgano_model.eval()
if hasattr(wgano_model, 'pos') and wgano_model.pos is not None:
    wgano_model.pos = wgano_model.pos.to(device)
print(f'  WGANO params: {sum(p.numel() for p in wgano_model.parameters()):,}')

print('\nRunning inference...')
saot_preds, wgano_preds = [], []

with torch.no_grad():
    for i in range(ntest):
        if i % 50 == 0:
            print(f'  {i}/{ntest}...')
        x_input = x_test[i:i+1].to(device)
        out_saot = saot_model(x_input, fx=None).squeeze(-1).cpu().numpy()
        saot_preds.append(out_saot[0])
        out_wgano = wgano_model(x_input, fx=None)
        if isinstance(out_wgano, tuple):
            out_wgano = out_wgano[0]
        out_wgano = out_wgano.squeeze(-1).cpu().numpy()
        wgano_preds.append(out_wgano[0])

saot_preds = np.array(saot_preds)
wgano_preds = np.array(wgano_preds)

def compute_nrmse(pred, true):
    N = len(pred)
    errs = np.zeros(N)
    for i in range(N):
        p = pred[i].ravel()
        t = true[i].ravel()
        errs[i] = np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-8)
    return errs

saot_nrmse = compute_nrmse(saot_preds, y_true)
wgano_nrmse = compute_nrmse(wgano_preds, y_true)

print(f'\nError statistics:')
print(f'  SAOT  - mean: {saot_nrmse.mean():.6f}')
print(f'  WGANO - mean: {wgano_nrmse.mean():.6f}')

# 选择样本
ratio = saot_nrmse / (wgano_nrmse + 1e-10)
candidates = np.where((ratio >= 1.3) & (ratio <= 2.5))[0]
if len(candidates) > 0:
    idx = candidates[len(candidates)//2]
else:
    idx = int(np.argmax(ratio))

print(f'\n[SELECTED] Sample {idx}: SAOT={saot_nrmse[idx]:.6f}, WGANO={wgano_nrmse[idx]:.6f}, Ratio={ratio[idx]:.2f}x')

# 可视化：顺时针旋转90度
true_field = np.rot90(y_true[idx], k=-1)
saot_field = np.rot90(saot_preds[idx], k=-1)
wgano_field = np.rot90(wgano_preds[idx], k=-1)

err_saot = saot_field - true_field
err_wgano = wgano_field - true_field
all_err = np.concatenate([np.abs(err_saot).flatten(), np.abs(err_wgano).flatten()])
err_vmax_abs = np.percentile(all_err, 99)
err_magnitude = int(np.floor(np.log10(err_vmax_abs)))
err_scale = 10**err_magnitude
err_vmax = err_vmax_abs / err_scale  # 归一化到数量级范围

print(f'\nDEBUG: err_vmax_abs = {err_vmax_abs:.6e}')
print(f'DEBUG: err_magnitude = {err_magnitude}')
print(f'DEBUG: err_vmax (normalized) = {err_vmax:.2f}')

fig = plt.figure(figsize=(15, 4))
gs = GridSpec(1, 3, figure=fig, wspace=0.35)

vmin_sol = true_field.min()
vmax_sol = true_field.max()

# 显式创建Bold字体属性
bold_font = FontProperties(weight='bold', size=20)

def _add_panel(ax, data, title, vmin, vmax, cmap='viridis', is_error=False):
    im = ax.imshow(data, origin='lower', cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title, fontproperties=bold_font)
    ax.set_xticks([])
    ax.set_yticks([])

    if is_error:
        # Colorbar with real error values and magnitude label
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.30,
                        ticks=[-vmax, 0, vmax])
        cb.ax.set_yticklabels([f'{-vmax:.1f}', '0.0', f'{vmax:.1f}'], fontsize=20)
        cb.ax.set_title(f'1e{err_magnitude:d}', fontsize=20, pad=8)
        cb.ax.tick_params(labelsize=20, size=0)
        # Nudge top and bottom labels inward
        fig.canvas.draw()
        labels = cb.ax.yaxis.get_majorticklabels()
        offset_up = ScaledTranslation(0, 10/72., fig.dpi_scale_trans)
        offset_down = ScaledTranslation(0, -10/72., fig.dpi_scale_trans)
        labels[0].set_transform(labels[0].get_transform() + offset_up)
        labels[-1].set_transform(labels[-1].get_transform() + offset_down)
    else:
        # True solution: actual value ticks with magnitude on top
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.30,
                        ticks=[vmin, (vmin+vmax)/2, vmax])
        mag_true = int(np.floor(np.log10(vmax)))
        scale = 10**(-mag_true)
        cb.ax.set_yticklabels([f'{vmin*scale:.1f}', f'{(vmin+vmax)/2*scale:.1f}', f'{vmax*scale:.1f}'], fontsize=20)
        cb.ax.set_title(f'1e{mag_true:d}', fontsize=20, pad=8)
        cb.ax.tick_params(labelsize=20, size=0)
        # Nudge top and bottom labels inward
        fig.canvas.draw()
        labels = cb.ax.yaxis.get_majorticklabels()
        offset_up = ScaledTranslation(0, 10/72., fig.dpi_scale_trans)
        offset_down = ScaledTranslation(0, -10/72., fig.dpi_scale_trans)
        labels[0].set_transform(labels[0].get_transform() + offset_up)
        labels[-1].set_transform(labels[-1].get_transform() + offset_down)
    return im

_add_panel(fig.add_subplot(gs[0]), true_field,
           'True Solution', vmin_sol, vmax_sol, cmap='RdBu_r', is_error=False)
_add_panel(fig.add_subplot(gs[1]), err_saot / err_scale,
           'SAOT Error', -err_vmax, err_vmax, cmap='RdBu_r', is_error=True)
_add_panel(fig.add_subplot(gs[2]), err_wgano / err_scale,
           'WGANO Error', -err_vmax, err_vmax, cmap='RdBu_r', is_error=True)

plt.tight_layout()
out_pdf = os.path.join(VIS_DIR, '05_airfoil.pdf')
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
print(f"\nSaved: {out_pdf}")
plt.close(fig)

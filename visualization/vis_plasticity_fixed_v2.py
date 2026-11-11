"""
Plasticity可视化修复版v2
关键修复：正确解包WGANO模型的返回值
"""
import os, sys
import numpy as np
import scipy.io as scio
import torch
import matplotlib
matplotlib.use('Agg')
from safe_load import safe_load
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import ScaledTranslation

# ── Font Config ────────────────────────────────────────────────────
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['text.usetex'] = False

FONT_TITLE = 20
FONT_LABEL = 18
FONT_CB = 20
FONT_CB_TL = 20

VIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(VIS_DIR, '../../..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'WGANO', 'n1'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'baseline', 'SAOT_NO'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'baseline', 'SAOT_NO', 'model'))

from model_n1 import Model as WGANOModel
from model_dict import get_model
from utils.normalizer import UnitTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

print('Loading Plasticity data...')
data_path = os.path.join(REPO_ROOT, 'data', 'pdebench', 'plas_N987_T20.mat')
N, ntrain, ntest = 987, 900, 80
s1, s2, T, Deformation = 101, 31, 20, 4

data = scio.loadmat(data_path)
input_data = torch.tensor(data['input'], dtype=torch.float)
output = torch.tensor(data['output'], dtype=torch.float).transpose(-2, -1)

# WGANO: 使用全局mean/std归一化（对齐train_n1.py line 67-70）
x_train_wgano = input_data[:ntrain, :s1].reshape(ntrain, s1, 1).repeat(1, 1, s2).reshape(ntrain, s1, s2, 1)
fx_mean = x_train_wgano.mean(dim=(0, 1, 2), keepdim=True)
fx_std = x_train_wgano.std(dim=(0, 1, 2), keepdim=True) + 1e-8
x_test_wgano = input_data[-ntest:, :s1].reshape(ntest, s1, 1).repeat(1, 1, s2).reshape(ntest, s1, s2, 1)
x_test_wgano = (x_test_wgano - fx_mean) / fx_std

# SAOT: 使用UnitTransformer归一化（对齐exp_plas.py）
x_train_saot = input_data[:ntrain, :s1].reshape(ntrain, s1, 1).repeat(1, 1, s2).reshape(ntrain, -1, 1)
x_normalizer_saot = UnitTransformer(x_train_saot)
x_test_saot = input_data[-ntest:, :s1].reshape(ntest, s1, 1).repeat(1, 1, s2).reshape(ntest, -1, 1)
x_test_saot = x_normalizer_saot.encode(x_test_saot).reshape(ntest, s1, s2, 1)

y_test = output[-ntest:, :s1, :s2].reshape(ntest, s1, s2, Deformation, T)

print(f'WGANO归一化: global mean={fx_mean.item():.4f}, std={fx_std.item():.4f}')
print(f'SAOT归一化: UnitTransformer (per-point)')

# SAOT: 保持原来的方式（之前评估是对的0.000914）
x_coords = np.linspace(0, 1, s1)
y_coords = np.linspace(0, 1, s2)
xx, yy = np.meshgrid(x_coords, y_coords)  # 默认indexing='xy'
pos = np.c_[xx.ravel(), yy.ravel()]
pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
pos_saot = pos.repeat(ntest, 1, 1).reshape(ntest, s1, s2, 2).to(device)

# WGANO: 使用indexing='ij'对齐训练代码
gx = np.linspace(0, 1, s1)
gy = np.linspace(0, 1, s2)
xx_wgano, yy_wgano = np.meshgrid(gx, gy, indexing='ij')
pos_wgano = torch.tensor(np.stack([xx_wgano, yy_wgano], axis=-1), dtype=torch.float)
pos_wgano = pos_wgano.unsqueeze(0).repeat(ntest, 1, 1, 1).to(device)

x_test_saot = x_test_saot.to(device)
x_test_wgano = x_test_wgano.to(device)
y_test = y_test.to(device)

t_vals = torch.tensor(np.linspace(0, 1, T), dtype=torch.float).to(device)

print('\n[SAOT] Loading model...')
_saot_mod = get_model(type('Args', (), {'model': 'SAOT_Structured_Mesh_2D'})())
_saot_cls = _saot_mod.Model if hasattr(_saot_mod, 'Model') else _saot_mod
saot_model = _saot_cls(
    space_dim=2, n_layers=8, n_hidden=192, dropout=0.0, n_head=8,
    Time_Input=True, mlp_ratio=1, fun_dim=1, out_dim=Deformation,
    slice_num=64, ref=8, unified_pos=0, H=s1, W=s2, is_filter=True
).to(device)
saot_ckpt = os.path.join(REPO_ROOT, 'experiment', 'saot_repro', 'plasticity', 'checkpoints', 'plas_SAOT_best.pt')
saot_model.load_state_dict(torch.load(saot_ckpt, map_location=device), strict=False)  # SAOT可以用strict=False
saot_model.eval()
print(f'  SAOT params: {sum(p.numel() for p in saot_model.parameters()):,}')

print('\n[WGANO] Loading model...')
wgano_model = WGANOModel(
    space_dim=2, n_layers=8, n_hidden=240, n_head=8, mlp_ratio=1,
    fun_dim=1, out_dim=Deformation, ref=8, unified_pos=0, H=s1, W=s2,
    Time_Input=True, use_qk_norm=False  # checkpoint没有qk_norm参数
).to(device)
wgano_ckpt = os.path.join(REPO_ROOT, 'experiment', 'table1', 'f1plasticity', 'checkpoints', 'f1plas_n1_best.pt')
wgano_model.load_state_dict(torch.load(wgano_ckpt, map_location=device))  # 不用strict=False
wgano_model.eval()
print(f'  WGANO params: {sum(p.numel() for p in wgano_model.parameters()):,}')

print('\nRunning autoregressive inference...')
saot_preds_all, wgano_preds_all = [], []

with torch.no_grad():
    for i in range(ntest):
        if i % 20 == 0:
            print(f'  {i}/{ntest}...')

        pos_i_saot = pos_saot[i:i+1]
        pos_i_wgano = pos_wgano[i:i+1]
        fx_saot = x_test_saot[i:i+1]
        fx_wgano = x_test_wgano[i:i+1]

        saot_pred_t, wgano_pred_t = [], []

        for t in range(T):
            t_input = t_vals[t].reshape(1, 1).repeat(1, 1)

            # SAOT: 返回单个tensor
            out_saot = saot_model(pos_i_saot, fx=fx_saot, T=t_input)
            saot_pred_t.append(out_saot.cpu().numpy()[..., np.newaxis])

            # WGANO: 返回(im, Ud, Us)三个值，对齐train_n1.py line 213
            im, _, _ = wgano_model(pos_i_wgano, fx=fx_wgano, T=t_input)
            wgano_pred_t.append(im.cpu().numpy()[..., np.newaxis])

        saot_preds_all.append(np.concatenate(saot_pred_t, axis=-1)[0])
        wgano_preds_all.append(np.concatenate(wgano_pred_t, axis=-1)[0])

saot_preds_all = np.array(saot_preds_all)
wgano_preds_all = np.array(wgano_preds_all)
y_true = y_test.cpu().numpy()

def compute_nrmse(pred, true):
    """使用与训练一致的TestLoss计算方式（rel方法，无h缩放）"""
    N = len(pred)
    errs = np.zeros(N)
    for i in range(N):
        p = pred[i].ravel()
        t = true[i].ravel()
        diff_norm = np.linalg.norm(p - t, ord=2)
        y_norm = np.linalg.norm(t, ord=2)
        errs[i] = diff_norm / (y_norm + 1e-8)
    return errs

saot_nrmse = compute_nrmse(saot_preds_all, y_true)
wgano_nrmse = compute_nrmse(wgano_preds_all, y_true)

print(f'\nError statistics:')
print(f'  SAOT  relL2: {saot_nrmse.mean():.6f} (expected 0.000914)')
print(f'  WGANO relL2: {wgano_nrmse.mean():.6f} (expected 0.000751)')

# 选择合适样本
ratio = saot_nrmse / (wgano_nrmse + 1e-10)
target_min, target_max = 0.7, 1.3
candidates = np.where((ratio >= target_min) & (ratio <= target_max))[0]
if len(candidates) > 0:
    idx = candidates[len(candidates)//2]
else:
    idx = int(np.argmin(np.abs(ratio - 1.0)))

print(f'\n[SELECTED] Sample {idx}: SAOT={saot_nrmse[idx]:.6f}, WGANO={wgano_nrmse[idx]:.6f}, Ratio={ratio[idx]:.2f}x')

# 可视化：最后时间步，第1个分量，顺时针旋转90度
c_vis, t_vis = 0, -1
true_field = np.rot90(y_true[idx, :, :, c_vis, t_vis], k=-1)
saot_field = np.rot90(saot_preds_all[idx, :, :, c_vis, t_vis], k=-1)
wgano_field = np.rot90(wgano_preds_all[idx, :, :, c_vis, t_vis], k=-1)

err_saot = saot_field - true_field
err_wgano = wgano_field - true_field

# 误差99百分位（统一计算，避免重复）
all_err = np.concatenate([np.abs(err_saot).flatten(), np.abs(err_wgano).flatten()])
err_vmax_raw = np.percentile(all_err, 99)
err_magnitude = int(np.floor(np.log10(err_vmax_raw)))
err_scale = 10**(-err_magnitude)
err_vmax_normalized = err_vmax_raw * err_scale

# Colorbar设置
true_min = true_field.min()
true_max = true_field.max()

print(f'真实值数据范围: [{true_min:.4f}, {true_max:.4f}]')
print(f'误差范围: SAOT [{err_saot.min():.4f}, {err_saot.max():.4f}], WGANO [{err_wgano.min():.4f}, {err_wgano.max():.4f}]')
print(f'误差99百分位: {err_vmax_raw:.6f} (归一化: {err_vmax_normalized:.1f}×10^{err_magnitude})')

fig = plt.figure(figsize=(15, 4))
gs = GridSpec(1, 3, figure=fig, wspace=0.35)

col_labels = ["True Solution", "SAOT Error", "WGANO Error"]
data_list = [true_field, err_saot, err_wgano]

for c, (data, label) in enumerate(zip(data_list, col_labels)):
    ax = fig.add_subplot(gs[c])

    if c == 0:
        # True solution
        if true_min < 0:
            # 有负值，使用对称范围
            true_absmax = max(abs(true_min), abs(true_max))
            true_magnitude = int(np.floor(np.log10(true_absmax)))
            true_scale = 10**true_magnitude
            tick_spacing = np.ceil(true_absmax / true_scale / 2 * 10) / 10 * true_scale

            if tick_spacing * 2 / true_scale >= 10:
                true_magnitude += 1
                true_scale = 10**true_magnitude
                tick_spacing = np.ceil(true_absmax / true_scale / 2 * 10) / 10 * true_scale

            vmin = -tick_spacing * 2
            vmax = tick_spacing * 2
        else:
            # 纯正值，从0开始
            true_magnitude = int(np.floor(np.log10(true_max)))
            true_scale = 10**true_magnitude
            tick_spacing = np.ceil(true_max / true_scale / 2 * 10) / 10 * true_scale

            if tick_spacing * 2 / true_scale >= 10:
                true_magnitude += 1
                true_scale = 10**true_magnitude
                tick_spacing = np.ceil(true_max / true_scale / 2 * 10) / 10 * true_scale

            vmin = 0
            vmax = tick_spacing * 2

        im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(label, fontsize=20, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        scale = 10**(-true_magnitude)
        if vmin < 0:
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.38, ticks=[vmin, 0, vmax])
            cb.ax.set_yticklabels([f'{vmin*scale:.1f}', '0.0', f'{vmax*scale:.1f}'], fontsize=FONT_CB)
        else:
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.38, ticks=[0, tick_spacing, vmax])
            cb.ax.set_yticklabels([f'{0:.1f}', f'{tick_spacing*scale:.1f}', f'{vmax*scale:.1f}'], fontsize=FONT_CB)

        cb.ax.set_title(f'1e{true_magnitude:d}', fontsize=FONT_CB_TL, pad=8)
        cb.ax.tick_params(size=0)
        # Offset
        fig.canvas.draw()
        labels = cb.ax.yaxis.get_majorticklabels()
        offset_up = ScaledTranslation(0, 10/72., fig.dpi_scale_trans)
        offset_down = ScaledTranslation(0, -10/72., fig.dpi_scale_trans)
        labels[0].set_transform(labels[0].get_transform() + offset_up)
        labels[-1].set_transform(labels[-1].get_transform() + offset_down)
    else:
        # Error maps - 使用之前统一计算的误差值
        vmin, vmax = -err_vmax_raw, err_vmax_raw
        im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(label, fontsize=20, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.38, ticks=[-err_vmax_raw, 0, err_vmax_raw])
        cb.ax.set_yticklabels([f'{-err_vmax_normalized:.1f}', '0.0', f'{err_vmax_normalized:.1f}'], fontsize=FONT_CB)
        cb.ax.set_title(f'1e{err_magnitude:d}', fontsize=FONT_CB_TL, pad=8)
        cb.ax.tick_params(size=0)
        # Offset
        fig.canvas.draw()
        labels = cb.ax.yaxis.get_majorticklabels()
        offset_up = ScaledTranslation(0, 10/72., fig.dpi_scale_trans)
        offset_down = ScaledTranslation(0, -10/72., fig.dpi_scale_trans)
        labels[0].set_transform(labels[0].get_transform() + offset_up)
        labels[-1].set_transform(labels[-1].get_transform() + offset_down)

plt.tight_layout()
out_pdf = os.path.join(VIS_DIR, '02_plasticity.pdf')
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
print(f"\nSaved: {out_pdf}")
plt.close(fig)

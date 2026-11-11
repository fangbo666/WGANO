"""
Pipe可视化优化 - 选择更好的样本
当前问题：样本ratio可能不够接近TABLE3
优化：在合理ratio范围内选择对比度更好的样本
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

# ── AAAI 2027 Font Config (统一标准) ────────────────────────────────────────
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

VIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(VIS_DIR, '../../..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'baseline', 'SAOT_NO'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'baseline', 'SAOT_NO', 'model'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'WGANO', 'n1'))

from SAOT_Structured_Mesh_2D import Model as SAOTModel
from model_n1 import Model as WGANOModel
from utils.normalizer import UnitTransformer
from utils.testloss import TestLoss

device = torch.device('cuda')
print(f'Device: {device}')

print('Loading Pipe data...')
ntrain, ntest = 1000, 200
s1 = s2 = 129

data_path = os.path.join(REPO_ROOT, 'data/pdebench/pipe-20260512T155752Z-3-001/pipe')
inputX = torch.tensor(np.load(os.path.join(data_path, 'Pipe_X.npy')), dtype=torch.float)
inputY = torch.tensor(np.load(os.path.join(data_path, 'Pipe_Y.npy')), dtype=torch.float)
inp = torch.stack([inputX, inputY], dim=-1)
output = torch.tensor(np.load(os.path.join(data_path, 'Pipe_Q.npy'))[:, 0], dtype=torch.float)

r1 = r2 = 1
x_train = inp[:ntrain, ::r1, ::r2][:, :s1, :s2]
y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
x_test = inp[-ntest:, ::r1, ::r2][:, :s1, :s2]
y_test = output[-ntest:, ::r1, ::r2][:, :s1, :s2]

x_train = x_train.reshape(ntrain, -1, 2)
x_test = x_test.reshape(ntest, -1, 2)
y_train = y_train.reshape(ntrain, -1)
y_test = y_test.reshape(ntest, -1)

x_normalizer = UnitTransformer(x_train)
y_normalizer = UnitTransformer(y_train)

x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

x_train = x_train.reshape(ntrain, s1, s2, -1)
x_test = x_test.reshape(ntest, s1, s2, -1)
y_train = y_train.reshape(ntrain, s1, s2, -1)
y_test = y_test.reshape(ntest, s1, s2, -1)

x_normalizer.cuda()
y_normalizer.cuda()

x_test = x_test.to(device)
y_test = y_test.to(device)

print(f'  Test samples: {ntest}, Grid: {s1}x{s2}')

print('\n[SAOT] Loading model...')
saot_model = SAOTModel(
    space_dim=2, n_layers=8, n_hidden=192, dropout=0.0, n_head=8,
    Time_Input=False, mlp_ratio=1, fun_dim=0, out_dim=1,
    slice_num=64, ref=8, unified_pos=0, H=s1, W=s2, is_filter=False
).to(device)
saot_ckpt = os.path.join(REPO_ROOT, 'experiment/saot_repro/pipe/checkpoints/pipe_SAOT_best.pt')
safe_load(saot_model, saot_ckpt)
saot_model.eval()
print(f'  SAOT params: {sum(p.numel() for p in saot_model.parameters()):,}')

print('\n[WGANO] Loading model...')
wgano_model = WGANOModel(
    space_dim=2, n_layers=7, n_hidden=184, dropout=0.0, n_head=8,
    mlp_ratio=1, fun_dim=0, out_dim=1, ref=8, unified_pos=0, H=s1, W=s2,
    use_qk_norm=False
).to(device)
wgano_ckpt = os.path.join(REPO_ROOT, 'experiment/table1/f1pipe/checkpoints/f1pipe_n1_best.pt')
safe_load(wgano_model, wgano_ckpt)
wgano_model.eval()
print(f'  WGANO params: {sum(p.numel() for p in wgano_model.parameters()):,}')

print('\nRunning inference...')
myloss = TestLoss(size_average=False)
saot_l2, wgano_l2 = 0.0, 0.0
saot_preds_all, wgano_preds_all = [], []

with torch.no_grad():
    for i in range(ntest):
        if i % 50 == 0:
            print(f'  {i}/{ntest}...')

        x_i = x_test[i:i+1]
        y_i = y_test[i:i+1]

        out_saot = saot_model(x_i, fx=None)
        out_saot_decoded = y_normalizer.decode(out_saot)
        # y_i是原始未归一化数据，不需要decode

        saot_l2 += myloss(out_saot_decoded.reshape(1, -1), y_i.reshape(1, -1)).item()
        saot_preds_all.append(out_saot_decoded.cpu().numpy()[0])

        out_wgano = wgano_model(x_i, fx=None)
        if isinstance(out_wgano, tuple):
            out_wgano = out_wgano[0]
        out_wgano_decoded = y_normalizer.decode(out_wgano)

        wgano_l2 += myloss(out_wgano_decoded.reshape(1, -1), y_i.reshape(1, -1)).item()
        wgano_preds_all.append(out_wgano_decoded.cpu().numpy()[0])

saot_preds_all = np.array(saot_preds_all)
wgano_preds_all = np.array(wgano_preds_all)
y_true = y_test.cpu().numpy()  # y_test已经是原始尺度，不需要decode

print(f'\nError statistics:')
print(f'  SAOT  relL2: {saot_l2/ntest:.6f}')
print(f'  WGANO relL2: {wgano_l2/ntest:.6f}')

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

# 优化样本选择：在ratio 1.5-2.5x范围内，选择对比度好的样本
ratio = saot_nrmse / (wgano_nrmse + 1e-10)
target_min, target_max = 1.5, 2.5
candidates = np.where((ratio >= target_min) & (ratio <= target_max))[0]

if len(candidates) > 0:
    # 在候选中，选择误差适中的样本（避免极端值）
    mean_saot = saot_nrmse[candidates].mean()
    dist_to_mean = np.abs(saot_nrmse[candidates] - mean_saot)
    idx = candidates[np.argmin(dist_to_mean)]
    print(f'\n[OPTIMIZED] Selected sample {idx}: SAOT={saot_nrmse[idx]:.6f}, WGANO={wgano_nrmse[idx]:.6f}, Ratio={ratio[idx]:.2f}x')
    print(f'  (Found {len(candidates)} candidates in ratio range {target_min}-{target_max}x)')
else:
    idx = int(np.argmax(ratio))
    print(f'\n[FALLBACK] Selected sample {idx}: SAOT={saot_nrmse[idx]:.6f}, WGANO={wgano_nrmse[idx]:.6f}, Ratio={ratio[idx]:.2f}x')

# 可视化
true_field = y_true[idx].reshape(s1, s2)
saot_field = saot_preds_all[idx].reshape(s1, s2)
wgano_field = wgano_preds_all[idx].reshape(s1, s2)

err_saot = saot_field - true_field
err_wgano = wgano_field - true_field
err_vmax = np.percentile(np.concatenate([np.abs(err_saot).flatten(), np.abs(err_wgano).flatten()]), 99)
err_magnitude = int(np.floor(np.log10(err_vmax)))

print(f'真实值数据范围: [{true_field.min():.4f}, {true_field.max():.4f}]')
print(f'误差范围: SAOT [{err_saot.min():.4f}, {err_saot.max():.4f}], WGANO [{err_wgano.min():.4f}, {err_wgano.max():.4f}]')
print(f'误差99百分位: {err_vmax:.6f} (magnitude: 1e{err_magnitude})')

fig = plt.figure(figsize=(15, 4))
gs = GridSpec(1, 3, figure=fig, wspace=0.35)
col_labels = ["True Solution", "SAOT Error", "WGANO Error"]
data_list = [true_field, err_saot, err_wgano]

for c, (data, label) in enumerate(zip(data_list, col_labels)):
    ax = fig.add_subplot(gs[c])

    if c == 0:
        vmin, vmax = data.min(), data.max()

        # 如果数据有负值，使用对称范围
        if vmin < 0:
            data_absmax = max(abs(vmin), abs(vmax))
            tick_spacing = np.ceil(data_absmax / 2 * 10) / 10
            cb_vmin = -tick_spacing * 2
            cb_vmax = tick_spacing * 2
            ticks = [cb_vmin, 0, cb_vmax]
        else:
            # 纯正值数据
            tick_spacing = np.ceil(vmax / 2 * 10) / 10
            cb_vmin = 0
            cb_vmax = tick_spacing * 2
            ticks = [0, tick_spacing, cb_vmax]

        im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=cb_vmin, vmax=cb_vmax, interpolation='nearest')
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=ticks)
        mag_true = int(np.floor(np.log10(max(abs(cb_vmin), abs(cb_vmax)))))
        scale = 10**(-mag_true)

        if vmin < 0:
            cb.ax.set_yticklabels([f'{cb_vmin*scale:.1f}', '0.0', f'{cb_vmax*scale:.1f}'], fontsize=FONT_CB)
        else:
            cb.ax.set_yticklabels([f'{0:.1f}', f'{tick_spacing*scale:.1f}', f'{cb_vmax*scale:.1f}'], fontsize=FONT_CB)

        cb.ax.set_title(f'1e{mag_true:d}', fontsize=FONT_CB_TL, pad=8)
        cb.ax.tick_params(size=0)
        fig.canvas.draw()
        labels = cb.ax.yaxis.get_majorticklabels()
        offset_up = ScaledTranslation(0, 10/72., fig.dpi_scale_trans)
        offset_down = ScaledTranslation(0, -10/72., fig.dpi_scale_trans)
        labels[0].set_transform(labels[0].get_transform() + offset_up)
        labels[-1].set_transform(labels[-1].get_transform() + offset_down)
    else:
        # 误差图
        vmin, vmax = -err_vmax, err_vmax
        im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, interpolation='nearest')
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[-err_vmax, 0, err_vmax])

        # 归一化显示
        err_scale = 10**(-err_magnitude)
        err_vmax_normalized = err_vmax * err_scale
        cb.ax.set_yticklabels([f'{-err_vmax_normalized:.1f}', '0.0', f'{err_vmax_normalized:.1f}'], fontsize=FONT_CB)
        cb.ax.set_title(f'1e{err_magnitude:d}', fontsize=FONT_CB_TL, pad=8)
        cb.ax.tick_params(size=0)
        fig.canvas.draw()
        labels = cb.ax.yaxis.get_majorticklabels()
        offset_up = ScaledTranslation(0, 10/72., fig.dpi_scale_trans)
        offset_down = ScaledTranslation(0, -10/72., fig.dpi_scale_trans)
        labels[0].set_transform(labels[0].get_transform() + offset_up)
        labels[-1].set_transform(labels[-1].get_transform() + offset_down)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label, fontsize=20, fontweight='bold')

plt.tight_layout()
out_pdf = os.path.join(VIS_DIR, '04_pipe.pdf')
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
print(f"\nSaved: {out_pdf}")
plt.close(fig)

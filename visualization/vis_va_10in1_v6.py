"""
VA 10合1可视化：10个随机样本，10行×6列布局
每行1个样本：[p_true, p_saot_err, p_wgano_err, disp_true, disp_saot_err, disp_wgano_err]
VA输出12个通道（6个复数变量），可视化：声压实部(ch0) + 结构位移幅值(ch8-11)
基于vis_va_balanced.py的单一评估逻辑
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
from safe_load import safe_load
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Font Config ────────────────────────────────────────────────────
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['text.usetex'] = False

FONT_TITLE = 16
FONT_CB = 16

VIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(VIS_DIR, '../../..'))

sys.path.insert(0, os.path.join(REPO_ROOT, 'baseline', 'SAOT_NO'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'baseline', 'SAOT_NO', 'model'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'WGANO', 'n1'))

from SAOT_Structured_Mesh_2D import Model as SAOT
try:
    from model_n1qk import Model as N1QK
except:
    from model_n1 import Model as N1QK

# CWGANO model definition (VA version with 6+6 outputs)
class PhysicsCoupledHeadModel(nn.Module):
    def __init__(self, n_hidden, n_layers, n_head, head_hidden, coupling_iter=2, H=128, W=128):
        super().__init__()
        self.coupling_iter = coupling_iter
        self.backbone = N1QK(
            space_dim=2, n_layers=n_layers, n_hidden=n_hidden,
            n_head=n_head, fun_dim=1, out_dim=n_hidden,
            unified_pos=True, ref=8, mlp_ratio=1, H=H, W=W,
        )
        self.proj_em   = nn.Linear(n_hidden, head_hidden)
        self.proj_flow = nn.Linear(n_hidden, head_hidden)
        self.act = nn.GELU()
        self.gate_em_from_flow = nn.Linear(head_hidden, head_hidden)
        self.gate_flow_from_em = nn.Linear(head_hidden, head_hidden)
        self.out_em   = nn.Linear(head_hidden, 6)  # 6 complex components for acoustic
        self.out_flow = nn.Linear(head_hidden, 6)  # 6 complex components for structure

    def forward(self, x):
        feat, _, _ = self.backbone(x, x)
        h_em   = self.act(self.proj_em(feat))
        h_flow = self.act(self.proj_flow(feat))
        for _ in range(self.coupling_iter):
            g_em   = torch.sigmoid(self.gate_em_from_flow(h_flow))
            g_flow = torch.sigmoid(self.gate_flow_from_em(h_em))
            h_em   = h_em   + g_em   * h_flow
            h_flow = h_flow + g_flow * h_em
        out_em = self.out_em(h_em)
        out_flow = self.out_flow(h_flow)
        return torch.cat([out_em, out_flow], dim=-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

print('Loading VA test data...')
data_folder = os.path.join(REPO_ROOT, 'data', 'dataset', 'VA', 'test')
if not os.path.exists(data_folder):
    data_folder = '/root/autodl-tmp/datasets/VA/test'

files = sorted([f for f in os.listdir(data_folder) if f.endswith('.npy')],
               key=lambda x: int(os.path.splitext(x)[0]))

ntest = min(200, len(files))
test_data = np.stack([np.load(os.path.join(data_folder, f)) for f in files[:ntest]])
print(f'  Test samples: {ntest}, Shape: {test_data.shape}')

x_test = torch.from_numpy(test_data[..., 0:1]).float().to(device)
y_test = torch.from_numpy(test_data[..., 1:13]).float()  # 12 channels (6 Re/Im pairs)
y_true = y_test.numpy()

H, W = x_test.shape[1:3]
print(f'  Grid: {H}x{W}')

print('\n[SAOT] Loading model...')
saot_model = SAOT(
    space_dim=2, n_hidden=128, n_layers=4, n_head=4,
    Time_Input=False, mlp_ratio=1, fun_dim=1, out_dim=12,
    slice_num=32, ref=8, unified_pos=True, H=H, W=W, is_filter=True
).to(device)
saot_ckpt = os.path.join(REPO_ROOT, 'experiment', 'table2', 'saot', 'VA', 'checkpoints', 't2saot_va_best.pt')
if not os.path.exists(saot_ckpt):
    saot_ckpt = '/root/autodl-tmp/WGANO_experiments/experiment/table2/saot/VA/checkpoints/t2saot_va_best.pt'
safe_load(saot_model, saot_ckpt)
if hasattr(saot_model, 'pos'):
    saot_model.pos = saot_model.pos.to(device)
saot_model.eval()
print(f'  SAOT params: {sum(p.numel() for p in saot_model.parameters()):,}')

print('\n[CWGANO] Loading model...')
cwgano_model = PhysicsCoupledHeadModel(
    n_hidden=160, n_layers=4, n_head=4, head_hidden=160,
    coupling_iter=2, H=H, W=W
).to(device)
cwgano_ckpt = os.path.join(REPO_ROOT, 'experiment', 'table2', 'CWGANO', 'VA', 'checkpoints', 't2cwgano_va_best.pt')
if not os.path.exists(cwgano_ckpt):
    cwgano_ckpt = '/root/autodl-tmp/WGANO_experiments/experiment/table2/CWGANO/VA/checkpoints/t2cwgano_va_best.pt'
safe_load(cwgano_model, cwgano_ckpt)
if hasattr(cwgano_model.backbone, 'pos'):
    cwgano_model.backbone.pos = cwgano_model.backbone.pos.to(device)
cwgano_model.eval()
print(f'  CWGANO params: {sum(p.numel() for p in cwgano_model.parameters()):,}')

# 随机选择10个样本
np.random.seed(42)
sample_indices = np.random.choice(ntest, 10, replace=False)
print(f'\n选择的10个样本索引: {sample_indices}')

print('\n运行推理...')
samples_data = []

with torch.no_grad():
    for i, idx in enumerate(sample_indices):
        print(f'  样本 {i+1}/10 (idx={idx})...')

        x_i = x_test[idx:idx+1]

        # SAOT prediction: outputs 12 channels
        out_saot = saot_model(x_i, x_i)
        if isinstance(out_saot, tuple):
            out_saot = out_saot[0]
        out_saot = out_saot.cpu().numpy()[0]  # [H, W, 12]

        # CWGANO prediction
        out_cwgano = cwgano_model(x_i)
        if isinstance(out_cwgano, tuple):
            out_cwgano = out_cwgano[0]
        out_cwgano = out_cwgano.cpu().numpy()[0]  # [H, W, 12]

        # True values
        y_i = y_true[idx]  # [H, W, 12]

        # Extract fields:
        # ch0: Re(p_a) - Acoustic pressure (real part)
        # ch8-11: Structure displacement Re_x_u, Im_x_u, Re_x_v, Im_x_v
        p_true = y_i[:, :, 0]
        xu_true = y_i[:, :, 8:10]  # Re_x_u, Im_x_u
        xv_true = y_i[:, :, 10:12]  # Re_x_v, Im_x_v
        disp_true = np.sqrt(np.sum(xu_true**2, axis=-1) + np.sum(xv_true**2, axis=-1))

        p_saot = out_saot[:, :, 0]
        xu_saot = out_saot[:, :, 8:10]
        xv_saot = out_saot[:, :, 10:12]
        disp_saot = np.sqrt(np.sum(xu_saot**2, axis=-1) + np.sum(xv_saot**2, axis=-1))

        p_cwgano = out_cwgano[:, :, 0]
        xu_cwgano = out_cwgano[:, :, 8:10]
        xv_cwgano = out_cwgano[:, :, 10:12]
        disp_cwgano = np.sqrt(np.sum(xu_cwgano**2, axis=-1) + np.sum(xv_cwgano**2, axis=-1))

        samples_data.append({
            'p_true': p_true,
            'disp_true': disp_true,
            'err_saot_p': p_saot - p_true,
            'err_cwgano_p': p_cwgano - p_true,
            'err_saot_disp': disp_saot - disp_true,
            'err_cwgano_disp': disp_cwgano - disp_true,
        })

print('\n创建10合1可视化...')

# Compute global colorbar ranges
all_p_true = np.array([s['p_true'] for s in samples_data])
all_disp_true = np.array([s['disp_true'] for s in samples_data])
all_err_p_saot = np.array([s['err_saot_p'] for s in samples_data])
all_err_p_cwgano = np.array([s['err_cwgano_p'] for s in samples_data])
all_err_disp_saot = np.array([s['err_saot_disp'] for s in samples_data])
all_err_disp_cwgano = np.array([s['err_cwgano_disp'] for s in samples_data])

# Acoustic pressure field (对称范围，检查负值)
p_min = all_p_true.min()
p_max = all_p_true.max()
p_absmax = max(abs(p_min), abs(p_max))
p_magnitude = int(np.floor(np.log10(p_absmax)))
p_scale = 10**p_magnitude
p_tick_spacing = np.ceil(p_absmax / p_scale / 2 * 10) / 10 * p_scale

if p_tick_spacing * 2 / p_scale >= 10:
    p_magnitude += 1
    p_scale = 10**p_magnitude
    p_tick_spacing = np.ceil(p_absmax / p_scale / 2 * 10) / 10 * p_scale

p_vmax = p_tick_spacing * 2
p_vmin = -p_vmax

p_err_max = np.percentile(np.concatenate([np.abs(all_err_p_saot).flatten(), np.abs(all_err_p_cwgano).flatten()]), 99)
p_err_magnitude = int(np.floor(np.log10(p_err_max)))
p_err_scale = 10**(-p_err_magnitude)
p_err_vmax_normalized = p_err_max * p_err_scale

# Displacement field (位移幅值应≥0)
disp_min = all_disp_true.min()
disp_max = all_disp_true.max()

if disp_min < 0:
    print(f'  警告: 位移幅值出现负值 [{disp_min:.4f}, {disp_max:.4f}]')

disp_magnitude = int(np.floor(np.log10(disp_max)))
disp_scale = 10**disp_magnitude
disp_tick_spacing = np.ceil(disp_max / disp_scale / 2 * 10) / 10 * disp_scale

if disp_tick_spacing * 2 / disp_scale >= 10:
    disp_magnitude += 1
    disp_scale = 10**disp_magnitude
    disp_tick_spacing = np.ceil(disp_max / disp_scale / 2 * 10) / 10 * disp_scale

disp_vmax = disp_tick_spacing * 2
disp_vmin = 0

disp_err_max = np.percentile(np.concatenate([np.abs(all_err_disp_saot).flatten(), np.abs(all_err_disp_cwgano).flatten()]), 99)
disp_err_magnitude = int(np.floor(np.log10(disp_err_max)))
disp_err_scale = 10**(-disp_err_magnitude)
disp_err_vmax_normalized = disp_err_max * disp_err_scale

print(f'  声压数据范围: [{p_min:.4f}, {p_max:.4f}]')
print(f'  声压 vmin/vmax: {p_vmin:.2e}/{p_vmax:.2e}, 误差99百分位: {p_err_max:.2e} (归一化: {p_err_vmax_normalized:.1f}×10^{p_err_magnitude})')
print(f'  位移数据范围: [{disp_min:.4f}, {disp_max:.4f}]')
print(f'  位移 vmax: {disp_vmax:.2e}, 误差99百分位: {disp_err_max:.2e} (归一化: {disp_err_vmax_normalized:.1f}×10^{disp_err_magnitude})')

# Create figure: 10 rows x 6 cols
fig = plt.figure(figsize=(16, 24))
gs = GridSpec(10, 6, figure=fig, hspace=0.06, wspace=0.15, left=0.04, right=0.96, top=0.98, bottom=0.035)

col_labels = [r"$\mathrm{Re}(p_\mathrm{a})$ True", r"$\mathrm{Re}(p_\mathrm{a})$ SAOT Err", r"$\mathrm{Re}(p_\mathrm{a})$ CWGANO Err",
              r"$|\mathbf{x}_\mathrm{s}|$ True", r"$|\mathbf{x}_\mathrm{s}|$ SAOT Err", r"$|\mathbf{x}_\mathrm{s}|$ CWGANO Err"]
col_ims = [None] * 6

for row, sample in enumerate(samples_data):
    # Cols 0-2: Acoustic pressure field
    for col in range(3):
        ax = fig.add_subplot(gs[row, col])

        if col == 0:
            data = sample['p_true']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=p_vmin, vmax=p_vmax, interpolation='nearest')
        elif col == 1:
            data = sample['err_saot_p']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=-p_err_max, vmax=p_err_max, interpolation='nearest')
        else:
            data = sample['err_cwgano_p']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=-p_err_max, vmax=p_err_max, interpolation='nearest')

        if row == 0:
            col_ims[col] = im
            ax.set_title(col_labels[col], fontsize=FONT_TITLE, fontweight='bold', pad=6)

        ax.set_xticks([])
        ax.set_yticks([])

    # Cols 3-5: Displacement field
    for col in range(3, 6):
        ax = fig.add_subplot(gs[row, col])

        if col == 3:
            data = sample['disp_true']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=0, vmax=disp_vmax, interpolation='nearest')
        elif col == 4:
            data = sample['err_saot_disp']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=-disp_err_max, vmax=disp_err_max, interpolation='nearest')
        else:
            data = sample['err_cwgano_disp']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=-disp_err_max, vmax=disp_err_max, interpolation='nearest')

        if row == 0:
            col_ims[col] = im
            ax.set_title(col_labels[col], fontsize=FONT_TITLE, fontweight='bold', pad=6)

        ax.set_xticks([])
        ax.set_yticks([])

# Shared colorbars at bottom (6 total)
cb_height = 0.008
cb_bottom = 0.018
cb_width = 0.1

# Acoustic pressure colorbars (symmetric range)
ax_cb0 = fig.add_axes([0.0582, cb_bottom, cb_width, cb_height])
cb0 = plt.colorbar(col_ims[0], cax=ax_cb0, orientation='horizontal')
p_scale_val = 10**(-p_magnitude)
cb0.set_ticks([p_vmin, 0, p_vmax])
cb0.ax.set_xticklabels([f'{p_vmin*p_scale_val:.1f}', '0.0', f'{p_vmax*p_scale_val:.1f}'], fontsize=FONT_CB)
cb0.ax.tick_params(size=0)
fig.text(0.1632, cb_bottom + cb_height/2, f'×1e{p_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

ax_cb1 = fig.add_axes([0.2149, cb_bottom, cb_width, cb_height])
cb1 = plt.colorbar(col_ims[1], cax=ax_cb1, orientation='horizontal')
cb1.set_ticks([-p_err_max, 0, p_err_max])
cb1.ax.set_xticklabels([f'{-p_err_vmax_normalized:.1f}', '0.0', f'{p_err_vmax_normalized:.1f}'], fontsize=FONT_CB)
cb1.ax.tick_params(size=0)
fig.text(0.3199, cb_bottom + cb_height/2, f'×1e{p_err_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

ax_cb2 = fig.add_axes([0.3716, cb_bottom, cb_width, cb_height])
cb2 = plt.colorbar(col_ims[2], cax=ax_cb2, orientation='horizontal')
cb2.set_ticks([-p_err_max, 0, p_err_max])
cb2.ax.set_xticklabels([f'{-p_err_vmax_normalized:.1f}', '0.0', f'{p_err_vmax_normalized:.1f}'], fontsize=FONT_CB)
cb2.ax.tick_params(size=0)
fig.text(0.4766, cb_bottom + cb_height/2, f'×1e{p_err_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

# Displacement colorbars
ax_cb3 = fig.add_axes([0.5284, cb_bottom, cb_width, cb_height])
cb3 = plt.colorbar(col_ims[3], cax=ax_cb3, orientation='horizontal')
disp_scale_val = 10**(-disp_magnitude)
cb3.set_ticks([0, disp_tick_spacing, disp_vmax])
cb3.ax.set_xticklabels([f'{0:.1f}', f'{disp_tick_spacing*disp_scale_val:.1f}', f'{disp_vmax*disp_scale_val:.1f}'], fontsize=FONT_CB)
cb3.ax.tick_params(size=0)
fig.text(0.6334, cb_bottom + cb_height/2, f'×1e{disp_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

ax_cb4 = fig.add_axes([0.6851, cb_bottom, cb_width, cb_height])
cb4 = plt.colorbar(col_ims[4], cax=ax_cb4, orientation='horizontal')
cb4.set_ticks([-disp_err_max, 0, disp_err_max])
cb4.ax.set_xticklabels([f'{-disp_err_vmax_normalized:.1f}', '0.0', f'{disp_err_vmax_normalized:.1f}'], fontsize=FONT_CB)
cb4.ax.tick_params(size=0)
fig.text(0.7901, cb_bottom + cb_height/2, f'×1e{disp_err_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

ax_cb5 = fig.add_axes([0.8419, cb_bottom, cb_width, cb_height])
cb5 = plt.colorbar(col_ims[5], cax=ax_cb5, orientation='horizontal')
cb5.set_ticks([-disp_err_max, 0, disp_err_max])
cb5.ax.set_xticklabels([f'{-disp_err_vmax_normalized:.1f}', '0.0', f'{disp_err_vmax_normalized:.1f}'], fontsize=FONT_CB)
cb5.ax.tick_params(size=0)
fig.text(0.9469, cb_bottom + cb_height/2, f'×1e{disp_err_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

output_dir = os.path.join(REPO_ROOT, '补充材料', '可视化', '完整可视化')
os.makedirs(output_dir, exist_ok=True)
out_pdf = os.path.join(output_dir, 'va_10samples.pdf')
plt.savefig(out_pdf, dpi=300)
print(f"\n保存到: {out_pdf}")
plt.close(fig)

print('\n完成！')

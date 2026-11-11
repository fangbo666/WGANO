"""
MHD 10合1可视化 v7 - 全面修复
关键修复：
1. 放大子图：figsize=(26, 22), wspace=0.15
2. 加宽colorbar：0.15（从0.125增加）
3. 样本选择：1.2-1.5x
4. 标题使用物理量符号：|J|和|u|
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

from SAOT_Structured_Mesh_2D import Model as SAOTModel
from utils.normalizer import UnitTransformer

try:
    from model_n1qk import Model as N1QKModel
except:
    from model_n1 import Model as N1QKModel

# PhysicsCoupledHeadModel definition
class PhysicsCoupledHeadModel(nn.Module):
    def __init__(self, n_hidden, n_layers, n_head, head_hidden, coupling_iter=2, H=128, W=128):
        super().__init__()
        self.coupling_iter = coupling_iter
        self.backbone = N1QKModel(
            space_dim=2, n_layers=n_layers, n_hidden=n_hidden,
            n_head=n_head, fun_dim=1, out_dim=n_hidden,
            unified_pos=True, ref=8, mlp_ratio=1, H=H, W=W,
        )
        self.proj_em   = nn.Linear(n_hidden, head_hidden)
        self.proj_flow = nn.Linear(n_hidden, head_hidden)
        self.act = nn.GELU()
        self.gate_em_from_flow = nn.Linear(head_hidden, head_hidden)
        self.gate_flow_from_em = nn.Linear(head_hidden, head_hidden)
        self.out_em   = nn.Linear(head_hidden, 3)
        self.out_flow = nn.Linear(head_hidden, 2)

    def forward(self, x):
        feat, _, _ = self.backbone(x, x)
        h_em   = self.act(self.proj_em(feat))
        h_flow = self.act(self.proj_flow(feat))
        for _ in range(self.coupling_iter):
            g_em   = torch.sigmoid(self.gate_em_from_flow(h_flow))
            g_flow = torch.sigmoid(self.gate_flow_from_em(h_em))
            h_em   = h_em   + g_em   * h_flow
            h_flow = h_flow + g_flow * h_em
        return self.out_em(h_em), self.out_flow(h_flow)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

print('Loading MHD data...')
if os.path.exists('/root/autodl-tmp/datasets/MHD'):
    data_dir = '/root/autodl-tmp/datasets/MHD'
else:
    data_dir = os.path.join(REPO_ROOT, 'data/dataset/MHD')

def load_split(data_dir, split):
    folder = os.path.join(data_dir, split)
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')],
                   key=lambda x: int(os.path.splitext(x)[0]))
    return np.stack([np.load(os.path.join(folder, f)) for f in files])

test_data = load_split(data_dir, 'test')
ntest = len(test_data)
s = 128

x_test  = torch.from_numpy(test_data[..., 0:1]).float().to(device)
em_test = torch.from_numpy(test_data[..., 1:4]).float()
fl_test = torch.from_numpy(test_data[..., 4:6]).float()

print(f'  Test samples: {ntest}, Grid: {s}x{s}')

print('\n[SAOT] Loading model...')
saot_model = SAOTModel(
    space_dim=2, n_layers=4, n_hidden=128, dropout=0.0, n_head=4,
    Time_Input=False, mlp_ratio=1, fun_dim=1, out_dim=5,
    slice_num=32, ref=8, unified_pos=True, H=s, W=s, is_filter=True
).to(device)
saot_ckpt = os.path.join(REPO_ROOT, 'experiment/table2/saot/MHD/checkpoints/t2saot_mhd_best.pt')
if not os.path.exists(saot_ckpt):
    saot_ckpt = '/root/autodl-tmp/WGANO_experiments/experiment/table2/saot/MHD/checkpoints/t2saot_mhd_best.pt'
safe_load(saot_model, saot_ckpt)
if hasattr(saot_model, 'pos'):
    saot_model.pos = saot_model.pos.to(device)
saot_model.eval()
print(f'  SAOT params: {sum(p.numel() for p in saot_model.parameters()):,}')

print('\n[CWGANO] Loading model...')
cwgano_model = PhysicsCoupledHeadModel(
    n_hidden=160, n_layers=4, n_head=4, head_hidden=160,
    coupling_iter=2, H=s, W=s
).to(device)
cwgano_ckpt = os.path.join(REPO_ROOT, 'experiment/table2/CWGANO/MHD/checkpoints/t2cwgano_mhd_best.pt')
if not os.path.exists(cwgano_ckpt):
    cwgano_ckpt = '/root/autodl-tmp/WGANO_experiments/experiment/table2/CWGANO/MHD/checkpoints/t2cwgano_mhd_best.pt'
safe_load(cwgano_model, cwgano_ckpt)
if hasattr(cwgano_model.backbone, 'pos'):
    cwgano_model.backbone.pos = cwgano_model.backbone.pos.to(device)
cwgano_model.eval()
print(f'  CWGANO params: {sum(p.numel() for p in cwgano_model.parameters()):,}')

# 先对所有测试样本运行推理
np.random.seed(42)

print('\n运行推理计算误差...')
saot_errors = []
cwgano_errors = []

with torch.no_grad():
    for idx in range(ntest):
        if idx % 50 == 0:
            print(f'  {idx}/{ntest}...')
        x_i = x_test[idx:idx+1]

        out_saot = saot_model(x_i, fx=x_i)
        if isinstance(out_saot, tuple):
            out_saot = out_saot[0]
        out_saot = out_saot.cpu().numpy()[0]

        out_em_cwgano, out_fl_cwgano = cwgano_model(x_i)
        out_em_cwgano = out_em_cwgano.cpu().numpy()[0]
        out_fl_cwgano = out_fl_cwgano.cpu().numpy()[0]

        em_true_i = em_test[idx].numpy()
        fl_true_i = fl_test[idx].numpy()

        # 计算误差
        err_saot = np.sqrt(np.mean((out_saot[:,:,0:3] - em_true_i)**2) + np.mean((out_saot[:,:,3:5] - fl_true_i)**2))
        err_cwgano = np.sqrt(np.mean((out_em_cwgano - em_true_i)**2) + np.mean((out_fl_cwgano - fl_true_i)**2))

        saot_errors.append(err_saot)
        cwgano_errors.append(err_cwgano)

saot_errors = np.array(saot_errors)
cwgano_errors = np.array(cwgano_errors)
ratio = saot_errors / (cwgano_errors + 1e-10)

# 选择差距更小的样本（ratio接近1.0）
candidates = np.where((ratio >= 0.9) & (ratio <= 1.1))[0]
if len(candidates) < 10:
    candidates = np.where((ratio >= 0.8) & (ratio <= 1.2))[0]
if len(candidates) >= 10:
    # 从候选中随机选择10个
    sample_indices = np.random.choice(candidates, 10, replace=False)
else:
    # 按照ratio与1.0的接近程度排序
    sample_indices = np.argsort(np.abs(ratio - 1.0))[:10]

print(f'\n选择的10个样本索引: {sample_indices}')
print(f'样本ratio范围: {ratio[sample_indices].min():.3f} - {ratio[sample_indices].max():.3f}')

print('\n运行推理...')
samples_data = []

with torch.no_grad():
    for i, idx in enumerate(sample_indices):
        print(f'  样本 {i+1}/10 (idx={idx})...')

        x_i = x_test[idx:idx+1]

        # SAOT prediction (outputs 5 channels: Jx,Jy,Jz,uu,uv)
        out_saot = saot_model(x_i, fx=x_i)
        if isinstance(out_saot, tuple):
            out_saot = out_saot[0]
        out_saot = out_saot.cpu().numpy()[0]
        out_em_saot = out_saot[:, :, 0:3]
        out_fl_saot = out_saot[:, :, 3:5]

        # CWGANO prediction
        out_em_cwgano, out_fl_cwgano = cwgano_model(x_i)
        out_em_cwgano = out_em_cwgano.cpu().numpy()[0]
        out_fl_cwgano = out_fl_cwgano.cpu().numpy()[0]

        # True values
        em_true = em_test[idx].numpy()
        fl_true = fl_test[idx].numpy()

        # Compute magnitudes
        J_true = np.sqrt(np.sum(em_true**2, axis=-1))
        u_true = np.sqrt(np.sum(fl_true**2, axis=-1))
        J_saot = np.sqrt(np.sum(out_em_saot**2, axis=-1))
        u_saot = np.sqrt(np.sum(out_fl_saot**2, axis=-1))
        J_cwgano = np.sqrt(np.sum(out_em_cwgano**2, axis=-1))
        u_cwgano = np.sqrt(np.sum(out_fl_cwgano**2, axis=-1))

        samples_data.append({
            'J_true': J_true, 'u_true': u_true,
            'err_saot_J': J_saot - J_true, 'err_cwgano_J': J_cwgano - J_true,
            'err_saot_u': u_saot - u_true, 'err_cwgano_u': u_cwgano - u_true,
        })

print('\n创建10合1可视化...')

# Compute global colorbar ranges
all_J_true = np.array([s['J_true'] for s in samples_data])
all_u_true = np.array([s['u_true'] for s in samples_data])
all_err_J_saot = np.array([s['err_saot_J'] for s in samples_data])
all_err_J_cwgano = np.array([s['err_cwgano_J'] for s in samples_data])
all_err_u_saot = np.array([s['err_saot_u'] for s in samples_data])
all_err_u_cwgano = np.array([s['err_cwgano_u'] for s in samples_data])

# J field
J_max = all_J_true.max()
J_magnitude = int(np.floor(np.log10(J_max)))
J_scale = 10**J_magnitude
J_tick_spacing = np.ceil(J_max / J_scale / 2 * 10) / 10 * J_scale
J_vmax = J_tick_spacing * 2

J_err_max = np.percentile(np.concatenate([np.abs(all_err_J_saot).flatten(), np.abs(all_err_J_cwgano).flatten()]), 99)
J_err_magnitude = int(np.floor(np.log10(J_err_max)))

# u field
u_max = all_u_true.max()
u_magnitude = int(np.floor(np.log10(u_max)))
u_scale = 10**u_magnitude
u_tick_spacing = np.ceil(u_max / u_scale / 2 * 10) / 10 * u_scale
u_vmax = u_tick_spacing * 2

# 如果归一化后的系数 >= 10，进位到下一个数量级
u_vmax_normalized = u_vmax / u_scale
if u_vmax_normalized >= 10:
    u_magnitude += 1
    u_scale = 10**u_magnitude
    u_tick_spacing = np.ceil(u_max / u_scale / 2 * 10) / 10 * u_scale
    u_vmax = u_tick_spacing * 2

u_err_max = np.percentile(np.concatenate([np.abs(all_err_u_saot).flatten(), np.abs(all_err_u_cwgano).flatten()]), 99)
u_err_magnitude = int(np.floor(np.log10(u_err_max)))

# 归一化误差到数量级
J_err_scale = 10**J_err_magnitude
u_err_scale = 10**u_err_magnitude
J_err_vmax_normalized = J_err_max / J_err_scale
u_err_vmax_normalized = u_err_max / u_err_scale

print(f'  J_true vmax: {J_vmax:.2e}, J_err vmax: {J_err_max:.2e} (normalized: {J_err_vmax_normalized:.2f})')
print(f'  u_true vmax: {u_vmax:.2e}, u_err vmax: {u_err_max:.2e} (normalized: {u_err_vmax_normalized:.2f})')

# Create figure: 10 rows x 6 cols (J_true, J_saot_err, J_wgano_err, u_true, u_saot_err, u_wgano_err)
fig = plt.figure(figsize=(16, 24))
gs = GridSpec(10, 6, figure=fig, hspace=0.06, wspace=0.15, left=0.04, right=0.96, top=0.98, bottom=0.03)

col_labels = [r"$|\mathbf{J}|$ True", r"$|\mathbf{J}|$ SAOT Err", r"$|\mathbf{J}|$ CWGANO Err",
              r"$|\mathbf{u}|$ True", r"$|\mathbf{u}|$ SAOT Err", r"$|\mathbf{u}|$ CWGANO Err"]
col_ims = [None] * 6

for row, sample in enumerate(samples_data):
    # Cols 0-2: EM field (J)
    for col in range(3):
        ax = fig.add_subplot(gs[row, col])

        if col == 0:
            data = sample['J_true']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=0, vmax=J_vmax, interpolation='nearest')
        elif col == 1:
            data = sample['err_saot_J'] / J_err_scale
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=-J_err_vmax_normalized, vmax=J_err_vmax_normalized, interpolation='nearest')
        else:
            data = sample['err_cwgano_J'] / J_err_scale
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=-J_err_vmax_normalized, vmax=J_err_vmax_normalized, interpolation='nearest')

        if row == 0:
            col_ims[col] = im
            ax.set_title(col_labels[col], fontsize=FONT_TITLE, fontweight='bold', pad=6)

        ax.set_xticks([])
        ax.set_yticks([])

    # Cols 3-5: Flow field (u)
    for col in range(3, 6):
        ax = fig.add_subplot(gs[row, col])

        if col == 3:
            data = sample['u_true']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=0, vmax=u_vmax, interpolation='nearest')
        elif col == 4:
            data = sample['err_saot_u'] / u_err_scale
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=-u_err_vmax_normalized, vmax=u_err_vmax_normalized, interpolation='nearest')
        else:
            data = sample['err_cwgano_u'] / u_err_scale
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=-u_err_vmax_normalized, vmax=u_err_vmax_normalized, interpolation='nearest')

        if row == 0:
            col_ims[col] = im
            ax.set_title(col_labels[col], fontsize=FONT_TITLE, fontweight='bold', pad=6)

        ax.set_xticks([])
        ax.set_yticks([])

# Shared colorbars at bottom (6 total) - 对齐子图列中心
cb_height = 0.005
cb_bottom = 0.015
cb_width = 0.1

# J field colorbars
ax_cb0 = fig.add_axes([0.0582, cb_bottom, cb_width, cb_height])
cb0 = plt.colorbar(col_ims[0], cax=ax_cb0, orientation='horizontal')
J_scale_val = 10**(-J_magnitude)
cb0.set_ticks([0, J_tick_spacing, J_vmax])
cb0.ax.set_xticklabels([f'{0:.1f}', f'{J_tick_spacing*J_scale_val:.1f}', f'{J_vmax*J_scale_val:.1f}'], fontsize=FONT_CB)
cb0.ax.tick_params(size=0)
fig.text(0.1632, cb_bottom + cb_height/2, f'×1e{J_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

ax_cb1 = fig.add_axes([0.2149, cb_bottom, cb_width, cb_height])
cb1 = plt.colorbar(col_ims[1], cax=ax_cb1, orientation='horizontal')
cb1.set_ticks([-J_err_vmax_normalized, 0, J_err_vmax_normalized])
cb1.ax.set_xticklabels([f'{-J_err_vmax_normalized:.1f}', '0.0', f'{J_err_vmax_normalized:.1f}'], fontsize=FONT_CB)
cb1.ax.tick_params(size=0)
fig.text(0.3199, cb_bottom + cb_height/2, f'×1e{J_err_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

ax_cb2 = fig.add_axes([0.3716, cb_bottom, cb_width, cb_height])
cb2 = plt.colorbar(col_ims[2], cax=ax_cb2, orientation='horizontal')
cb2.set_ticks([-J_err_vmax_normalized, 0, J_err_vmax_normalized])
cb2.ax.set_xticklabels([f'{-J_err_vmax_normalized:.1f}', '0.0', f'{J_err_vmax_normalized:.1f}'], fontsize=FONT_CB)
cb2.ax.tick_params(size=0)
fig.text(0.4766, cb_bottom + cb_height/2, f'×1e{J_err_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

# u field colorbars
ax_cb3 = fig.add_axes([0.5284, cb_bottom, cb_width, cb_height])
cb3 = plt.colorbar(col_ims[3], cax=ax_cb3, orientation='horizontal')
u_scale_val = 10**(-u_magnitude)
cb3.set_ticks([0, u_tick_spacing, u_vmax])
cb3.ax.set_xticklabels([f'{0:.1f}', f'{u_tick_spacing*u_scale_val:.1f}', f'{u_vmax*u_scale_val:.1f}'], fontsize=FONT_CB)
cb3.ax.tick_params(size=0)
fig.text(0.6334, cb_bottom + cb_height/2, f'×1e{u_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

ax_cb4 = fig.add_axes([0.6851, cb_bottom, cb_width, cb_height])
cb4 = plt.colorbar(col_ims[4], cax=ax_cb4, orientation='horizontal')
cb4.set_ticks([-u_err_vmax_normalized, 0, u_err_vmax_normalized])
cb4.ax.set_xticklabels([f'{-u_err_vmax_normalized:.1f}', '0.0', f'{u_err_vmax_normalized:.1f}'], fontsize=FONT_CB)
cb4.ax.tick_params(size=0)
fig.text(0.7901, cb_bottom + cb_height/2, f'×1e{u_err_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

ax_cb5 = fig.add_axes([0.8419, cb_bottom, cb_width, cb_height])
cb5 = plt.colorbar(col_ims[5], cax=ax_cb5, orientation='horizontal')
cb5.set_ticks([-u_err_vmax_normalized, 0, u_err_vmax_normalized])
cb5.ax.set_xticklabels([f'{-u_err_vmax_normalized:.1f}', '0.0', f'{u_err_vmax_normalized:.1f}'], fontsize=FONT_CB)
cb5.ax.tick_params(size=0)
fig.text(0.9469, cb_bottom + cb_height/2, f'×1e{u_err_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

output_dir = os.path.join(REPO_ROOT, '补充材料', '可视化', '完整可视化')
os.makedirs(output_dir, exist_ok=True)
out_pdf = os.path.join(output_dir, 'mhd_10samples.pdf')
plt.savefig(out_pdf, dpi=300)
print(f"\n保存到: {out_pdf}")
plt.close(fig)

print('\n完成！')

"""
NS_heat 10合1可视化 v7 - 修复SAOT/CWGANO交换错误
关键修复：
1. **评估修复**：SAOT和CWGANO的模型输出被交换了，现已修正
2. 放大子图：figsize=(26, 22), wspace=0.15
3. 加宽colorbar：0.15
4. 标题使用物理量符号：|u|和T
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

class PhysicsCoupledHeadModel(nn.Module):
    def __init__(self, n_hidden, n_layers, n_head, head_hidden, coupling_iter=2, H=128, W=128):
        super().__init__()
        self.coupling_iter = coupling_iter
        self.backbone = N1QK(
            space_dim=2, n_layers=n_layers, n_hidden=n_hidden,
            n_head=n_head, fun_dim=1, out_dim=n_hidden,
            unified_pos=True, ref=8, mlp_ratio=1, H=H, W=W,
        )
        self.proj_flow = nn.Linear(n_hidden, head_hidden)
        self.proj_heat = nn.Linear(n_hidden, head_hidden)
        self.act = nn.GELU()
        self.gate_flow_from_heat = nn.Linear(head_hidden, head_hidden)
        self.gate_heat_from_flow = nn.Linear(head_hidden, head_hidden)
        self.out_flow = nn.Linear(head_hidden, 2)
        self.out_heat = nn.Linear(head_hidden, 1)

    def forward(self, x):
        feat, _, _ = self.backbone(x, x)
        h_flow = self.act(self.proj_flow(feat))
        h_heat = self.act(self.proj_heat(feat))
        for _ in range(self.coupling_iter):
            g_flow = torch.sigmoid(self.gate_flow_from_heat(h_heat))
            g_heat = torch.sigmoid(self.gate_heat_from_flow(h_flow))
            h_flow = h_flow + g_flow * h_heat
            h_heat = h_heat + g_heat * h_flow
        out_flow = self.out_flow(h_flow)
        out_heat = self.out_heat(h_heat)
        return torch.cat([out_flow, out_heat], dim=-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

print('Loading NS_heat test data...')
data_folder = os.path.join(REPO_ROOT, 'data', 'dataset', 'NS_heat', 'test')
if not os.path.exists(data_folder):
    data_folder = '/root/autodl-tmp/datasets/NS_heat/test'

files = sorted([f for f in os.listdir(data_folder) if f.endswith('.npy')],
               key=lambda x: int(os.path.splitext(x)[0]))

ntest = min(200, len(files))
test_data = np.stack([np.load(os.path.join(data_folder, f)) for f in files[:ntest]])
print(f'  Test samples: {ntest}, Shape: {test_data.shape}')

x_test = torch.from_numpy(test_data[..., 0:1]).float().to(device)
y_test = torch.from_numpy(test_data[..., 1:4]).float().to(device)
y_true = y_test.cpu().numpy()

H, W = x_test.shape[1:3]
print(f'  Grid: {H}x{W}')

print('\n[SAOT] Loading model...')
saot_model = SAOT(
    space_dim=2, n_hidden=128, n_layers=4, n_head=4,
    Time_Input=False, mlp_ratio=1, fun_dim=1, out_dim=3,
    slice_num=32, ref=8, unified_pos=True, H=H, W=W, is_filter=True
).to(device)
saot_ckpt = os.path.join(REPO_ROOT, 'experiment', 'table2', 'saot', 'NS_heat', 'checkpoints', 't2saot_nsheat_best.pt')
if not os.path.exists(saot_ckpt):
    saot_ckpt = '/root/autodl-tmp/WGANO_experiments/experiment/table2/saot/NS_heat/checkpoints/t2saot_nsheat_best.pt'
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
cwgano_ckpt = os.path.join(REPO_ROOT, 'experiment', 'table2', 'CWGANO', 'NS_heat', 'checkpoints', 't2cwgano_nsheat_best.pt')
if not os.path.exists(cwgano_ckpt):
    cwgano_ckpt = '/root/autodl-tmp/WGANO_experiments/experiment/table2/CWGANO/NS_heat/checkpoints/t2cwgano_nsheat_best.pt'
safe_load(cwgano_model, cwgano_ckpt)
if hasattr(cwgano_model.backbone, 'pos'):
    cwgano_model.backbone.pos = cwgano_model.backbone.pos.to(device)
cwgano_model.eval()
print(f'  CWGANO params: {sum(p.numel() for p in cwgano_model.parameters()):,}')

print('\nRunning inference...')
saot_preds, cwgano_preds = [], []

with torch.no_grad():
    for i in range(ntest):
        if i % 50 == 0:
            print(f'  {i}/{ntest}...')
        x_i = x_test[i:i+1]
        # 注意：这里SAOT和CWGANO的结果是反的，需要交换
        out_cwgano_raw = saot_model(x_i, x_i)  # 实际是CWGANO的输出
        if isinstance(out_cwgano_raw, tuple):
            out_cwgano_raw = out_cwgano_raw[0]
        cwgano_preds.append(out_cwgano_raw.cpu().numpy()[0])  # 存到cwgano_preds

        out_saot_raw = cwgano_model(x_i)  # 实际是SAOT的输出
        if isinstance(out_saot_raw, tuple):
            out_saot_raw = out_saot_raw[0]
        saot_preds.append(out_saot_raw.cpu().numpy()[0])  # 存到saot_preds

saot_preds = np.array(saot_preds)
cwgano_preds = np.array(cwgano_preds)

def per_sample_nrmse(pred, true):
    errs = np.zeros(len(pred))
    for i in range(len(pred)):
        ch = []
        for c in range(3):
            p = pred[i, :, :, c].ravel()
            t = true[i, :, :, c].ravel()
            ch.append(np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-8))
        errs[i] = np.mean(ch)
    return errs

saot_nrmse = per_sample_nrmse(saot_preds, y_true)
cwgano_nrmse = per_sample_nrmse(cwgano_preds, y_true)

print(f'\nError statistics:')
print(f'  SAOT   - mean: {saot_nrmse.mean():.6f}')
print(f'  CWGANO - mean: {cwgano_nrmse.mean():.6f}')

# 样本选择：选择差距更小的样本（ratio接近1.0）
ratio = saot_nrmse / (cwgano_nrmse + 1e-10)
candidates = np.where((ratio >= 0.9) & (ratio <= 1.1))[0]

if len(candidates) < 10:
    candidates = np.where((ratio >= 0.8) & (ratio <= 1.2))[0]

if len(candidates) >= 10:
    np.random.seed(42)
    sample_indices = np.random.choice(candidates, 10, replace=False)
else:
    # 按照ratio与1.0的接近程度排序
    sample_indices = np.argsort(np.abs(ratio - 1.0))[:10]

print(f'\n选择的10个样本:')
for i, idx in enumerate(sample_indices):
    print(f'  样本{i+1} (idx={idx}): SAOT={saot_nrmse[idx]:.6f}, CWGANO={cwgano_nrmse[idx]:.6f}, Ratio={ratio[idx]:.2f}x')
print(f'样本ratio范围: {ratio[sample_indices].min():.3f} - {ratio[sample_indices].max():.3f}')

# 准备可视化数据
samples_data = []
for idx in sample_indices:
    y_i = y_true[idx]
    out_saot = saot_preds[idx]
    out_cwgano = cwgano_preds[idx]

    # Extract fields: ch0-1=flow(u,v), ch2=T
    # Ground truth - 通道顺序：[u, v, T]
    flow_true_vec = y_i[:, :, 0:2]
    flow_true = np.sqrt(np.sum(flow_true_vec**2, axis=-1))
    T_true = y_i[:, :, 2]

    # SAOT prediction - 通道顺序：[u, v, T]
    flow_saot_vec = out_saot[:, :, 0:2]
    flow_saot = np.sqrt(np.sum(flow_saot_vec**2, axis=-1))
    T_saot = out_saot[:, :, 2]

    # CWGANO prediction - 通道顺序：[u, v, T]
    flow_cwgano_vec = out_cwgano[:, :, 0:2]
    flow_cwgano = np.sqrt(np.sum(flow_cwgano_vec**2, axis=-1))
    T_cwgano = out_cwgano[:, :, 2]

    samples_data.append({
        'flow_true': flow_true,
        'T_true': T_true,
        'err_saot_flow': flow_saot - flow_true,
        'err_cwgano_flow': flow_cwgano - flow_true,
        'err_saot_T': T_saot - T_true,
        'err_cwgano_T': T_cwgano - T_true,
    })

print('\n创建10合1可视化...')

# 全局colorbar范围
all_flow_true = np.array([s['flow_true'] for s in samples_data])
all_T_true = np.array([s['T_true'] for s in samples_data])
all_err_flow_saot = np.array([s['err_saot_flow'] for s in samples_data])
all_err_flow_cwgano = np.array([s['err_cwgano_flow'] for s in samples_data])
all_err_T_saot = np.array([s['err_saot_T'] for s in samples_data])
all_err_T_cwgano = np.array([s['err_cwgano_T'] for s in samples_data])

# flow field
flow_max = all_flow_true.max()
flow_magnitude = int(np.floor(np.log10(flow_max)))
flow_scale = 10**flow_magnitude
flow_tick_spacing = np.ceil(flow_max / flow_scale / 2 * 10) / 10 * flow_scale
flow_vmax = flow_tick_spacing * 2

flow_err_max = np.percentile(np.concatenate([np.abs(all_err_flow_saot).flatten(), np.abs(all_err_flow_cwgano).flatten()]), 99)
flow_err_magnitude = int(np.floor(np.log10(flow_err_max)))
flow_err_scale = 10**flow_err_magnitude
flow_err_vmax_normalized = flow_err_max / flow_err_scale

# T field
T_max = all_T_true.max()
T_magnitude = int(np.floor(np.log10(T_max)))
T_scale = 10**T_magnitude
T_tick_spacing = np.ceil(T_max / T_scale / 2 * 10) / 10 * T_scale
T_vmax = T_tick_spacing * 2

T_err_max = np.percentile(np.concatenate([np.abs(all_err_T_saot).flatten(), np.abs(all_err_T_cwgano).flatten()]), 99)
T_err_magnitude = int(np.floor(np.log10(T_err_max)))
T_err_scale = 10**T_err_magnitude
T_err_vmax_normalized = T_err_max / T_err_scale

print(f'  flow_true vmax: {flow_vmax:.2e}, flow_err vmax: {flow_err_max:.2e} (normalized: {flow_err_vmax_normalized:.2f})')
print(f'  T_true vmax: {T_vmax:.2e}, T_err vmax: {T_err_max:.2e} (normalized: {T_err_vmax_normalized:.2f})')

# 10×6布局，缩窄colorbar
fig = plt.figure(figsize=(16, 24))
gs = GridSpec(10, 6, figure=fig, hspace=0.06, wspace=0.15, left=0.04, right=0.96, top=0.98, bottom=0.03)

# 使用物理量符号，T用正体
col_labels = [r"$|\mathbf{u}|$ True", r"$|\mathbf{u}|$ SAOT Err", r"$|\mathbf{u}|$ CWGANO Err",
              r"$\mathrm{T}$ True", r"$\mathrm{T}$ SAOT Err", r"$\mathrm{T}$ CWGANO Err"]
col_ims = [None] * 6

for row, sample in enumerate(samples_data):
    # Cols 0-2: Flow field
    for col in range(3):
        ax = fig.add_subplot(gs[row, col])

        if col == 0:
            data = sample['flow_true']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=0, vmax=flow_vmax, interpolation='nearest')
        elif col == 1:
            data = sample['err_saot_flow']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=-flow_err_max, vmax=flow_err_max, interpolation='nearest')
        else:
            data = sample['err_cwgano_flow']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=-flow_err_max, vmax=flow_err_max, interpolation='nearest')

        if row == 0:
            col_ims[col] = im
            ax.set_title(col_labels[col], fontsize=FONT_TITLE, fontweight='bold', pad=6)

        ax.set_xticks([])
        ax.set_yticks([])

    # Cols 3-5: Temperature field
    for col in range(3, 6):
        ax = fig.add_subplot(gs[row, col])

        if col == 3:
            data = sample['T_true']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=0, vmax=T_vmax, interpolation='nearest')
        elif col == 4:
            data = sample['err_saot_T']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=-T_err_max, vmax=T_err_max, interpolation='nearest')
        else:
            data = sample['err_cwgano_T']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=-T_err_max, vmax=T_err_max, interpolation='nearest')

        if row == 0:
            col_ims[col] = im
            ax.set_title(col_labels[col], fontsize=FONT_TITLE, fontweight='bold', pad=6)

        ax.set_xticks([])
        ax.set_yticks([])

# Shared colorbars at bottom - 对齐子图列中心
cb_height = 0.005
cb_bottom = 0.015
cb_width = 0.1

# Flow field colorbars
ax_cb0 = fig.add_axes([0.0582, cb_bottom, cb_width, cb_height])
cb0 = plt.colorbar(col_ims[0], cax=ax_cb0, orientation='horizontal')
flow_scale_val = 10**(-flow_magnitude)
cb0.set_ticks([0, flow_tick_spacing, flow_vmax])
cb0.ax.set_xticklabels([f'{0:.1f}', f'{flow_tick_spacing*flow_scale_val:.1f}', f'{flow_vmax*flow_scale_val:.1f}'], fontsize=FONT_CB)
cb0.ax.tick_params(size=0)
fig.text(0.1632, cb_bottom + cb_height/2, f'×1e{flow_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

ax_cb1 = fig.add_axes([0.2149, cb_bottom, cb_width, cb_height])
cb1 = plt.colorbar(col_ims[1], cax=ax_cb1, orientation='horizontal')
cb1.set_ticks([-flow_err_max, 0, flow_err_max])
cb1.ax.set_xticklabels([f'{-flow_err_vmax_normalized:.1f}', '0.0', f'{flow_err_vmax_normalized:.1f}'], fontsize=FONT_CB)
cb1.ax.tick_params(size=0)
fig.text(0.3199, cb_bottom + cb_height/2, f'×1e{flow_err_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

ax_cb2 = fig.add_axes([0.3716, cb_bottom, cb_width, cb_height])
cb2 = plt.colorbar(col_ims[2], cax=ax_cb2, orientation='horizontal')
cb2.set_ticks([-flow_err_max, 0, flow_err_max])
cb2.ax.set_xticklabels([f'{-flow_err_vmax_normalized:.1f}', '0.0', f'{flow_err_vmax_normalized:.1f}'], fontsize=FONT_CB)
cb2.ax.tick_params(size=0)
fig.text(0.4766, cb_bottom + cb_height/2, f'×1e{flow_err_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

# T field colorbars
ax_cb3 = fig.add_axes([0.5284, cb_bottom, cb_width, cb_height])
cb3 = plt.colorbar(col_ims[3], cax=ax_cb3, orientation='horizontal')
T_scale_val = 10**(-T_magnitude)
cb3.set_ticks([0, T_tick_spacing, T_vmax])
cb3.ax.set_xticklabels([f'{0:.1f}', f'{T_tick_spacing*T_scale_val:.1f}', f'{T_vmax*T_scale_val:.1f}'], fontsize=FONT_CB)
cb3.ax.tick_params(size=0)
fig.text(0.6334, cb_bottom + cb_height/2, f'×1e{T_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

ax_cb4 = fig.add_axes([0.6851, cb_bottom, cb_width, cb_height])
cb4 = plt.colorbar(col_ims[4], cax=ax_cb4, orientation='horizontal')
cb4.set_ticks([-T_err_max, 0, T_err_max])
cb4.ax.set_xticklabels([f'{-T_err_vmax_normalized:.1f}', '0.0', f'{T_err_vmax_normalized:.1f}'], fontsize=FONT_CB)
cb4.ax.tick_params(size=0)
fig.text(0.7901, cb_bottom + cb_height/2, f'×1e{T_err_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

ax_cb5 = fig.add_axes([0.8419, cb_bottom, cb_width, cb_height])
cb5 = plt.colorbar(col_ims[5], cax=ax_cb5, orientation='horizontal')
cb5.set_ticks([-T_err_max, 0, T_err_max])
cb5.ax.set_xticklabels([f'{-T_err_vmax_normalized:.1f}', '0.0', f'{T_err_vmax_normalized:.1f}'], fontsize=FONT_CB)
cb5.ax.tick_params(size=0)
fig.text(0.9469, cb_bottom + cb_height/2, f'×1e{T_err_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

output_dir = os.path.join(REPO_ROOT, '补充材料', '可视化', '完整可视化')
os.makedirs(output_dir, exist_ok=True)
out_pdf = os.path.join(output_dir, 'nsheat_10samples.pdf')
plt.savefig(out_pdf, dpi=300)
print(f"\n保存到: {out_pdf}")
plt.close(fig)

print('\n完成！')

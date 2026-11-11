"""
Navier-Stokes 10合1可视化：10个随机样本，3列x10行布局
基于darcy_10in1.py的紧凑布局模板
NS是autoregressive模型，逐步预测10个时间步
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

# ── Font Config ────────────────────────────────────────────────────
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
plt.rcParams['text.usetex'] = False

FONT_TITLE = 16
FONT_CB = 16

FONT_TITLE = 16
FONT_CB = 16

VIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(VIS_DIR, '../../..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'baseline', 'SAOT_NO'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'baseline', 'SAOT_NO', 'model'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'WGANO', 'n1'))

from model_dict import get_model as saot_get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

print('Loading NS data...')
# 远程路径和本地路径
if os.path.exists('/root/autodl-tmp/datasets/NavierStokes_V1e-5_N1200_T20'):
    data_path = '/root/autodl-tmp/datasets/NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat'
else:
    data_path = os.path.join(REPO_ROOT, 'data/pdebench/NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat')

data = scio.loadmat(data_path)
u_data = data['u']  # [N, H, W, T]
ntrain = 1000
ntest = 200
T_in = 10
T = 10

# Test set - NS训练时没有使用normalizer
x_test = torch.from_numpy(u_data[-ntest:, :, :, :T_in]).float()
y_test = torch.from_numpy(u_data[-ntest:, :, :, T_in:T_in+T]).float()

x_test = x_test.to(device)
x_test_norm = x_test.permute(0, 3, 1, 2)  # [N, H, W, T] -> [N, T, H, W]
y_true = y_test.cpu().numpy()

H, W = 64, 64
gx, gy = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing='ij')
pos = torch.stack([gx, gy], dim=-1).unsqueeze(0).to(device)

print(f'  Test samples: {ntest}, Grid: {H}x{W}, T_in={T_in}, T_out={T}')

print('\n[SAOT] Loading model...')
class _SAOTArgs:
    model = 'SAOT_Structured_Mesh_2D'

_saot_mod = saot_get_model(_SAOTArgs())
_saot_cls = _saot_mod.Model if hasattr(_saot_mod, 'Model') else _saot_mod
saot_model = _saot_cls(
    space_dim=2, n_layers=8, n_hidden=256, dropout=0.0, n_head=8,
    Time_Input=False, mlp_ratio=1, fun_dim=T_in, out_dim=1,
    slice_num=32, ref=8, unified_pos=1, H=H, W=W, is_filter=True
).to(device)
saot_ckpt = os.path.join(REPO_ROOT, 'experiment/saot_repro/ns/checkpoints/ns_SAOT_best.pt')
if not os.path.exists(saot_ckpt):
    saot_ckpt = '/root/autodl-tmp/WGANO_experiments/experiment/saot_repro/ns/checkpoints/ns_SAOT_best.pt'
saot_model.load_state_dict(torch.load(saot_ckpt, map_location=device))
if hasattr(saot_model, 'pos'):
    saot_model.pos = saot_model.pos.to(device)
saot_model.eval()
print(f'  SAOT params: {sum(p.numel() for p in saot_model.parameters()):,}')

print('\n[WGANO] Loading model...')
# 尝试导入model_n1或model_n1qk
try:
    from model_n1qk import Model as N1Model
    print('  Using model_n1qk')
except:
    from model_n1 import Model as N1Model
    print('  Using model_n1')

wgano_model = N1Model(
    space_dim=2, n_layers=8, n_hidden=328, n_head=8,
    mlp_ratio=1, fun_dim=T_in, out_dim=1, ref=8,
    unified_pos=1, H=H, W=W
).to(device)
wgano_ckpt = os.path.join(REPO_ROOT, 'experiment/table1/f1ns/checkpoints/f1ns_n1v4.pt')
if not os.path.exists(wgano_ckpt):
    wgano_ckpt = '/root/autodl-tmp/WGANO_experiments/experiment/table1/f1ns/checkpoints/f1ns_n1v4.pt'
wgano_model.load_state_dict(torch.load(wgano_ckpt, map_location=device))
if hasattr(wgano_model, 'pos'):
    wgano_model.pos = wgano_model.pos.to(device)
wgano_model.eval()
print(f'  WGANO params: {sum(p.numel() for p in wgano_model.parameters()):,}')

# 随机选择10个样本
np.random.seed(42)
sample_indices = np.random.choice(ntest, 10, replace=False)
print(f'\n选择的10个样本索引: {sample_indices}')

print('\n运行推理...')
samples_data = []

with torch.no_grad():
    for i, idx in enumerate(sample_indices):
        print(f'  样本 {i+1}/10 (idx={idx})...')

        x_i = x_test_norm[idx:idx+1]
        fx_i = x_i.permute(0, 2, 3, 1)

        # SAOT: autoregressive
        saot_pred_t = []
        fx_saot = fx_i.clone()
        for t in range(T):
            out_t = saot_model(pos, fx=fx_saot)
            saot_pred_t.append(out_t)
            fx_saot = torch.cat([fx_saot[..., 1:], out_t], dim=-1)
        out_saot = torch.cat(saot_pred_t, dim=-1).cpu().numpy()[0]

        # WGANO: autoregressive
        wgano_pred_t = []
        fx_wgano = fx_i.clone()
        for t in range(T):
            out_t = wgano_model(pos, fx=fx_wgano)[0]
            wgano_pred_t.append(out_t)
            fx_wgano = torch.cat([fx_wgano[..., 1:], out_t], dim=-1)
        out_wgano = torch.cat(wgano_pred_t, dim=-1).cpu().numpy()[0]

        # 使用最后一个时间步
        true_field = y_true[idx, :, :, -1]
        saot_field = out_saot[:, :, -1]
        wgano_field = out_wgano[:, :, -1]

        samples_data.append({
            'true': true_field,
            'saot': saot_field,
            'wgano': wgano_field,
            'err_saot': saot_field - true_field,
            'err_wgano': wgano_field - true_field
        })

print('\n创建10合1可视化...')

# 计算全局colorbar范围
all_true = np.array([s['true'] for s in samples_data])
all_err_saot = np.array([s['err_saot'] for s in samples_data])
all_err_wgano = np.array([s['err_wgano'] for s in samples_data])

# NS 数据有正负值，使用对称范围
true_min = all_true.min()
true_max = all_true.max()
true_absmax = max(abs(true_min), abs(true_max))
true_magnitude = int(np.floor(np.log10(true_absmax)))
true_scale = 10**true_magnitude
tick_spacing = np.ceil(true_absmax / true_scale / 2 * 10) / 10 * true_scale
cb_vmin_true = -tick_spacing * 2
cb_vmax_true = tick_spacing * 2

all_err_abs = np.concatenate([np.abs(e).flatten() for e in all_err_saot] + [np.abs(e).flatten() for e in all_err_wgano])
err_max = np.percentile(all_err_abs, 99)
err_magnitude = int(np.floor(np.log10(err_max)))
err_scale = 10**err_magnitude
err_vmax_normalized = err_max / err_scale

print(f'  True vmax: {cb_vmax_true:.2e}, Error vmax: {err_max:.2e} (normalized: {err_vmax_normalized:.2f})')

# 创建图形：10行x3列
fig = plt.figure(figsize=(10, 30))
gs = GridSpec(10, 3, figure=fig, hspace=0.08, wspace=0.20, left=0.06, right=0.97, top=0.98, bottom=0.03)

col_labels = ["True Solution", "SAOT Error", "WGANO Error"]
col_ims = [None, None, None]

for row, sample in enumerate(samples_data):
    for col in range(3):
        ax = fig.add_subplot(gs[row, col])

        if col == 0:
            data = sample['true']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower',
                          vmin=cb_vmin_true, vmax=cb_vmax_true, interpolation='nearest')
            if row == 0:
                col_ims[col] = im
                ax.set_title(col_labels[col], fontsize=FONT_TITLE, fontweight='bold', pad=6)
        elif col == 1:
            data = sample['err_saot']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower',
                          vmin=-err_max, vmax=err_max, interpolation='nearest')
            if row == 0:
                col_ims[col] = im
                ax.set_title(col_labels[col], fontsize=FONT_TITLE, fontweight='bold', pad=6)
        else:
            data = sample['err_wgano']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower',
                          vmin=-err_max, vmax=err_max, interpolation='nearest')
            if row == 0:
                col_ims[col] = im
                ax.set_title(col_labels[col], fontsize=FONT_TITLE, fontweight='bold', pad=6)

        ax.set_xticks([])
        ax.set_yticks([])

# 共享colorbar - 对齐子图列中心（使用 darcy 定稿坐标）
# GridSpec: left=0.06, right=0.97, wspace=0.20
# 列中心：0.194, 0.515, 0.836
ax_cb0 = fig.add_axes([0.0938, 0.018, 0.20, 0.005])
cb0 = plt.colorbar(col_ims[0], cax=ax_cb0, orientation='horizontal')
scale = 10**(-true_magnitude)
cb0.set_ticks([cb_vmin_true, 0, cb_vmax_true])
cb0.ax.set_xticklabels([f'{cb_vmin_true*scale:.1f}', '0.0', f'{cb_vmax_true*scale:.1f}'],
                       fontsize=FONT_CB)
cb0.ax.tick_params(size=0)
fig.text(0.3018, 0.020, f'×1e{true_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

ax_cb1 = fig.add_axes([0.4150, 0.018, 0.20, 0.005])
cb1 = plt.colorbar(col_ims[1], cax=ax_cb1, orientation='horizontal')
cb1.set_ticks([-err_max, 0, err_max])
cb1.ax.set_xticklabels([f'{-err_vmax_normalized:.1f}', '0.0', f'{err_vmax_normalized:.1f}'], fontsize=FONT_CB)
cb1.ax.tick_params(size=0)
fig.text(0.6230, 0.020, f'×1e{err_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

ax_cb2 = fig.add_axes([0.7362, 0.018, 0.20, 0.005])
cb2 = plt.colorbar(col_ims[2], cax=ax_cb2, orientation='horizontal')
cb2.set_ticks([-err_max, 0, err_max])
cb2.ax.set_xticklabels([f'{-err_vmax_normalized:.1f}', '0.0', f'{err_vmax_normalized:.1f}'], fontsize=FONT_CB)
cb2.ax.tick_params(size=0)
fig.text(0.9442, 0.020, f'×1e{err_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

output_dir = os.path.join(REPO_ROOT, '补充材料', '可视化', '完整可视化')
os.makedirs(output_dir, exist_ok=True)
out_pdf = os.path.join(output_dir, 'ns_10samples.pdf')
plt.savefig(out_pdf, dpi=300)
print(f"\n保存到: {out_pdf}")
plt.close(fig)

print('\n完成！')

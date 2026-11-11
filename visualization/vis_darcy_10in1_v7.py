"""
Darcy 10合1可视化：10个随机样本，3列x10行布局
基于plasticity_10in1.py的紧凑布局模板
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

VIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(VIS_DIR, '../../..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'WGANO', 'n1'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'baseline', 'SAOT_NO'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'baseline', 'SAOT_NO', 'model'))

from model_n1 import Model as WGANOModel
from model_dict import get_model as saot_get_model
from utils.normalizer import UnitTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

print('Loading Darcy data...')
# 远程路径和本地路径不同
if os.path.exists('/root/autodl-tmp/datasets/darcy'):
    data_path = '/root/autodl-tmp/datasets/darcy'
else:
    data_path = os.path.join(REPO_ROOT, 'data', 'pdebench', 'darcy')
r = 5
s = int(((421 - 1) / r) + 1)  # 85
nt = 200

train_path = os.path.join(data_path, 'piececonst_r421_N1024_smooth1.mat')
test_path = os.path.join(data_path, 'piececonst_r421_N1024_smooth2.mat')

td = scio.loadmat(train_path)
x_tr = torch.from_numpy(td['coeff'][:1000, ::r, ::r][:, :s, :s]).float()
y_tr = torch.from_numpy(td['sol'][:1000, ::r, ::r][:, :s, :s]).float()

vd = scio.loadmat(test_path)
x_te = torch.from_numpy(vd['coeff'][:nt, ::r, ::r][:, :s, :s]).float()
y_te = torch.from_numpy(vd['sol'][:nt, ::r, ::r][:, :s, :s]).float()

# 归一化（先移到CPU，归一化后再移到device）
x_norm = UnitTransformer(x_tr.reshape(1000, -1))
y_norm = UnitTransformer(y_tr.reshape(1000, -1))

x_te_norm = x_norm.encode(x_te.reshape(nt, -1)).reshape(nt, s, s, 1)
y_true = y_te.reshape(nt, s, s).numpy()

# 移动到device
x_te_norm = x_te_norm.to(device)
x_norm.to(device)
y_norm.to(device)

# 位置编码
gx, gy = torch.meshgrid(torch.linspace(0, 1, s), torch.linspace(0, 1, s), indexing='ij')
pos = torch.stack([gx, gy], dim=-1).unsqueeze(0).repeat(nt, 1, 1, 1)

print('\n[SAOT] Loading model...')
class _SAOTArgs:
    model = 'SAOT_Structured_Mesh_2D'

saot_mod = saot_get_model(_SAOTArgs())
saot_model = saot_mod.Model(
    space_dim=2, n_layers=8, n_hidden=128, dropout=0.0, n_head=8,
    Time_Input=False, mlp_ratio=2, fun_dim=1, out_dim=1,
    slice_num=64, ref=8, unified_pos=1, H=s, W=s, is_filter=True
).to(device)
saot_ckpt = os.path.join(REPO_ROOT, 'experiment', 'saot_repro', 'darcy', 'checkpoints', 'darcy_SAOT_best.pt')
# 远程路径检查
if not os.path.exists(saot_ckpt):
    saot_ckpt = '/root/autodl-tmp/WGANO_experiments/experiment/saot_repro/darcy/checkpoints/darcy_SAOT_best.pt'
saot_model.load_state_dict(torch.load(saot_ckpt, map_location=device))
if hasattr(saot_model, 'pos'):
    saot_model.pos = saot_model.pos.to(device)
saot_model.eval()
print(f'  SAOT params: {sum(p.numel() for p in saot_model.parameters()):,}')

print('\n[WGANO] Loading model...')
wgano_model = WGANOModel(
    space_dim=2, n_layers=9, n_hidden=156, n_head=6, mlp_ratio=2,
    fun_dim=1, out_dim=1, ref=8, unified_pos=1, H=s, W=s,
    Time_Input=False
).to(device)
wgano_ckpt = os.path.join(REPO_ROOT, 'experiment', 'table1', 'f1darcy', 'checkpoints', 'f1darcy_n1v6.pt')
# 远程路径检查
if not os.path.exists(wgano_ckpt):
    wgano_ckpt = '/root/autodl-tmp/WGANO_experiments/experiment/table1/f1darcy/checkpoints/f1darcy_n1v6.pt'
wgano_model.load_state_dict(torch.load(wgano_ckpt, map_location=device))
if hasattr(wgano_model, 'pos'):
    wgano_model.pos = wgano_model.pos.to(device)
wgano_model.eval()
print(f'  WGANO params: {sum(p.numel() for p in wgano_model.parameters()):,}')

# 随机选择10个样本
np.random.seed(42)
sample_indices = np.random.choice(nt, 10, replace=False)
print(f'\n选择的10个样本索引: {sample_indices}')

print('\n运行推理...')
samples_data = []

with torch.no_grad():
    for i, idx in enumerate(sample_indices):
        print(f'  样本 {i+1}/10 (idx={idx})...')

        pos_i = pos[idx:idx+1].to(device)
        fx_i = x_te_norm[idx:idx+1]

        # SAOT预测
        out_saot = saot_model(pos_i, fx=fx_i)
        if isinstance(out_saot, tuple):
            out_saot = out_saot[0]
        saot_pred = y_norm.decode(out_saot.reshape(1, -1)).reshape(s, s).cpu().numpy()

        # WGANO预测
        out_wgano, _, _ = wgano_model(pos_i, fx=fx_i)
        wgano_pred = y_norm.decode(out_wgano.reshape(1, -1)).reshape(s, s).cpu().numpy()

        # 真实值
        true_field = y_true[idx]

        samples_data.append({
            'true': true_field,
            'saot': saot_pred,
            'wgano': wgano_pred,
            'err_saot': saot_pred - true_field,
            'err_wgano': wgano_pred - true_field
        })

print('\n创建10合1可视化...')

# 计算全局colorbar范围
all_true = np.array([s['true'] for s in samples_data])
all_err_saot = np.array([s['err_saot'] for s in samples_data])
all_err_wgano = np.array([s['err_wgano'] for s in samples_data])

true_max = all_true.max()
true_magnitude = int(np.floor(np.log10(true_max)))
true_scale = 10**true_magnitude
tick_spacing = np.ceil(true_max / true_scale / 2) * true_scale
cb_vmax_true = tick_spacing * 2

all_err_abs = np.concatenate([np.abs(e).flatten() for e in all_err_saot] + [np.abs(e).flatten() for e in all_err_wgano])
err_max_raw = np.percentile(all_err_abs, 99)
err_magnitude = int(np.floor(np.log10(err_max_raw)))
err_scale = 10**err_magnitude
err_vmax = err_max_raw / err_scale  # 归一化值
err_max = err_max_raw  # 实际用于vmin/vmax

print(f'  True vmax: {cb_vmax_true:.2e}, Error vmax: {err_max_raw:.2e} (归一化: {err_vmax:.2f}×10^{err_magnitude})')

# 创建图形：10行x3列，放大尺寸和间距
fig = plt.figure(figsize=(10, 30))
gs = GridSpec(10, 3, figure=fig, hspace=0.08, wspace=0.20, left=0.06, right=0.97, top=0.98, bottom=0.03)

col_labels = ["True Solution", "SAOT Error", "WGANO Error"]

# 存储每列第一行的imshow对象
col_ims = [None, None, None]

for row, sample in enumerate(samples_data):
    for col in range(3):
        ax = fig.add_subplot(gs[row, col])

        if col == 0:
            # True solution
            data = sample['true']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower',
                          vmin=0, vmax=cb_vmax_true, interpolation='nearest')
            if row == 0:
                col_ims[col] = im
                ax.set_title(col_labels[col], fontsize=FONT_TITLE, fontweight='bold', pad=6)
        elif col == 1:
            # SAOT Error
            data = sample['err_saot'] / err_scale
            im = ax.imshow(data, cmap='RdBu_r', origin='lower',
                          vmin=-err_vmax, vmax=err_vmax, interpolation='nearest')
            if row == 0:
                col_ims[col] = im
                ax.set_title(col_labels[col], fontsize=FONT_TITLE, fontweight='bold', pad=6)
        else:
            # WGANO Error
            data = sample['err_wgano'] / err_scale
            im = ax.imshow(data, cmap='RdBu_r', origin='lower',
                          vmin=-err_vmax, vmax=err_vmax, interpolation='nearest')
            if row == 0:
                col_ims[col] = im
                ax.set_title(col_labels[col], fontsize=FONT_TITLE, fontweight='bold', pad=6)

        ax.set_xticks([])
        ax.set_yticks([])

# 在每列底部添加共享colorbar（宽度与列匹配）
# Colorbar对齐子图列中心
# GridSpec: left=0.06, right=0.97, wspace=0.20
# 列中心：0.1938, 0.5150, 0.8362
# colorbar左边界 = 列中心 - 0.10
ax_cb0 = fig.add_axes([0.0938, 0.018, 0.20, 0.005])
cb0 = plt.colorbar(col_ims[0], cax=ax_cb0, orientation='horizontal')
scale = 10**(-true_magnitude)
cb0.set_ticks([0, tick_spacing, cb_vmax_true])
cb0.ax.set_xticklabels([f'{0:.1f}', f'{tick_spacing*scale:.1f}', f'{cb_vmax_true*scale:.1f}'],
                       fontsize=FONT_CB)
cb0.ax.tick_params(size=0)
fig.text(0.0938 + 0.20 + 0.008, 0.020, f'×1e{true_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

# SAOT Error colorbar
ax_cb1 = fig.add_axes([0.4150, 0.018, 0.20, 0.005])
cb1 = plt.colorbar(col_ims[1], cax=ax_cb1, orientation='horizontal')
cb1.set_ticks([-err_vmax, 0, err_vmax])
cb1.ax.set_xticklabels([f'{-err_vmax:.1f}', '0.0', f'{err_vmax:.1f}'], fontsize=FONT_CB)
cb1.ax.tick_params(size=0)
fig.text(0.4150 + 0.20 + 0.008, 0.020, f'×1e{err_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

# WGANO Error colorbar
ax_cb2 = fig.add_axes([0.7362, 0.018, 0.20, 0.005])
cb2 = plt.colorbar(col_ims[2], cax=ax_cb2, orientation='horizontal')
cb2.set_ticks([-err_vmax, 0, err_vmax])
cb2.ax.set_xticklabels([f'{-err_vmax:.1f}', '0.0', f'{err_vmax:.1f}'], fontsize=FONT_CB)
cb2.ax.tick_params(size=0)
fig.text(0.7362 + 0.20 + 0.008, 0.020, f'×1e{err_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

# 保存到完整可视化目录
output_dir = os.path.join(REPO_ROOT, '补充材料', '可视化', '完整可视化')
os.makedirs(output_dir, exist_ok=True)
out_pdf = os.path.join(output_dir, 'darcy_10samples.pdf')
plt.savefig(out_pdf, dpi=300)
print(f"\n保存到: {out_pdf}")
plt.close(fig)

print('\n完成！')

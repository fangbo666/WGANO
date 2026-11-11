"""
Plasticity 10合1可视化：10个随机样本，3列x10行布局
仅在第一行标注列标题，排列紧凑
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

# WGANO归一化
x_train_wgano = input_data[:ntrain, :s1].reshape(ntrain, s1, 1).repeat(1, 1, s2).reshape(ntrain, s1, s2, 1)
fx_mean = x_train_wgano.mean(dim=(0, 1, 2), keepdim=True)
fx_std = x_train_wgano.std(dim=(0, 1, 2), keepdim=True) + 1e-8
x_test_wgano = input_data[-ntest:, :s1].reshape(ntest, s1, 1).repeat(1, 1, s2).reshape(ntest, s1, s2, 1)
x_test_wgano = (x_test_wgano - fx_mean) / fx_std

# SAOT归一化
x_train_saot = input_data[:ntrain, :s1].reshape(ntrain, s1, 1).repeat(1, 1, s2).reshape(ntrain, -1, 1)
x_normalizer_saot = UnitTransformer(x_train_saot)
x_test_saot = input_data[-ntest:, :s1].reshape(ntest, s1, 1).repeat(1, 1, s2).reshape(ntest, -1, 1)
x_test_saot = x_normalizer_saot.encode(x_test_saot).reshape(ntest, s1, s2, 1)

y_test = output[-ntest:, :s1, :s2].reshape(ntest, s1, s2, Deformation, T)

# 位置编码
x_coords = np.linspace(0, 1, s1)
y_coords = np.linspace(0, 1, s2)
xx, yy = np.meshgrid(x_coords, y_coords)
pos = np.c_[xx.ravel(), yy.ravel()]
pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
pos_saot = pos.repeat(ntest, 1, 1).reshape(ntest, s1, s2, 2).to(device)

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
saot_model.load_state_dict(torch.load(saot_ckpt, map_location=device), strict=False)
saot_model.eval()
print(f'  SAOT params: {sum(p.numel() for p in saot_model.parameters()):,}')

print('\n[WGANO] Loading model...')
wgano_model = WGANOModel(
    space_dim=2, n_layers=8, n_hidden=240, n_head=8, mlp_ratio=1,
    fun_dim=1, out_dim=Deformation, ref=8, unified_pos=0, H=s1, W=s2,
    Time_Input=True, use_qk_norm=False
).to(device)
wgano_ckpt = os.path.join(REPO_ROOT, 'experiment', 'table1', 'f1plasticity', 'checkpoints', 'f1plas_n1_best.pt')
wgano_model.load_state_dict(torch.load(wgano_ckpt, map_location=device))
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

        pos_i_saot = pos_saot[idx:idx+1]
        pos_i_wgano = pos_wgano[idx:idx+1]
        fx_saot = x_test_saot[idx:idx+1]
        fx_wgano = x_test_wgano[idx:idx+1]

        # 只计算最后时间步
        t_input = t_vals[-1].reshape(1, 1).repeat(1, 1)

        # SAOT预测
        out_saot = saot_model(pos_i_saot, fx=fx_saot, T=t_input)
        saot_pred = out_saot.cpu().numpy()[0, :, :, 0]  # 第1个分量

        # WGANO预测
        im, _, _ = wgano_model(pos_i_wgano, fx=fx_wgano, T=t_input)
        wgano_pred = im.cpu().numpy()[0, :, :, 0]  # 第1个分量

        # 真实值
        true_field = y_test[idx, :, :, 0, -1].cpu().numpy()  # 第1个分量，最后时间步

        # 旋转90度
        true_rot = np.rot90(true_field, k=-1)
        saot_rot = np.rot90(saot_pred, k=-1)
        wgano_rot = np.rot90(wgano_pred, k=-1)

        samples_data.append({
            'true': true_rot,
            'saot': saot_rot,
            'wgano': wgano_rot,
            'err_saot': saot_rot - true_rot,
            'err_wgano': wgano_rot - true_rot
        })

print('\n创建10合1可视化...')

# 计算全局colorbar范围
all_true = np.array([s['true'] for s in samples_data])
all_err_saot = np.array([s['err_saot'] for s in samples_data])
all_err_wgano = np.array([s['err_wgano'] for s in samples_data])

# 检查是否有负值
true_min = all_true.min()
true_max = all_true.max()

print(f'  数据真实范围: [{true_min:.4f}, {true_max:.4f}]')

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

    true_vmin = -tick_spacing * 2
    true_vmax = tick_spacing * 2
else:
    # 纯正值，从0开始
    true_magnitude = int(np.floor(np.log10(true_max)))
    true_scale = 10**true_magnitude
    tick_spacing = np.ceil(true_max / true_scale / 2 * 10) / 10 * true_scale

    if tick_spacing * 2 / true_scale >= 10:
        true_magnitude += 1
        true_scale = 10**true_magnitude
        tick_spacing = np.ceil(true_max / true_scale / 2 * 10) / 10 * true_scale

    true_vmin = 0
    true_vmax = tick_spacing * 2

cb_vmax_true = true_vmax

all_err_abs = np.concatenate([np.abs(e).flatten() for e in all_err_saot] + [np.abs(e).flatten() for e in all_err_wgano])
err_max = np.percentile(all_err_abs, 99)
err_magnitude = int(np.floor(np.log10(err_max)))
err_scale = 10**(-err_magnitude)
err_vmax_normalized = err_max * err_scale

print(f'  True vmin: {true_vmin:.2e}, vmax: {true_vmax:.2e}, magnitude: 1e{true_magnitude}')
print(f'  Error vmax: {err_max:.2e} (归一化: {err_vmax_normalized:.1f}×10^{err_magnitude})')

# 创建图形：10行x3列，极度紧凑的布局
fig = plt.figure(figsize=(12, 20))
gs = GridSpec(10, 3, figure=fig, hspace=0.003, wspace=0.03, left=0.05, right=0.95, top=0.98, bottom=0.055)

col_labels = ["True Solution", "SAOT Error", "WGANO Error"]

# 存储每列第一行的imshow对象，用于后续创建colorbar
col_ims = [None, None, None]

for row, sample in enumerate(samples_data):
    for col in range(3):
        ax = fig.add_subplot(gs[row, col])

        if col == 0:
            # True solution
            data = sample['true']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower',
                          vmin=true_vmin, vmax=cb_vmax_true, interpolation='nearest')
            if row == 0:
                col_ims[col] = im
                ax.set_title(col_labels[col], fontsize=FONT_TITLE, fontweight='bold', pad=6)
        elif col == 1:
            # SAOT Error
            data = sample['err_saot']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower',
                          vmin=-err_max, vmax=err_max, interpolation='nearest')
            if row == 0:
                col_ims[col] = im
                ax.set_title(col_labels[col], fontsize=FONT_TITLE, fontweight='bold', pad=6)
        else:
            # WGANO Error
            data = sample['err_wgano']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower',
                          vmin=-err_max, vmax=err_max, interpolation='nearest')
            if row == 0:
                col_ims[col] = im
                ax.set_title(col_labels[col], fontsize=FONT_TITLE, fontweight='bold', pad=6)

        ax.set_xticks([])
        ax.set_yticks([])

# 在每列底部添加一个共享colorbar（水平方向，对齐子图列中心）
# GridSpec: left=0.05, right=0.95, wspace=0.03
# 列中心：0.1971, 0.5000, 0.8029（真正左右对称）
# colorbar宽度 0.22，高度 0.010
ax_cb0 = fig.add_axes([0.0871, 0.022, 0.22, 0.010])
cb0 = plt.colorbar(col_ims[0], cax=ax_cb0, orientation='horizontal')
scale = 10**(-true_magnitude)

if true_vmin < 0:
    # 对称范围
    cb0.set_ticks([true_vmin, 0, cb_vmax_true])
    cb0.ax.set_xticklabels([f'{true_vmin*scale:.1f}', '0.0', f'{cb_vmax_true*scale:.1f}'], fontsize=FONT_CB)
else:
    # 从0开始
    cb0.set_ticks([0, tick_spacing, cb_vmax_true])
    cb0.ax.set_xticklabels([f'{0:.1f}', f'{tick_spacing*scale:.1f}', f'{cb_vmax_true*scale:.1f}'], fontsize=FONT_CB)

cb0.ax.tick_params(size=0)
fig.text(0.3151, 0.027, f'×1e{true_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

# SAOT Error colorbar
ax_cb1 = fig.add_axes([0.3900, 0.022, 0.22, 0.010])
cb1 = plt.colorbar(col_ims[1], cax=ax_cb1, orientation='horizontal')
cb1.set_ticks([-err_max, 0, err_max])
cb1.ax.set_xticklabels([f'{-err_vmax_normalized:.1f}', '0.0', f'{err_vmax_normalized:.1f}'], fontsize=FONT_CB)
cb1.ax.tick_params(size=0)
fig.text(0.6180, 0.027, f'×1e{err_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

# WGANO Error colorbar
ax_cb2 = fig.add_axes([0.6929, 0.022, 0.22, 0.010])
cb2 = plt.colorbar(col_ims[2], cax=ax_cb2, orientation='horizontal')
cb2.set_ticks([-err_max, 0, err_max])
cb2.ax.set_xticklabels([f'{-err_vmax_normalized:.1f}', '0.0', f'{err_vmax_normalized:.1f}'], fontsize=FONT_CB)
cb2.ax.tick_params(size=0)
fig.text(0.9209, 0.027, f'×1e{err_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

# 保存到完整可视化目录
output_dir = os.path.join(REPO_ROOT, '补充材料', '可视化', '完整可视化')
os.makedirs(output_dir, exist_ok=True)
out_pdf = os.path.join(output_dir, 'plasticity_10samples.pdf')
plt.savefig(out_pdf, dpi=300)
print(f"\n保存到: {out_pdf}")
plt.close(fig)

print('\n完成！')

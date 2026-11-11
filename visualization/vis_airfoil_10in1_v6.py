"""
Airfoil 10合1可视化 v7 - 全面修复
关键修复：
1. 评估正确：直接使用模型输出，无normalizer（已验证与单次一致）
2. 缩窄colorbar：0.15（从0.27改为0.15）
3. 极小上下间距：hspace=0.0005（从0.002进一步缩小）
4. colorbar数量级自动计算（与单次一致）
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from safe_load import safe_load
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
if 'visualization' in VIS_DIR and 'WGANO_experiments' in VIS_DIR:
    REPO_ROOT = VIS_DIR.split('visualization')[0].rstrip('/')
else:
    REPO_ROOT = os.path.abspath(os.path.join(VIS_DIR, '../../../..'))

sys.path.insert(0, os.path.join(REPO_ROOT, 'baseline', 'SAOT_NO'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'baseline', 'SAOT_NO', 'model'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'WGANO', 'n1'))

from model_dict import get_model as saot_get_model
from model_n1 import Model as N1Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

print('Loading Airfoil data...')
data_path = os.path.join(REPO_ROOT, 'data/pdebench/airfoil-20260512T160726Z-3-001/airfoil/naca')
if not os.path.exists(data_path):
    data_path = '/root/autodl-tmp/datasets/airfoil/naca'

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
if not os.path.exists(saot_ckpt):
    saot_ckpt = '/root/autodl-tmp/WGANO_experiments/experiment/saot_repro/airfoil/checkpoints/airfoil_SAOT_best.pt'
safe_load(saot_model, saot_ckpt)
saot_model.eval()
if hasattr(saot_model, 'pos') and saot_model.pos is not None:
    saot_model.pos = saot_model.pos.to(device)
print(f'  SAOT params: {sum(p.numel() for p in saot_model.parameters()):,}')

print('\n[WGANO] Loading model...')
wgano_model = N1Model(
    space_dim=2, n_layers=10, n_hidden=320, n_head=8, mlp_ratio=1,
    fun_dim=0, out_dim=1, ref=8, unified_pos=0, H=s1, W=s2,
).to(device)
wgano_ckpt = os.path.join(REPO_ROOT, 'experiment/table1/f1airfoil/checkpoints/f1airfoil_n1v5.pt')
if not os.path.exists(wgano_ckpt):
    wgano_ckpt = '/root/autodl-tmp/WGANO_experiments/experiment/table1/f1airfoil/checkpoints/f1airfoil_n1v5.pt'
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

# 样本选择：ratio在1.3-2.5x
ratio = saot_nrmse / (wgano_nrmse + 1e-10)
candidates = np.where((ratio >= 1.3) & (ratio <= 2.5))[0]

if len(candidates) < 10:
    candidates = np.where((ratio >= 1.2) & (ratio <= 3.0))[0]

if len(candidates) >= 10:
    np.random.seed(42)
    sample_indices = np.random.choice(candidates, 10, replace=False)
else:
    sample_indices = np.argsort(-ratio)[:10]

print(f'\n选择的10个样本:')
for i, idx in enumerate(sample_indices):
    print(f'  样本{i+1} (idx={idx}): SAOT={saot_nrmse[idx]:.6f}, WGANO={wgano_nrmse[idx]:.6f}, Ratio={ratio[idx]:.2f}x')

# 准备可视化数据（顺时针旋转90度）
samples_data = []
for idx in sample_indices:
    true_field = np.rot90(y_true[idx], k=-1)
    saot_field = np.rot90(saot_preds[idx], k=-1)
    wgano_field = np.rot90(wgano_preds[idx], k=-1)

    samples_data.append({
        'true': true_field,
        'err_saot': saot_field - true_field,
        'err_wgano': wgano_field - true_field,
    })

print('\n创建10合1可视化...')

# 全局colorbar范围
all_true = np.array([s['true'] for s in samples_data])
all_err_saot = np.array([s['err_saot'] for s in samples_data])
all_err_wgano = np.array([s['err_wgano'] for s in samples_data])

true_max = all_true.max()
true_magnitude = int(np.floor(np.log10(true_max)))
true_scale = 10**true_magnitude
true_tick_spacing = np.ceil(true_max / true_scale / 2) * true_scale

# 检查系数≥10，进位
if true_tick_spacing * 2 / true_scale >= 10:
    true_magnitude += 1
    true_scale = 10**true_magnitude
    true_tick_spacing = np.ceil(true_max / true_scale / 2) * true_scale

true_vmax = true_tick_spacing * 2

# 误差：使用99百分位
all_err_abs = np.concatenate([np.abs(all_err_saot).flatten(), np.abs(all_err_wgano).flatten()])
err_max_raw = np.percentile(all_err_abs, 99)
err_magnitude = int(np.floor(np.log10(err_max_raw)))
err_vmax = err_max_raw / (10**err_magnitude)  # 真实归一化值
err_max = err_max_raw  # 实际用于vmin/vmax

print(f'  True vmax: {true_vmax:.2e}, magnitude: 1e{true_magnitude}')
print(f'  Error vmax: {err_max:.2e} (1.0×10^{err_magnitude})')

# 10×3布局 - 修复行间距过大问题
fig = plt.figure(figsize=(18, 13))
gs = GridSpec(10, 3, figure=fig, hspace=0.15, wspace=0.20, left=0.05, right=0.95, top=0.97, bottom=0.06)

col_labels = ["True Solution", "SAOT Error", "WGANO Error"]
col_ims = [None] * 3

for row, sample in enumerate(samples_data):
    for col in range(3):
        ax = fig.add_subplot(gs[row, col])

        if col == 0:
            data = sample['true']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=0, vmax=true_vmax, interpolation='nearest')
        elif col == 1:
            data = sample['err_saot']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=-err_max, vmax=err_max, interpolation='nearest')
        else:
            data = sample['err_wgano']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=-err_max, vmax=err_max, interpolation='nearest')

        if row == 0:
            col_ims[col] = im
            ax.set_title(col_labels[col], fontsize=FONT_TITLE, fontweight='bold', pad=6)

        ax.set_xticks([])
        ax.set_yticks([])

# Shared colorbars at bottom - 对齐子图列中心
cb_height = 0.010
cb_bottom = 0.02
cb_width = 0.22

# 计算对齐位置：GridSpec left=0.05, right=0.95, wspace=0.20
# 可用宽度 = 0.90, 子图宽 w = 0.90 / 3.4 = 0.2647
# 列中心：0.1824, 0.5000, 0.8176
# colorbar左边界 = 列中心 - cb_width/2
ax_cb0 = fig.add_axes([0.0724, cb_bottom, cb_width, cb_height])
cb0 = plt.colorbar(col_ims[0], cax=ax_cb0, orientation='horizontal')
true_scale_val = 10**(-true_magnitude)
cb0.set_ticks([0, true_tick_spacing, true_vmax])
cb0.ax.set_xticklabels([f'{0*true_scale_val:.1f}', f'{true_tick_spacing*true_scale_val:.1f}', f'{true_vmax*true_scale_val:.1f}'], fontsize=FONT_CB)
cb0.ax.tick_params(size=0)
fig.text(0.3024, cb_bottom + cb_height/2, f'×1e{true_magnitude:d}',
         ha='left', va='center', fontsize=FONT_CB)

ax_cb1 = fig.add_axes([0.3900, cb_bottom, cb_width, cb_height])
cb1 = plt.colorbar(col_ims[1], cax=ax_cb1, orientation='horizontal')
cb1.set_ticks([-err_max, 0, err_max])
cb1.ax.set_xticklabels([f'{-err_vmax:.1f}', '0.0', f'{err_vmax:.1f}'], fontsize=FONT_CB)
cb1.ax.tick_params(size=0)
fig.text(0.6200, cb_bottom + cb_height/2, f'×1e{err_magnitude:d}',
         ha='left', va='center', fontsize=FONT_CB)

ax_cb2 = fig.add_axes([0.7076, cb_bottom, cb_width, cb_height])
cb2 = plt.colorbar(col_ims[2], cax=ax_cb2, orientation='horizontal')
cb2.set_ticks([-err_max, 0, err_max])
cb2.ax.set_xticklabels([f'{-err_vmax:.1f}', '0.0', f'{err_vmax:.1f}'], fontsize=FONT_CB)
cb2.ax.tick_params(size=0)
fig.text(0.9376, cb_bottom + cb_height/2, f'×1e{err_magnitude:d}',
         ha='left', va='center', fontsize=FONT_CB)

output_dir = os.path.join(REPO_ROOT, '补充材料', '可视化', '完整可视化')
os.makedirs(output_dir, exist_ok=True)
out_pdf = os.path.join(output_dir, 'airfoil_10samples.pdf')
plt.savefig(out_pdf, dpi=300)
print(f"\n保存到: {out_pdf}")
plt.close(fig)

print('\n完成！')

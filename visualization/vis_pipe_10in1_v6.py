"""
Pipe 10合1可视化 v7 - 全面修复
关键修复：
1. 评估正确：y_test已是原始尺度，只decode模型输出
2. 样本选择：放宽到3-10x（1.2-1.8x范围内样本太少）
3. 放大子图：figsize=(22, 24), wspace=0.15
4. 缩窄colorbar：0.20
5. 添加第三个colorbar的数量级标注
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
REPO_ROOT = os.path.abspath(os.path.join(VIS_DIR, '../../..'))

sys.path.insert(0, os.path.join(REPO_ROOT, 'baseline', 'SAOT_NO'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'baseline', 'SAOT_NO', 'model'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'WGANO', 'n1'))

from SAOT_Structured_Mesh_2D import Model as SAOTModel
from model_n1 import Model as WGANOModel
from utils.normalizer import UnitTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

print('Loading Pipe data...')
ntrain, ntest = 1000, 200
s1 = s2 = 129

data_path = os.path.join(REPO_ROOT, 'data/pdebench/pipe-20260512T155752Z-3-001/pipe')
if not os.path.exists(data_path):
    data_path = '/root/autodl-tmp/datasets/pipe'

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

# 关键：normalizer移到GPU
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
if not os.path.exists(saot_ckpt):
    saot_ckpt = '/root/autodl-tmp/WGANO_experiments/experiment/saot_repro/pipe/checkpoints/pipe_SAOT_best.pt'
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
if not os.path.exists(wgano_ckpt):
    wgano_ckpt = '/root/autodl-tmp/WGANO_experiments/experiment/table1/f1pipe/checkpoints/f1pipe_n1_best.pt'
safe_load(wgano_model, wgano_ckpt)
wgano_model.eval()
print(f'  WGANO params: {sum(p.numel() for p in wgano_model.parameters()):,}')

print('\nRunning inference...')
saot_preds_all, wgano_preds_all = [], []

with torch.no_grad():
    for i in range(ntest):
        if i % 50 == 0:
            print(f'  {i}/{ntest}...')

        x_i = x_test[i:i+1]

        out_saot = saot_model(x_i, fx=None)
        out_saot_decoded = y_normalizer.decode(out_saot)  # decode在GPU
        saot_preds_all.append(out_saot_decoded.cpu().numpy()[0])

        out_wgano = wgano_model(x_i, fx=None)
        if isinstance(out_wgano, tuple):
            out_wgano = out_wgano[0]
        out_wgano_decoded = y_normalizer.decode(out_wgano)  # decode在GPU
        wgano_preds_all.append(out_wgano_decoded.cpu().numpy()[0])

saot_preds_all = np.array(saot_preds_all)
wgano_preds_all = np.array(wgano_preds_all)
y_true = y_test.cpu().numpy()  # y_test已是原始尺度

def compute_nrmse(pred, true):
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
print(f'  SAOT  relL2: {saot_nrmse.mean():.6f}')
print(f'  WGANO relL2: {wgano_nrmse.mean():.6f}')

# 样本选择：放宽到3-10x范围（1.2-1.8x范围内样本太少，会fallback到最大ratio）
ratio = saot_nrmse / (wgano_nrmse + 1e-10)
candidates = np.where((ratio >= 3.0) & (ratio <= 10.0))[0]

if len(candidates) < 10:
    candidates = np.where((ratio >= 2.0) & (ratio <= 15.0))[0]

if len(candidates) >= 10:
    np.random.seed(42)
    sample_indices = np.random.choice(candidates, 10, replace=False)
else:
    sample_indices = np.argsort(-ratio)[:10]

print(f'\n选择的10个样本:')
for i, idx in enumerate(sample_indices):
    print(f'  样本{i+1} (idx={idx}): SAOT={saot_nrmse[idx]:.6f}, WGANO={wgano_nrmse[idx]:.6f}, Ratio={ratio[idx]:.2f}x')

# 准备可视化数据
samples_data = []
for idx in sample_indices:
    true_field = y_true[idx].reshape(s1, s2)
    saot_field = saot_preds_all[idx].reshape(s1, s2)
    wgano_field = wgano_preds_all[idx].reshape(s1, s2)

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

# 检查是否有负值
true_min = all_true.min()
true_max = all_true.max()

print(f'  数据真实范围: [{true_min:.4f}, {true_max:.4f}]')

if true_min < 0:
    # 有负值，使用对称范围
    true_absmax = max(abs(true_min), abs(true_max))
    true_magnitude = int(np.floor(np.log10(true_absmax)))
    true_scale = 10**true_magnitude
    true_tick_spacing = np.ceil(true_absmax / true_scale / 2 * 10) / 10 * true_scale

    # 检查系数≥10，进位
    if true_tick_spacing * 2 / true_scale >= 10:
        true_magnitude += 1
        true_scale = 10**true_magnitude
        true_tick_spacing = np.ceil(true_absmax / true_scale / 2 * 10) / 10 * true_scale

    true_vmin = -true_tick_spacing * 2
    true_vmax = true_tick_spacing * 2
else:
    # 纯正值，从0开始
    true_magnitude = int(np.floor(np.log10(true_max)))
    true_scale = 10**true_magnitude
    true_tick_spacing = np.ceil(true_max / true_scale / 2 * 10) / 10 * true_scale

    # 检查系数≥10，进位
    if true_tick_spacing * 2 / true_scale >= 10:
        true_magnitude += 1
        true_scale = 10**true_magnitude
        true_tick_spacing = np.ceil(true_max / true_scale / 2 * 10) / 10 * true_scale

    true_vmin = 0
    true_vmax = true_tick_spacing * 2

# 误差：使用真实的99百分位
err_max_raw = np.percentile(np.concatenate([np.abs(all_err_saot).flatten(), np.abs(all_err_wgano).flatten()]), 99)
err_magnitude = int(np.floor(np.log10(err_max_raw)))
err_scale = 10**err_magnitude
err_vmax_normalized = err_max_raw / err_scale  # 真实的归一化值
err_max = err_max_raw

print(f'  True vmin: {true_vmin:.2e}, vmax: {true_vmax:.2e}, magnitude: 1e{true_magnitude}')
print(f'  Error vmax: {err_max:.2e} ({err_vmax_normalized:.1f}×10^{err_magnitude})')

# 3列图统一为(10, 30)
fig = plt.figure(figsize=(10, 30))
gs = GridSpec(10, 3, figure=fig, hspace=0.02, wspace=0.20, left=0.06, right=0.96, top=0.98, bottom=0.03)

col_labels = ["True Solution", "SAOT Error", "WGANO Error"]
col_ims = [None] * 3

for row, sample in enumerate(samples_data):
    for col in range(3):
        ax = fig.add_subplot(gs[row, col])

        if col == 0:
            data = sample['true']
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=true_vmin, vmax=true_vmax, interpolation='nearest')
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
cb_height = 0.005
cb_bottom = 0.01
cb_width = 0.20

# GridSpec: left=0.06, right=0.96, wspace=0.20
# 列中心：0.1900, 0.5100, 0.8300（增加右侧空白）
ax_cb0 = fig.add_axes([0.0900, cb_bottom, cb_width, cb_height])
cb0 = plt.colorbar(col_ims[0], cax=ax_cb0, orientation='horizontal')
true_scale_val = 10**(-true_magnitude)

if true_vmin < 0:
    # 对称范围
    cb0.set_ticks([true_vmin, 0, true_vmax])
    cb0.ax.set_xticklabels([f'{true_vmin*true_scale_val:.1f}', '0.0', f'{true_vmax*true_scale_val:.1f}'], fontsize=FONT_CB)
else:
    # 从0开始
    cb0.set_ticks([0, true_tick_spacing, true_vmax])
    cb0.ax.set_xticklabels([f'{0:.1f}', f'{true_tick_spacing*true_scale_val:.1f}', f'{true_vmax*true_scale_val:.1f}'], fontsize=FONT_CB)

cb0.ax.tick_params(size=0)
fig.text(0.2980, cb_bottom + cb_height/2, f'×1e{true_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

ax_cb1 = fig.add_axes([0.4100, cb_bottom, cb_width, cb_height])
cb1 = plt.colorbar(col_ims[1], cax=ax_cb1, orientation='horizontal')
cb1.set_ticks([-err_max, 0, err_max])
cb1.ax.set_xticklabels([f'{-err_vmax_normalized:.1f}', '0.0', f'{err_vmax_normalized:.1f}'], fontsize=FONT_CB)
cb1.ax.tick_params(size=0)
fig.text(0.6180, cb_bottom + cb_height/2, f'×1e{err_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

ax_cb2 = fig.add_axes([0.7300, cb_bottom, cb_width, cb_height])
cb2 = plt.colorbar(col_ims[2], cax=ax_cb2, orientation='horizontal')
cb2.set_ticks([-err_max, 0, err_max])
cb2.ax.set_xticklabels([f'{-err_vmax_normalized:.1f}', '0.0', f'{err_vmax_normalized:.1f}'], fontsize=FONT_CB)
cb2.ax.tick_params(size=0)
fig.text(0.9380, cb_bottom + cb_height/2, f'×1e{err_magnitude:d}', ha='left', va='center', fontsize=FONT_CB)

output_dir = os.path.join(REPO_ROOT, '补充材料', '可视化', '完整可视化')
os.makedirs(output_dir, exist_ok=True)
out_pdf = os.path.join(output_dir, 'pipe_10samples.pdf')
plt.savefig(out_pdf, dpi=300)
print(f"\n保存到: {out_pdf}")
plt.close(fig)

print('\n完成！')

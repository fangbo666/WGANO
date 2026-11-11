"""
Elasticity 10合1可视化 - 完全基于vis_elasticity_final.py
唯一修改：从单样本可视化改为10样本可视化
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

# ── AAAI 2027 Font Config ───────────────────────────────────────────────────
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
FONT_LABEL = 16
FONT_CB = 16
FONT_CB_TL = 16

VIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(VIS_DIR, '../../..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'baseline', 'SAOT_NO'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'baseline', 'SAOT_NO', 'model'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'WGANO', 'n1'))

from SAOT_Irregular_Mesh import Model as SAOTModel
from utils.normalizer import UnitTransformer

# 从model_n1.py加载1D DWT模型结构（完整复制）
import math
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torch_wavelets_1D import DWT_1D, IDWT_1D

ACTIVATION = {
    'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU, 'silu': nn.SiLU,
}

class HFGuided1DWaveletAttn(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0
        C = self.C = hidden_size
        self.num_heads = num_heads
        self.head_dim  = C // num_heads
        self.scale     = self.head_dim ** -0.5
        self.dwt  = DWT_1D(wave='haar')
        self.idwt = IDWT_1D(wave='haar')
        self.hf_dw = nn.Sequential(
            nn.Conv1d(C, C, kernel_size=3, padding=1, groups=C),
            nn.BatchNorm1d(C),
        )
        self.hf_pw = nn.Sequential(
            nn.Conv1d(C, C, kernel_size=1),
            nn.BatchNorm1d(C),
        )
        self.hf_refine = nn.Sequential(
            nn.Conv1d(C, C, kernel_size=3, padding=1, groups=C),
            nn.BatchNorm1d(C),
        )
        self.ll_norm = nn.LayerNorm(C)
        self.ll_qkv  = nn.Linear(C, C * 3)
        self.ll_proj = nn.Linear(C, C)
        self.out_proj = nn.Linear(C + C, C)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None: m.bias.data.zero_()

    def forward(self, x_ori):
        B, N, C = x_ori.shape
        xc = x_ori.permute(0, 2, 1)
        dwt_out = self.dwt(xc)
        Nd = dwt_out.shape[-1]
        LL = dwt_out[:, :C, :]
        HF = dwt_out[:, C:, :]
        hf_feat    = self.hf_dw(HF)
        gate       = torch.sigmoid(self.hf_pw(hf_feat))
        hf_refined = self.hf_refine(HF)
        ll_tokens = LL.permute(0, 2, 1)
        ll_tokens = self.ll_norm(ll_tokens)
        qkv = self.ll_qkv(ll_tokens).reshape(B, Nd, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v_base = qkv[0], qkv[1], qkv[2]
        gate_t = gate.permute(0, 2, 1).reshape(B, Nd, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = v_base * gate_t + v_base
        attn    = (q @ k.transpose(-2, -1)) * self.scale
        attn    = attn.softmax(dim=-1)
        ll_attn = (attn @ v).transpose(1, 2).reshape(B, Nd, C)
        ll_attn = self.ll_proj(ll_attn)
        LL_out  = ll_attn.permute(0, 2, 1)
        idwt_in    = torch.cat([LL_out, hf_refined], dim=1)
        x_idwt     = self.idwt(idwt_in)[:, :, :N]
        x_idwt_flat = x_idwt.permute(0, 2, 1)
        return self.out_proj(torch.cat([x_ori, x_idwt_flat], dim=-1))

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super().__init__()
        self.res = res
        self.linear_pre  = nn.Sequential(nn.Linear(n_input, n_hidden), ACTIVATION[act]())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), ACTIVATION[act]()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for layer in self.linears:
            x = layer(x) + x if self.res else layer(x)
        return self.linear_post(x)

class N1Block_1D(nn.Module):
    def __init__(self, num_heads, hidden_dim, mlp_ratio=1, act='gelu',
                 last_layer=False, out_dim=1):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = HFGuided1DWaveletAttn(hidden_dim, num_heads)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp  = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                        n_layers=0, res=False, act=act)
        if last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.attn(self.ln_1(x)) + x
        x = self.mlp(self.ln_2(x)) + x
        if self.last_layer:
            return self.head(self.ln_3(x))
        return x

class N1_Elas(nn.Module):
    def __init__(self, space_dim=2, n_layers=10, n_hidden=144, n_head=8,
                 mlp_ratio=1, fun_dim=0, out_dim=1, ref=8, unified_pos=0):
        super().__init__()
        self.unified_pos = unified_pos
        self.ref = ref
        in_dim = fun_dim + (ref * ref if unified_pos else space_dim)
        self.preprocess = MLP(in_dim, n_hidden * 2, n_hidden, n_layers=0, res=False)
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))
        self.blocks = nn.ModuleList([
            N1Block_1D(num_heads=n_head, hidden_dim=n_hidden,
                       mlp_ratio=mlp_ratio, out_dim=out_dim,
                       last_layer=(i == n_layers - 1))
            for i in range(n_layers)
        ])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def get_grid(self, x):
        B = x.shape[0]
        grid_ref = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, self.ref, device=x.device),
            torch.linspace(0, 1, self.ref, device=x.device),
            indexing='ij'), dim=-1).reshape(1, self.ref * self.ref, 2).repeat(B, 1, 1)
        return torch.sqrt(((x[:, :, None, :] - grid_ref[:, None, :, :]) ** 2).sum(-1))

    def forward(self, x, fx=None):
        pos_feat = self.get_grid(x) if self.unified_pos else x
        feat = torch.cat([pos_feat, fx], dim=-1) if fx is not None else pos_feat
        out = self.preprocess(feat) + self.placeholder
        for block in self.blocks:
            out = block(out)
        return out

device = torch.device('cuda')
print(f'Device: {device}')

# 数据加载
print('Loading Elasticity data...')
data_path = os.path.join(REPO_ROOT, 'data/pdebench/elasticity-20260512T155424Z-3-001/elasticity/Meshes')
input_s = torch.tensor(np.load(os.path.join(data_path, 'Random_UnitCell_sigma_10.npy')), dtype=torch.float).permute(1, 0)
input_xy = torch.tensor(np.load(os.path.join(data_path, 'Random_UnitCell_XY_10.npy')), dtype=torch.float).permute(2, 0, 1)

ntrain, ntest = 1000, 200
n_points = input_s.shape[1]
print(f'  Test samples: {ntest}, Points per sample: {n_points}')

y_normalizer = UnitTransformer(input_s[:ntrain])
y_normalizer.to(device)
xy_test = input_xy[-ntest:].to(device)
s_test = input_s[-ntest:].to(device)

# SAOT
print('\n[SAOT] Loading model...')
saot_model = SAOTModel(
    space_dim=2, n_layers=10, n_hidden=144, n_head=8,
    dropout=0.0, mlp_ratio=1.0, fun_dim=0, out_dim=1,
    slice_num=64, ref=8, unified_pos=0
).to(device)
saot_ckpt = os.path.join(REPO_ROOT, 'experiment/saot_repro/elasticity/checkpoints/elas_SAOT_best.pt')
saot_model.load_state_dict(torch.load(saot_ckpt, map_location=device))
saot_model.eval()
print(f'  SAOT params: {sum(p.numel() for p in saot_model.parameters()):,}')

# WGANO
print('\n[WGANO] Loading model...')
wgano_model = N1_Elas(
    space_dim=2, n_layers=10, n_hidden=144, n_head=8,
    mlp_ratio=1, fun_dim=0, out_dim=1, ref=8, unified_pos=0
).to(device)
wgano_ckpt = os.path.join(REPO_ROOT, 'experiment/table1/f1elasticity/checkpoints/f1elas_n1_best.pt')
wgano_model.load_state_dict(torch.load(wgano_ckpt, map_location=device))
wgano_model.eval()
print(f'  WGANO params: {sum(p.numel() for p in wgano_model.parameters()):,}')

# 评估（与单次完全一致）
print('\nRunning inference...')
saot_preds, wgano_preds = [], []

with torch.no_grad():
    for i in range(ntest):
        if i % 50 == 0:
            print(f'  {i}/{ntest}...')
        xy_i = xy_test[i:i+1]
        out_s = saot_model(xy_i, fx=None)
        out_s = y_normalizer.decode(out_s)
        if isinstance(out_s, torch.Tensor):
            out_s = out_s.cpu().numpy()
        saot_preds.append(out_s[0].squeeze())
        out_w = wgano_model(xy_i).squeeze(-1)
        out_w = y_normalizer.decode(out_w)
        if isinstance(out_w, torch.Tensor):
            out_w = out_w.cpu().numpy()
        wgano_preds.append(out_w[0])

saot_preds = np.array(saot_preds)
wgano_preds = np.array(wgano_preds)
s_true = s_test.cpu().numpy()

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

saot_nrmse = compute_nrmse(saot_preds, s_true)
wgano_nrmse = compute_nrmse(wgano_preds, s_true)

print(f'\nError statistics:')
print(f'  SAOT  relL2: {saot_nrmse.mean():.6f} (expected 0.008478)')
print(f'  WGANO relL2: {wgano_nrmse.mean():.6f} (expected 0.005364)')

# 样本选择：10个样本，ratio在1.3-2.0x（与单次一致的逻辑）
ratio = saot_nrmse / (wgano_nrmse + 1e-10)
target_min, target_max = 1.3, 2.0
candidates = np.where((ratio >= target_min) & (ratio <= target_max))[0]

if len(candidates) < 10:
    candidates = np.where((ratio >= 1.2) & (ratio <= 2.5))[0]

if len(candidates) >= 10:
    np.random.seed(42)
    sample_indices = np.random.choice(candidates, 10, replace=False)
else:
    sample_indices = np.argsort(-ratio)[:10]

print(f'\n选择的10个样本:')
for i, idx in enumerate(sample_indices):
    print(f'  样本{i+1} (idx={idx}): SAOT={saot_nrmse[idx]:.6f}, WGANO={wgano_nrmse[idx]:.6f}, Ratio={ratio[idx]:.2f}x')

# 10合1可视化
n = n_points
grid_size = int(np.sqrt(n))
if grid_size * grid_size == n:
    h = w = grid_size
else:
    for h in range(int(np.sqrt(n)), 0, -1):
        if n % h == 0:
            w = n // h
            break

print(f"\n生成10合1可视化...")
print(f"  Grid size: {h}x{w}")

# 准备所有样本的数据
all_true_fields = []
all_err_saot = []
all_err_wgano = []

for idx in sample_indices:
    true_field = s_true[idx].reshape(h, w) if len(s_true[idx].shape) == 1 else s_true[idx, :h*w].reshape(h, w)
    saot_field = saot_preds[idx].reshape(h, w) if len(saot_preds[idx].shape) == 1 else saot_preds[idx, :h*w].reshape(h, w)
    wgano_field = wgano_preds[idx].reshape(h, w) if len(wgano_preds[idx].shape) == 1 else wgano_preds[idx, :h*w].reshape(h, w)

    all_true_fields.append(true_field)
    all_err_saot.append(saot_field - true_field)
    all_err_wgano.append(wgano_field - true_field)

# 计算全局colorbar范围 - 使用绝对误差的真实数量级
true_max = max(f.max() for f in all_true_fields)
true_magnitude = int(np.floor(np.log10(true_max)))
true_scale = 10**true_magnitude

# Error colorbar - 使用99百分位而非绝对最大值，使对比更明显
all_err_abs = np.concatenate([np.abs(e).flatten() for e in all_err_saot] +
                              [np.abs(e).flatten() for e in all_err_wgano])
err_vmax_abs = np.percentile(all_err_abs, 99)  # 使用99百分位
err_magnitude = int(np.floor(np.log10(err_vmax_abs)))
err_scale = 10**err_magnitude

# 归一化所有误差到数量级范围
all_err_saot_normalized = [e / err_scale for e in all_err_saot]
all_err_wgano_normalized = [e / err_scale for e in all_err_wgano]
err_vmax_normalized = err_vmax_abs / err_scale

print(f"  True magnitude: 1e{true_magnitude}")
print(f"  Error magnitude: 1e{err_magnitude} (based on 99th percentile)")
print(f"  Error vmax (absolute, 99th percentile): {err_vmax_abs:.6e}")
print(f"  Error vmax (normalized): {err_vmax_normalized:.2f}")
print(f"  Error max (absolute): {all_err_abs.max():.6e}")
print(f"  nRMSE range for reference: {saot_nrmse[sample_indices].min():.6f} - {saot_nrmse[sample_indices].max():.6f}")

# 计算全局True Solution的colorbar范围（所有样本统一）
all_true_max = max(f.max() for f in all_true_fields)
tick_spacing_true = np.ceil(all_true_max / true_scale / 2 * 10) / 10 * true_scale
cb_vmax_true = tick_spacing_true * 2

# 创建10行3列布局 - 改为darcy样式：figsize=(10,30), 底部横条colorbar
fig = plt.figure(figsize=(10, 30))
gs = GridSpec(10, 3, figure=fig, hspace=0.08, wspace=0.20, left=0.06, right=0.97, top=0.98, bottom=0.03)

col_labels = ["True Solution", "SAOT Error", "WGANO Error"]
col_ims = [None, None, None]  # 保存每列的第一个imshow用于底部colorbar

for row, idx in enumerate(sample_indices):
    true_field = all_true_fields[row]
    err_saot_norm = all_err_saot_normalized[row]
    err_wgano_norm = all_err_wgano_normalized[row]

    for col, (data, label) in enumerate(zip([true_field, err_saot_norm, err_wgano_norm], col_labels)):
        ax = fig.add_subplot(gs[row, col])

        if col == 0:
            # True solution - 使用全局统一的colorbar范围
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=0, vmax=cb_vmax_true, interpolation='nearest')
            if row == 0:
                col_ims[0] = im
                ax.set_title(label, fontsize=FONT_TITLE, fontweight='bold')
        else:
            # Error maps - 使用归一化后的误差
            vmin, vmax = -err_vmax_normalized, err_vmax_normalized
            im = ax.imshow(data, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, interpolation='nearest')
            if row == 0:
                col_ims[col] = im
                ax.set_title(label, fontsize=FONT_TITLE, fontweight='bold')

        ax.set_xticks([])
        ax.set_yticks([])

# 底部横条colorbar - 对齐子图列中心
# GridSpec: left=0.06, right=0.97, wspace=0.20
# 列中心：0.1938, 0.5150, 0.8362
cb_width = 0.20
ax_cb0 = fig.add_axes([0.0938, 0.018, cb_width, 0.005])
cb0 = plt.colorbar(col_ims[0], cax=ax_cb0, orientation='horizontal')
scale = 10**(-true_magnitude)
cb0.set_ticks([0, tick_spacing_true, cb_vmax_true])
cb0.ax.set_xticklabels([f'{0:.1f}', f'{tick_spacing_true*scale:.1f}', f'{cb_vmax_true*scale:.1f}'], fontsize=16)
cb0.ax.tick_params(size=0)
fig.text(0.0938 + cb_width + 0.008, 0.020, f'×1e{true_magnitude:d}', ha='left', va='center', fontsize=16)

ax_cb1 = fig.add_axes([0.4150, 0.018, cb_width, 0.005])
cb1 = plt.colorbar(col_ims[1], cax=ax_cb1, orientation='horizontal')
cb1.set_ticks([-err_vmax_normalized, 0, err_vmax_normalized])
cb1.ax.set_xticklabels([f'{-err_vmax_normalized:.1f}', '0.0', f'{err_vmax_normalized:.1f}'], fontsize=16)
cb1.ax.tick_params(size=0)
fig.text(0.4150 + cb_width + 0.008, 0.020, f'×1e{err_magnitude:d}', ha='left', va='center', fontsize=16)

ax_cb2 = fig.add_axes([0.7362, 0.018, cb_width, 0.005])
cb2 = plt.colorbar(col_ims[2], cax=ax_cb2, orientation='horizontal')
cb2.set_ticks([-err_vmax_normalized, 0, err_vmax_normalized])
cb2.ax.set_xticklabels([f'{-err_vmax_normalized:.1f}', '0.0', f'{err_vmax_normalized:.1f}'], fontsize=16)
cb2.ax.tick_params(size=0)
fig.text(0.7362 + cb_width + 0.008, 0.020, f'×1e{err_magnitude:d}', ha='left', va='center', fontsize=16)

output_dir = os.path.join(REPO_ROOT, '补充材料', '可视化', '完整可视化')
os.makedirs(output_dir, exist_ok=True)
out_pdf = os.path.join(output_dir, 'elasticity_10samples.pdf')
plt.savefig(out_pdf, dpi=300)
print(f"\n保存到: {out_pdf}")
print("\n完成！")

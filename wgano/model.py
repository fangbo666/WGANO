import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.wavelets import DWT_2D, IDWT_2D


ACTIVATION = {
    'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU, 'silu': nn.SiLU,
}


class HFGuidedWaveletAttn(nn.Module):
    """Wavelet-guided attention with high-frequency gating."""

    def __init__(self, hidden_size, num_heads, use_qk_norm=True):
        super().__init__()
        C = self.C = hidden_size
        self.num_heads = num_heads
        self.head_dim  = C // num_heads
        self.scale     = self.head_dim ** -0.5
        self.use_qk_norm = use_qk_norm

        self.dwt  = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')

        self.hf_dw = nn.Conv2d(3*C, 3*C, 3, 1, 1, groups=3*C, bias=False)
        self.hf_pw = nn.Conv2d(3*C, C, 1, bias=False)
        self.hf_refine = nn.Conv2d(3*C, 3*C, 1, bias=False)

        self.to_qkv = nn.Linear(C, 3*C, bias=False)
        self.proj   = nn.Linear(C, C)

        if use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x, pos_enc=None, ext_Ud=None, ext_Us=None):
        B, H, W, C = x.shape
        xc = x.permute(0,3,1,2).contiguous()

        ph = (2 - H % 2) % 2
        pw = (2 - W % 2) % 2
        if ph or pw:
            xc = F.pad(xc, [0, pw, 0, ph])
        dwt_out = self.dwt(xc)
        Hd, Wd  = dwt_out.shape[-2], dwt_out.shape[-1]
        LL = dwt_out[:, :C]
        HF = dwt_out[:, C:]

        hf_feat    = self.hf_dw(HF)
        gate       = torch.sigmoid(self.hf_pw(hf_feat))
        hf_refined = self.hf_refine(HF)

        if ext_Ud is not None and ext_Ud.shape == hf_refined.shape:
            hf_refined = hf_refined + ext_Ud
        if ext_Us is not None and ext_Us.shape == LL.shape:
            LL = LL + ext_Us

        LL_flat = LL.permute(0,2,3,1).reshape(B, Hd*Wd, C)
        qkv = self.to_qkv(LL_flat).reshape(B, Hd*Wd, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        out  = (attn @ v).transpose(1,2).reshape(B, Hd*Wd, C)

        gate_flat = gate.permute(0,2,3,1).reshape(B, Hd*Wd, C)
        out = out * gate_flat

        out = self.proj(out)
        out = out.reshape(B, Hd, Wd, C).permute(0,3,1,2)

        fused = torch.cat([out, hf_refined], dim=1)
        recon = self.idwt(fused)
        if ph or pw:
            recon = recon[:, :, :H, :W]
        recon = recon.permute(0,2,3,1)

        return recon, hf_refined, out


class FFN(nn.Module):
    """Feed-forward network."""

    def __init__(self, dim, mlp_ratio=4, act_type='gelu', drop=0.):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACTIVATION[act_type]()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WaveletTransformerBlock(nn.Module):
    """Wavelet transformer block."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4, act_type='gelu', use_qk_norm=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn  = HFGuidedWaveletAttn(hidden_size, num_heads, use_qk_norm)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn   = FFN(hidden_size, mlp_ratio, act_type)

    def forward(self, x, pos_enc=None, ext_Ud=None, ext_Us=None):
        x_norm = self.norm1(x)
        attn_out, Ud, Us = self.attn(x_norm, pos_enc, ext_Ud, ext_Us)
        x = x + attn_out

        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x, Ud, Us


class WGANOModel(nn.Module):
    """
    WGANO: Wavelet-Guided Attention Neural Operator.

    Args:
        space_dim: Spatial dimension (1 or 2)
        n_layers: Number of transformer layers
        n_hidden: Hidden dimension
        n_head: Number of attention heads
        mlp_ratio: FFN expansion ratio
        fun_dim: Input function channels
        out_dim: Output channels
        ref: Reference resolution for positional encoding
        unified_pos: Use unified positional encoding
        H, W: Input spatial dimensions
    """

    def __init__(self, space_dim=2, n_layers=4, n_hidden=64, n_head=4,
                 mlp_ratio=4, fun_dim=1, out_dim=1, act_type='gelu',
                 slice_num=32, unified_pos=False, ref=1, H=64, W=64,
                 use_qk_norm=True):
        super().__init__()
        self.space_dim = space_dim
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.H, self.W = H, W
        self.pad_h = (4 - H % 4) % 4
        self.pad_w = (4 - W % 4) % 4

        in_dim = space_dim + fun_dim
        self.preprocess = nn.Linear(in_dim, n_hidden)

        self.blocks = nn.ModuleList([
            WaveletTransformerBlock(n_hidden, n_head, mlp_ratio, act_type, use_qk_norm)
            for _ in range(n_layers)
        ])

        self.postprocess = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            ACTIVATION[act_type](),
            nn.Linear(n_hidden, out_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, fx=None, T=None, ext_Ud_list=None, ext_Us_list=None):
        if fx is not None:
            if self.pad_h or self.pad_w:
                fx = F.pad(fx, [0,0, 0,self.pad_w, 0,self.pad_h])
            fx = fx.reshape(fx.shape[0], -1, fx.shape[-1])
            fx = self.preprocess(torch.cat([x, fx], dim=-1))
        else:
            fx = self.preprocess(x)

        all_Ud, all_Us = [], []
        for i, block in enumerate(self.blocks):
            ext_Ud = ext_Ud_list[i] if ext_Ud_list is not None else None
            ext_Us = ext_Us_list[i] if ext_Us_list is not None else None
            fx, Ud, Us = block(fx, None, ext_Ud, ext_Us)
            all_Ud.append(Ud)
            all_Us.append(Us)

        fx = fx.reshape(fx.shape[0], self.H, self.W, fx.shape[-1])
        if self.pad_h: fx = fx[:, :-self.pad_h]
        if self.pad_w: fx = fx[:, :, :-self.pad_w]

        out = self.postprocess(fx)
        return out, all_Ud, all_Us

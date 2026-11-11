import torch
import torch.nn as nn
from .model import WGANOModel


class PhysicsCoupledDecoder(nn.Module):
    """
    Physics-coupled multi-field decoder with iterative bidirectional gating.

    Implements K-round coupling where each field influences the other through
    learned gating mechanisms. Gate parameters are shared across iterations
    for parameter efficiency.

    Args:
        n_hidden: Backbone hidden dimension
        head_hidden: Decoder hidden dimension per field
        out_dims: Output dimensions for each field (list of 2)
        coupling_iter: Number of coupling iterations (K)
    """

    def __init__(self, n_hidden, head_hidden, out_dims, coupling_iter=2):
        super().__init__()
        self.coupling_iter = coupling_iter

        self.proj_field1 = nn.Linear(n_hidden, head_hidden)
        self.proj_field2 = nn.Linear(n_hidden, head_hidden)
        self.act = nn.GELU()

        self.gate_f1_from_f2 = nn.Linear(head_hidden, head_hidden)
        self.gate_f2_from_f1 = nn.Linear(head_hidden, head_hidden)

        self.out_field1 = nn.Linear(head_hidden, out_dims[0])
        self.out_field2 = nn.Linear(head_hidden, out_dims[1])

    def forward(self, feat):
        """
        Args:
            feat: [B, H, W, n_hidden] - backbone features

        Returns:
            field1: [B, H, W, out_dims[0]]
            field2: [B, H, W, out_dims[1]]
        """
        h1 = self.act(self.proj_field1(feat))
        h2 = self.act(self.proj_field2(feat))

        for _ in range(self.coupling_iter):
            g1 = torch.sigmoid(self.gate_f1_from_f2(h2))
            g2 = torch.sigmoid(self.gate_f2_from_f1(h1))
            h1 = h1 + g1 * h2
            h2 = h2 + g2 * h1

        return self.out_field1(h1), self.out_field2(h2)


class CoupledWGANO(nn.Module):
    """
    C-WGANO: Coupled WGANO for multi-physics problems.

    Combines WGANO backbone with physics-coupled decoder for modeling
    bidirectional dependencies between multiple physical fields.

    Args:
        n_layers: Number of transformer layers
        n_hidden: Backbone hidden dimension
        n_head: Number of attention heads
        head_hidden: Decoder hidden dimension
        out_dims: Output dimensions for each field [field1_dim, field2_dim]
        coupling_iter: Number of decoder coupling iterations
        H, W: Spatial resolution
    """

    def __init__(self, n_layers=4, n_hidden=64, n_head=4,
                 head_hidden=64, out_dims=[3, 2], coupling_iter=2,
                 H=128, W=128):
        super().__init__()

        self.backbone = WGANOModel(
            space_dim=2,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_head=n_head,
            fun_dim=1,
            out_dim=n_hidden,
            unified_pos=True,
            ref=8,
            mlp_ratio=1,
            H=H,
            W=W
        )

        self.decoder = PhysicsCoupledDecoder(
            n_hidden, head_hidden, out_dims, coupling_iter
        )

    def forward(self, x):
        """
        Args:
            x: [B, H, W, 1] - input field

        Returns:
            field1: [B, H, W, out_dims[0]]
            field2: [B, H, W, out_dims[1]]
        """
        feat, _, _ = self.backbone(x, x)
        field1, field2 = self.decoder(feat)
        return field1, field2

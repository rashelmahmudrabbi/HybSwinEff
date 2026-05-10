"""
HybSwinEff: Hybrid CNN-Transformer Architecture for Blood Cell Cancer Classification.

This module defines the complete model architecture including:
    - ProjToken         : Linear projection with dropout for token embedding
    - MultiHeadCrossAttention : Cross-attention between CNN and Swin features
    - GatedFusion       : Sigmoid-gated adaptive feature fusion
    - HAFB              : Hierarchical Attention Fusion Block
    - ResidualFusionBlock : Two-layer MLP with residual skip connection
    - ELMClassifier     : Extreme Learning Machine (Stage 2 classifier)
    - MLPHead           : Task-specific MLP classification head
    - HybSwinEffELM     : Full model wrapper integrating all components

Author : Rashel Mahmud Rabbi
Date   : November 2025
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None

from .utils import Config, maybe_load_pretrained_backbone


# ---------------------------------------------------------------------------
# Projection Token
# ---------------------------------------------------------------------------

class ProjToken(nn.Module):
    """
    Projects backbone token features to a shared embedding dimension.

    Args:
        in_dim  : Input feature dimension (e.g. 1024 for CNN, 768 for Swin).
        out_dim : Target embedding dimension (D_f).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor of shape (B, T, in_dim).
        Returns:
            Tensor of shape (B, T, out_dim).
        """
        return self.lin(x)


# ---------------------------------------------------------------------------
# Multi-Head Cross Attention
# ---------------------------------------------------------------------------

class MultiHeadCrossAttention(nn.Module):
    """
    Standard scaled dot-product multi-head cross-attention.

    Query comes from one modality; keys and values from the other,
    enabling bidirectional information exchange between CNN and Swin tokens.

    Args:
        dim       : Feature dimension (must be divisible by num_heads).
        num_heads : Number of attention heads.
        dropout   : Dropout probability on attention weights.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.dh = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            Q         : Query tensor  (B, Nq, D).
            K         : Key tensor    (B, Nk, D).
            V         : Value tensor  (B, Nk, D).
            attn_mask : Optional additive mask (B, num_heads, Nq, Nk).

        Returns:
            out      : Attended output tensor (B, Nq, D).
            attn     : Attention weight map   (B, num_heads, Nq, Nk).
        """
        B, Nq, D = Q.shape
        Nk = K.shape[1]

        q = self.q(Q).view(B, Nq, self.num_heads, self.dh).transpose(1, 2)
        k = self.k(K).view(B, Nk, self.num_heads, self.dh).transpose(1, 2)
        v = self.v(V).view(B, Nk, self.num_heads, self.dh).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, Nq, D)
        return self.o(out), attn


# ---------------------------------------------------------------------------
# Gated Fusion
# ---------------------------------------------------------------------------

class GatedFusion(nn.Module):
    """
    Sigmoid-gated adaptive weighting between attended and original features.

    A compact MLP maps the mean-pooled attended output to a per-channel
    gate in [0, 1].  The gate interpolates between the attention output
    and the original token sequence.

    Args:
        dim         : Feature dimension.
        gate_hidden : Hidden size of the gate MLP.
    """

    def __init__(self, dim: int, gate_hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, dim),
            nn.Sigmoid()
        )

    def forward(self, attn_out: torch.Tensor, orig: torch.Tensor):
        """
        Args:
            attn_out : Cross-attended tokens (B, T, D).
            orig     : Original tokens       (B, T, D).

        Returns:
            fused : Gated combination (B, T, D).
            g     : Gate values       (B, 1, D).
        """
        pooled = attn_out.mean(dim=1)          # (B, D)
        g = self.mlp(pooled).unsqueeze(1)      # (B, 1, D)
        fused = g * attn_out + (1 - g) * orig
        return fused, g


# ---------------------------------------------------------------------------
# Hierarchical Attention Fusion Block (HAFB)
# ---------------------------------------------------------------------------

class HAFB(nn.Module):
    """
    Hierarchical Attention Fusion Block.

    Performs bidirectional cross-attention between multi-scale CNN tokens
    (Fc_multi) and Swin Transformer tokens (Fv_proj), then applies
    independent gated fusion on each side.

    Args:
        D_f         : Shared embedding dimension.
        heads       : Number of cross-attention heads.
        gate_hidden : Hidden size for each GatedFusion MLP.
        dropout     : Dropout on attention weights.
    """

    def __init__(
        self,
        D_f: int = 256,
        heads: int = 8,
        gate_hidden: int = 64,
        dropout: float = 0.0
    ):
        super().__init__()
        self.attn_cv = MultiHeadCrossAttention(D_f, num_heads=heads, dropout=dropout)
        self.attn_vc = MultiHeadCrossAttention(D_f, num_heads=heads, dropout=dropout)
        self.gate_c = GatedFusion(D_f, gate_hidden)
        self.gate_v = GatedFusion(D_f, gate_hidden)
        self.dropout = nn.Dropout(0.2)

    def forward(self, Fc_multi: torch.Tensor, Fv_proj: torch.Tensor):
        """
        Args:
            Fc_multi : Multi-scale CNN tokens  (B, T_c, D_f).
            Fv_proj  : Swin tokens             (B, T_v, D_f).

        Returns:
            Fc_fused     : Fused CNN tokens   (B, T_c, D_f).
            Fv_fused     : Fused Swin tokens  (B, T_v, D_f).
            attn_map_cv  : CNN→Swin attention map.
            attn_map_vc  : Swin→CNN attention map.
            g_c          : CNN gate values.
            g_v          : Swin gate values.
        """
        A_cv, attn_map_cv = self.attn_cv(Fc_multi, Fv_proj, Fv_proj)
        A_vc, attn_map_vc = self.attn_vc(Fv_proj, Fc_multi, Fc_multi)

        Fc_fused, g_c = self.gate_c(A_cv, Fc_multi)
        Fv_fused, g_v = self.gate_v(A_vc, Fv_proj)

        Fc_fused = self.dropout(Fc_fused)
        Fv_fused = self.dropout(Fv_fused)

        return Fc_fused, Fv_fused, attn_map_cv, attn_map_vc, g_c, g_v


# ---------------------------------------------------------------------------
# Residual Fusion Block (RFB)
# ---------------------------------------------------------------------------

class ResidualFusionBlock(nn.Module):
    """
    Two-layer MLP with Layer Normalization, ReLU, Dropout, and a learned
    linear skip connection that projects the input directly to out_dim.

    This block refines the concatenated mean-pooled CNN + Swin features
    while preserving gradient flow through the skip path.

    Args:
        in_dim  : Input dimension  (default: 512 = 2 × 256).
        hidden1 : First hidden dimension.
        out_dim : Output dimension.
        dropout : Dropout rate on the second linear layer.
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden1: int = 256,
        out_dim: int = 256,
        dropout: float = 0.4
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden1)
        self.ln1 = nn.LayerNorm(hidden1)
        self.fc2 = nn.Linear(hidden1, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)
        self.skip = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input tensor (B, in_dim).
        Returns:
            Output tensor (B, out_dim).
        """
        z = F.relu(self.ln1(self.fc1(x)))
        z = self.dropout(F.relu(self.ln2(self.fc2(z))))
        skip = self.skip(x)
        return z + skip


# ---------------------------------------------------------------------------
# ELM Classifier  (Stage 2 — fitted analytically, not via gradient descent)
# ---------------------------------------------------------------------------

class ELMClassifier:
    """
    Extreme Learning Machine classifier used in Stage 2.

    Random weights W_r and biases b_r are fixed at construction.
    The output weight matrix beta is solved analytically via
    Ridge regression: beta = (H^T H + λI)^{-1} H^T Y.

    Args:
        in_dim      : Input feature dimension (must match RFB out_dim).
        hidden_size : Number of random ELM hidden units.
        activation  : 'relu' or 'tanh'.
        ridge_lambda: L2 regularisation coefficient.
        device      : Torch device string.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_size: int = Config.ELM_HIDDEN,
        activation: str = 'relu',
        ridge_lambda: float = Config.RIDGE_LAMBDA,
        device: str = 'cpu'
    ):
        import numpy as np

        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.ridge_lambda = ridge_lambda
        self.device = device

        # Fixed random projections
        self.W_r = torch.randn(in_dim, hidden_size, device=device) * math.sqrt(2.0 / in_dim)
        self.b_r = torch.zeros(hidden_size, device=device)
        self.beta = None

        if activation == 'relu':
            self.act = lambda x: F.relu(x)
        elif activation == 'tanh':
            self.act = lambda x: torch.tanh(x)
        else:
            raise ValueError(f"Unsupported ELM activation: {activation}")

    def fit(self, X, Y_onehot):
        """
        Analytically solve for beta using Ridge regression.

        Args:
            X       : Feature matrix  (N, in_dim)  — numpy array.
            Y_onehot: One-hot targets (N, n_classes) — numpy array.
        """
        import numpy as np

        H = self.act(
            torch.from_numpy(X).to(self.device).float() @ self.W_r + self.b_r
        ).cpu().numpy()

        HtH = H.T @ H
        reg = self.ridge_lambda * np.eye(self.hidden_size)
        beta = np.linalg.solve(HtH + reg, H.T @ Y_onehot)
        self.beta = torch.from_numpy(beta).to(self.device).float()

    def predict_logits(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute class logits for a batch of features.

        Args:
            X : Feature tensor (B, in_dim) on the correct device.
        Returns:
            logits : Tensor (B, n_classes).
        """
        H = self.act(X @ self.W_r + self.b_r)
        return H @ self.beta


# ---------------------------------------------------------------------------
# MLP Classification Head
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    """
    Two-layer MLP classification head: Linear → ReLU → Dropout → Linear.

    Args:
        in_dim      : Input dimension (RFB output, default 256).
        hidden      : Hidden layer size.
        num_classes : Number of output classes.
    """

    def __init__(self, in_dim: int = 256, hidden: int = 128, num_classes: int = None):
        super().__init__()
        assert num_classes is not None, "num_classes must be specified"
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Full Model: HybSwinEffELM
# ---------------------------------------------------------------------------

class HybSwinEffELM(nn.Module):
    """
    HybSwinEff + ELM — Full hybrid CNN-Transformer model for blood cell
    cancer classification.

    Architecture overview
    ---------------------
    Input image (B, 3, 224, 224)
        ├─ EfficientNetV2-S backbone  →  Z_cnn  (B, 1280, 7, 7)
        └─ Swin-Tiny backbone         →  Z_swin (B, 7, 7, 768)

    Projection Block
        ├─ ProjToken(CNN)  →  Fc_proj  (B, 49, D_f)
        └─ ProjToken(Swin) →  Fv_proj  (B, 49, D_f)

    Multi-scale pooling on Fc_proj → Fc_multi (B, T_c, D_f)
        where T_c = 49 (full) + 16 (4×4) + 49 (7×7 re-pool) tokens

    HAFB (bidirectional cross-attention + gated fusion)
        →  Fc_fused, Fv_fused

    Global mean-pooling → μ_c, μ_v  (B, D_f) each
    Concatenation       → F_comb    (B, 2·D_f)

    Residual Fusion Block → F_head  (B, 256)

    ┌─ MLP Head  → logits_mlp  (B, num_classes)   [gradient-trained]
    └─ ELM Head  → logits_elm  (B, num_classes)   [analytically fitted]

    Args:
        num_classes : Number of target classes.
        D_f         : Shared projection dimension (default 256).
        elm_hidden  : Number of ELM hidden units.
        device      : Torch device string.
    """

    def __init__(
        self,
        num_classes: int = None,
        D_f: int = 256,
        elm_hidden: int = Config.ELM_HIDDEN,
        device: str = 'cpu'
    ):
        super().__init__()
        if timm is None:
            raise ImportError(
                "timm is required. Install with: pip install timm"
            )
        assert num_classes is not None, "num_classes must be specified"

        self.device = device
        self.num_classes = num_classes

        # ------------------------------------------------------------------
        # Parallel backbones (pretrained on ImageNet-1K via timm)
        # ------------------------------------------------------------------
        self.cnn = maybe_load_pretrained_backbone(
            'efficientnetv2_s', pretrained=True, device=device
        )
        self.swin = maybe_load_pretrained_backbone(
            'swin_tiny_patch4_window7_224', pretrained=True, device=device
        )

        # Probe output dimensions with a dummy forward pass
        with torch.no_grad():
            dummy = torch.randn(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(device)
            cnn_feat = self.cnn.forward_features(dummy)
            swin_feat = self.swin.forward_features(dummy)
            swin_feat = self._process_swin_features(swin_feat)
            print(f"[HybSwinEff] CNN feature shape  : {cnn_feat.shape}")
            print(f"[HybSwinEff] Swin feature shape : {swin_feat.shape}")

        cnn_out_dim = cnn_feat.shape[1]
        swin_out_dim = swin_feat.shape[-1]

        # ------------------------------------------------------------------
        # Projection, fusion, and classification blocks
        # ------------------------------------------------------------------
        self.proj_c = ProjToken(in_dim=cnn_out_dim, out_dim=D_f).to(device)
        self.proj_v = ProjToken(in_dim=swin_out_dim, out_dim=D_f).to(device)
        self.hafb = HAFB(D_f=D_f, heads=8).to(device)
        self.rfb = ResidualFusionBlock(
            in_dim=2 * D_f, hidden1=256, out_dim=256
        ).to(device)
        self.mlp_head = MLPHead(
            in_dim=256, hidden=128, num_classes=num_classes
        ).to(device)

        # ELM is initialised here but fitted externally via Stage 2
        self.elm = ELMClassifier(
            in_dim=256,
            hidden_size=elm_hidden,
            activation='relu',
            ridge_lambda=Config.RIDGE_LAMBDA,
            device=device
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_swin_features(self, swin_feat: torch.Tensor) -> torch.Tensor:
        """
        Normalise Swin output to (B, T, C) regardless of timm version.

        timm may return NHWC (B, H, W, C) or NCHW (B, C, H, W).
        Both are handled to produce (B, 49, 768) for Swin-Tiny.
        """
        if swin_feat.dim() == 4:
            B, d1, d2, d3 = swin_feat.shape
            # NHWC layout from newer timm (H=7, W=7, C=768)
            if d1 == 7 and d2 == 7:
                swin_feat = swin_feat.permute(0, 3, 1, 2)   # → NCHW
            # Now NCHW → (B, 49, C)
            B, C, H, W = swin_feat.shape
            swin_feat = swin_feat.flatten(2).transpose(1, 2).contiguous()
        elif swin_feat.dim() == 3:
            pass  # already (B, T, C)
        else:
            raise ValueError(
                f"Unexpected Swin output shape: {swin_feat.shape}"
            )
        return swin_feat

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_backbones(self, x: torch.Tensor):
        """Extract raw features from both backbones."""
        cnn_feat = self.cnn.forward_features(x)
        swin_feat = self._process_swin_features(
            self.swin.forward_features(x)
        )
        return cnn_feat, swin_feat

    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False
    ):
        """
        Full forward pass.

        Args:
            x                  : Input image batch (B, 3, H, W).
            return_intermediate: If True, return a dict with F_head and
                                 attention maps in addition to logits.

        Returns:
            logits_mlp (Tensor) if return_intermediate is False.
            dict       (dict)   if return_intermediate is True.
        """
        B = x.shape[0]

        # ── Backbone features ──────────────────────────────────────────
        cnn_feat, Fv = self.forward_backbones(x)    # cnn: (B,C,H,W)  swin: (B,T,D)

        # ── CNN token projection ───────────────────────────────────────
        B, Cc, Hc, Wc = cnn_feat.shape
        Fc_flat = cnn_feat.flatten(2).transpose(1, 2).contiguous()  # (B, 49, Cc)
        Fc_proj = self.proj_c(Fc_flat)                              # (B, 49, D_f)

        # ── Multi-scale pooling on CNN tokens ──────────────────────────
        Fc_reshaped = (
            Fc_proj.transpose(1, 2).contiguous().view(B, -1, Hc, Wc)
        )
        P2 = (
            F.adaptive_avg_pool2d(Fc_reshaped, (7, 7))
            .flatten(2).transpose(1, 2).contiguous()
        )                                                            # (B, 49, D_f)
        P4 = (
            F.adaptive_avg_pool2d(Fc_reshaped, (4, 4))
            .flatten(2).transpose(1, 2).contiguous()
        )                                                            # (B, 16, D_f)
        Fc_multi = torch.cat([Fc_proj, P2, P4], dim=1)             # (B, T_c, D_f)

        # ── Swin token projection ──────────────────────────────────────
        Fv_proj = self.proj_v(Fv)                                   # (B, T_v, D_f)

        # ── HAFB bidirectional cross-attention ─────────────────────────
        Fc_fused, Fv_fused, attn_cv, attn_vc, g_c, g_v = self.hafb(
            Fc_multi, Fv_proj
        )

        # ── Global mean pooling & concatenation ────────────────────────
        mu_c = Fc_fused.mean(dim=1)                                 # (B, D_f)
        mu_v = Fv_fused.mean(dim=1)                                 # (B, D_f)
        F_comb = torch.cat([mu_c, mu_v], dim=1)                    # (B, 2·D_f)

        # ── Residual Fusion Block ──────────────────────────────────────
        F_head = self.rfb(F_comb)                                   # (B, 256)

        # ── MLP classification head ────────────────────────────────────
        logits_mlp = self.mlp_head(F_head)                          # (B, num_classes)

        if return_intermediate:
            return {
                'logits_mlp': logits_mlp,
                'F_head': F_head,
                'attn_map_cv': attn_cv,
                'attn_map_vc': attn_vc,
                'g_c': g_c,
                'g_v': g_v,
            }
        return logits_mlp
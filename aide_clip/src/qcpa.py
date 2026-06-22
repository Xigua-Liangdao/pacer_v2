# qcpa.py
"""
Query-Conditional Prompt Attention (QCPA) — full CAPC replacement.

This module replaces the entire CAPC head, not just the prompt weighting.
The original CAPC had three steps:
    S1: alpha = softmax(prompt_weights)            # static, class-specific
    S2: scale by exp(tau_c)                         # per-class temperature
    S3: add b_c                                     # per-class bias

This module replaces them with:
    S1': QCPA — attention(query=h, key/val=T) with scope = local or global
    S2': Mahalanobis-style learned distance scale per class (replaces simple temp)
    S3': low-rank class bias generator (replaces flat b_c) — optional

The redesigned S2/S3 are safer because they reduce to the original S2/S3
when their extra capacity is pruned (zero init for low-rank parts).

ABLATION FLAGS (each independently switchable):
    attention_scope: 'local' | 'global'
        local  — within each class, P prompts only (Q1's Option 1)
        global — across all C*P prompts, learn class assignments (Q1's Option 3
                 first stage; second stage handled by `dual_stage`)
    dual_stage:  if True and scope='global', adds a second class-level
                 attention. Implements Q1's Option 3 fully.
    use_residual_gate: per-class gate to blend dynamic vs static prototype
    use_mahalanobis_temp: replaces S2 with diagonal Mahalanobis scaling
    use_lowrank_bias:    replaces S3 with low-rank bias generator
"""
from __future__ import annotations
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Core attention block
# ----------------------------------------------------------------------------
class _MultiHeadCrossAttn(nn.Module):
    """Generic MHA used internally; not exposed."""
    def __init__(self, dim, num_heads, attn_dropout, proj_dropout):
        super().__init__()
        assert dim % num_heads == 0
        self.h, self.d = num_heads, dim // num_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.adrop = nn.Dropout(attn_dropout)
        self.pdrop = nn.Dropout(proj_dropout)
        # near-uniform start
        nn.init.xavier_uniform_(self.q.weight, gain=0.1)
        nn.init.xavier_uniform_(self.k.weight, gain=0.1)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.xavier_uniform_(self.o.weight)

    def forward(self, q, k, v, return_attn=False):
        """
        q: [..., Nq, D]
        k: [..., Nk, D]
        v: [..., Nk, D]
        returns: [..., Nq, D]
        """
        Nq = q.shape[-2]
        Nk = k.shape[-2]
        D = q.shape[-1]
        H, Hd = self.h, self.d

        q = self.q(q).reshape(*q.shape[:-1], H, Hd).transpose(-2, -3)  # [..., H, Nq, Hd]
        k = self.k(k).reshape(*k.shape[:-1], H, Hd).transpose(-2, -3)  # [..., H, Nk, Hd]
        v = self.v(v).reshape(*v.shape[:-1], H, Hd).transpose(-2, -3)

        attn_logits = (q @ k.transpose(-2, -1)) / (Hd ** 0.5)          # [..., H, Nq, Nk]
        attn = F.softmax(attn_logits, dim=-1)
        attn_d = self.adrop(attn)
        out = attn_d @ v                                                # [..., H, Nq, Hd]
        out = out.transpose(-2, -3).reshape(*out.shape[:-3], Nq, D)     # [..., Nq, D]
        out = self.pdrop(self.o(out))
        if return_attn:
            return out, attn
        return out


# ----------------------------------------------------------------------------
# QCPA head
# ----------------------------------------------------------------------------
class QCPAHead(nn.Module):
    def __init__(
        self,
        feat_dim: int = 512,
        num_classes: int = 5,
        num_prompts: int = 9,            # using structured prompts now
        num_heads: int = 4,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        # --- core knobs ---
        attention_scope: Literal['local', 'global'] = 'local',
        dual_stage: bool = False,
        # --- calibration knobs ---
        use_residual_gate: bool = True,
        use_mahalanobis_temp: bool = True,
        use_lowrank_bias: bool = True,
        bias_rank: int = 16,
        # --- global scale (kept from original CAPC) ---
        global_logit_scale_init: float = 4.6052,  # log(100), matches CLIP default
    ):
        super().__init__()
        self.C = num_classes
        self.P = num_prompts
        self.D = feat_dim
        self.scope = attention_scope
        self.dual_stage = dual_stage
        self.use_residual_gate = use_residual_gate
        self.use_mahalanobis = use_mahalanobis_temp
        self.use_lowrank_bias = use_lowrank_bias

        # Stage 1: prompt-level attention
        self.attn_prompt = _MultiHeadCrossAttn(
            feat_dim, num_heads, attn_dropout, proj_dropout
        )

        # Stage 2: class-level attention (only if dual_stage=True)
        if dual_stage:
            assert attention_scope == 'global', \
                "dual_stage only meaningful with global scope"
            self.attn_class = _MultiHeadCrossAttn(
                feat_dim, num_heads, attn_dropout, proj_dropout
            )

        # Residual gate per class
        if use_residual_gate:
            self.gate_logit = nn.Parameter(torch.zeros(num_classes))

        # Redesigned S2: Mahalanobis-style diagonal scale per class
        # Replaces simple exp(tau_c). Gives the model per-dimension control.
        if use_mahalanobis_temp:
            # log-parameterized; init at 0 -> diag = 1 (identity, equiv to plain cosine)
            self.log_diag = nn.Parameter(torch.zeros(num_classes, feat_dim))
            # also keep a scalar per-class temp for backward compatibility
            self.log_tau = nn.Parameter(torch.zeros(num_classes))
        else:
            # fall back to original S2
            self.log_tau = nn.Parameter(torch.zeros(num_classes))

        # Redesigned S3: low-rank bias generator
        # b_c = U_c @ V @ h  (h-conditional bias) + b0_c (static)
        # Replaces flat b_c. With U=0 init, reduces to original S3.
        if use_lowrank_bias:
            self.bias_U = nn.Parameter(torch.zeros(num_classes, bias_rank))
            self.bias_V = nn.Parameter(torch.randn(bias_rank, feat_dim) * 0.02)
            self.bias_static = nn.Parameter(torch.zeros(num_classes))
        else:
            self.bias_static = nn.Parameter(torch.zeros(num_classes))

        # Global scale (kappa in original CAPC)
        self.log_scale = nn.Parameter(torch.tensor(global_logit_scale_init))

    # ------------------------------------------------------------------
    # Prototype computation
    # ------------------------------------------------------------------
    def _local_prototype(self, h, T, return_attn=False):
        """
        Within-class attention: each class attends to its own P prompts.
        h: [B, D], T: [C, P, D] -> [B, C, D]
        """
        B, D = h.shape
        C, P, _ = T.shape
        # broadcast h across classes
        q = h.unsqueeze(1).unsqueeze(2).expand(B, C, 1, D)   # [B, C, 1, D]
        k = T.unsqueeze(0).expand(B, C, P, D)                # [B, C, P, D]
        v = k
        if return_attn:
            out, attn = self.attn_prompt(q, k, v, return_attn=True)  # out:[B,C,1,D] attn:[B,C,H,1,P]
            return out.squeeze(2), attn.squeeze(-2)                  # [B,C,D], [B,C,H,P]
        out = self.attn_prompt(q, k, v)
        return out.squeeze(2), None

    def _global_prototype(self, h, T, return_attn=False):
        """
        Cross-class attention: query attends to all C*P prompts at once,
        then we group results back into C classes by averaging within group.
        h: [B, D], T: [C, P, D] -> [B, C, D]
        """
        B, D = h.shape
        C, P, _ = T.shape
        T_flat = T.reshape(C * P, D)                          # [C*P, D]
        q = h.unsqueeze(1)                                    # [B, 1, D]
        k = T_flat.unsqueeze(0).expand(B, C * P, D)           # [B, C*P, D]
        v = k
        if return_attn:
            out, attn = self.attn_prompt(q, k, v, return_attn=True)  # out:[B,1,D] attn:[B,H,1,C*P]
            attn = attn.squeeze(-2).reshape(B, -1, C, P)             # [B, H, C, P]
        else:
            out = self.attn_prompt(q, k, v)
            attn = None
        # out is one global vector — broadcast to each class slot
        proto_global = out.squeeze(1).unsqueeze(1).expand(B, C, D)   # [B, C, D]

        if self.dual_stage:
            # Stage 2: per-class attention over the C "static" class anchors
            T_class_anchor = T.mean(dim=1)                       # [C, D]
            q2 = proto_global                                    # [B, C, D]
            k2 = T_class_anchor.unsqueeze(0).expand(B, C, D)     # [B, C, D]
            v2 = k2
            # treat as one Nq=C, Nk=C attention (no broadcast over an extra dim)
            proto = self.attn_class(q2, k2, v2)                  # [B, C, D]
            return proto, attn
        return proto_global, attn

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, h, T, return_aux=False):
        """
        h: [B, D]    L2-normalized adapted visual feature
        T: [C, P, D] L2-normalized frozen text bank
        returns: logits [B, C]
        """
        B, D = h.shape
        C, P, _ = T.shape

        # --- Prototype: dynamic ---
        if self.scope == 'local':
            proto_dyn, attn = self._local_prototype(h, T, return_attn=return_aux)
        else:
            proto_dyn, attn = self._global_prototype(h, T, return_attn=return_aux)
        proto_dyn = F.normalize(proto_dyn, dim=-1)               # [B, C, D]

        # --- Prototype: static fallback (mean over P) ---
        proto_static = F.normalize(T.mean(dim=1), dim=-1)        # [C, D]
        proto_static = proto_static.unsqueeze(0).expand(B, C, D)

        # --- Residual gate blend ---
        if self.use_residual_gate:
            g = torch.sigmoid(self.gate_logit).view(1, C, 1)     # [1, C, 1]
            proto = g * proto_dyn + (1.0 - g) * proto_static
            proto = F.normalize(proto, dim=-1)
        else:
            proto = proto_dyn

        # --- Similarity (S1' done) ---
        h_n = F.normalize(h, dim=-1).unsqueeze(1)                # [B, 1, D]

        if self.use_mahalanobis:
            # diag-Mahalanobis: sim_c = sum_d diag_c[d] * h[d] * proto_c[d]
            diag = self.log_diag.exp().unsqueeze(0)              # [1, C, D]
            sims = (h_n * proto * diag).sum(dim=-1)              # [B, C]
            sims = sims * self.log_tau.exp().unsqueeze(0)        # [1, C] scalar temp
        else:
            sims = (h_n * proto).sum(dim=-1)                     # [B, C]
            sims = sims * self.log_tau.exp().unsqueeze(0)

        # --- Bias (S3') ---
        if self.use_lowrank_bias:
            # b_c(h) = U_c · (V · h)   shape: [B, C]
            Vh = h @ self.bias_V.t()                             # [B, R]
            b_dyn = Vh @ self.bias_U.t()                         # [B, C]
            bias = b_dyn + self.bias_static.unsqueeze(0)         # [B, C]
        else:
            bias = self.bias_static.unsqueeze(0).expand(B, C)

        logits = self.log_scale.exp() * sims + bias              # [B, C]

        if return_aux:
            aux = {
                "attn": attn.detach() if attn is not None else None,
                "gate": torch.sigmoid(self.gate_logit).detach() if self.use_residual_gate else None,
                "log_tau": self.log_tau.detach(),
                "log_diag_norm": self.log_diag.detach().norm(dim=-1) if self.use_mahalanobis else None,
                "log_scale": self.log_scale.detach().item(),
            }
            return logits, aux
        return logits


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, P, D = 4, 5, 9, 512
    h = F.normalize(torch.randn(B, D), dim=-1)
    T = F.normalize(torch.randn(C, P, D), dim=-1)

    print("=== local scope ===")
    head = QCPAHead(D, C, P, attention_scope='local')
    logits, aux = head(h, T, return_aux=True)
    assert logits.shape == (B, C)
    print(f"  logits: {logits.shape}, attn: {aux['attn'].shape}")
    logits.sum().backward()
    print("  backward OK")

    print("=== global scope, single stage ===")
    head = QCPAHead(D, C, P, attention_scope='global', dual_stage=False)
    logits, aux = head(h, T, return_aux=True)
    assert logits.shape == (B, C)
    print(f"  logits: {logits.shape}, attn: {aux['attn'].shape}")
    logits.sum().backward()
    print("  backward OK")

    print("=== global scope, dual stage ===")
    head = QCPAHead(D, C, P, attention_scope='global', dual_stage=True)
    logits, aux = head(h, T, return_aux=True)
    assert logits.shape == (B, C)
    print(f"  logits: {logits.shape}")
    logits.sum().backward()
    print("  backward OK")

    print("=== param count ===")
    head = QCPAHead(D, C, P)
    n = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"  trainable params: {n:,}")
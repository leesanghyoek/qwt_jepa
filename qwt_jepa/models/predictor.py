"""JEPA predictor (Phase D - Buoc D2).

Transformer hep hon encoder (d_pred < d_model, depth nho). Dau vao la embedding
context da chieu ve d_pred cong positional embed cua tung token, noi them mask
token dat tai vi tri target. Dau ra `z_pred` tai cac vi tri target, chieu nguoc
ve d_model de so voi z_target.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import TransformerBlock


class Predictor(nn.Module):
    def __init__(self, cfg: dict, n_tokens: int):
        super().__init__()
        d_model = int(cfg["model"]["d_model"])
        pcfg = cfg["model"]["predictor"]
        d_pred = int(pcfg["d_pred"])
        depth = int(pcfg["depth"])
        heads = int(pcfg["heads"])

        self.in_proj = nn.Linear(d_model, d_pred)
        self.out_proj = nn.Linear(d_pred, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_pred))
        self.pos_embed = nn.Parameter(torch.zeros(n_tokens, d_pred))
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_pred, heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_pred)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        z_ctx: torch.Tensor,        # [B, n_ctx, d_model]
        ctx_index: torch.Tensor,    # LongTensor [n_ctx]
        tgt_index: torch.Tensor,    # LongTensor [n_tgt]
    ) -> torch.Tensor:
        b = z_ctx.shape[0]
        n_tgt = tgt_index.numel()

        h_ctx = self.in_proj(z_ctx) + self.pos_embed[ctx_index].unsqueeze(0)
        h_tgt = self.mask_token.expand(b, n_tgt, -1) + self.pos_embed[tgt_index].unsqueeze(0)

        h = torch.cat([h_ctx, h_tgt], dim=1)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)

        z_pred = self.out_proj(h[:, h_ctx.shape[1]:])   # [B, n_tgt, d_model]
        return z_pred

"""Standard autoregressive language model baseline with matched parameter count."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .transformer import Transformer


class AutoregressiveLM(nn.Module):
    """GPT-style language model sharing the nanoEBM backbone."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # Use untied output head so the parameter budget mirrors the EBM energy head
        self.transformer = Transformer(config, tie_weights=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **_: object,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, dict]:
        logits, loss = self.transformer(idx, targets=targets)
        metrics: dict = {}
        if loss is not None:
            metrics["perplexity"] = torch.exp(loss.detach()).item()
            metrics["nll_loss"] = loss.detach().item()
        return loss, logits, metrics

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self.transformer(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

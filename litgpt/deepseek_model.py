# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""DeepSeek-specific model components that extend the base LitGPT model."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from litgpt.deepseek_config import DeepseekConfig
from litgpt.model import Block, LLaMAMLP, GPT


class DeepseekBlock(Block):
    """Block that handles DeepSeek-specific MLP initialization."""

    def __init__(self, config: DeepseekConfig, block_idx: int) -> None:
        # Call parent init first but override mlp afterwards
        super().__init__(config, block_idx)

        # every moe_layer_freq layers is a MoE layer
        is_moe_layer = (
            config.mlp_class_name == "DeepseekMoE"
            and block_idx >= config.first_k_dense_replace
            and block_idx % config.moe_layer_freq == 0
        )
        if config.mlp_class_name == "DeepseekMoE" and not is_moe_layer:
            self.mlp = LLaMAMLP(config)


class DeepseekGPT(GPT):
    """GPT model with DeepSeek-specific Block implementation."""

    def __init__(self, config: DeepseekConfig) -> None:
        # Call parent init first
        super().__init__(config)

        # Replace blocks with DeepseekBlock
        self.transformer.h = nn.ModuleList(
            DeepseekBlock(config, block_idx) for block_idx in range(config.n_layer)
        )


class DeepseekMoEGate(nn.Module):
    """Gate module for DeepSeek MoE."""

    def __init__(self, config: DeepseekConfig) -> None:
        super().__init__()
        self.config = config
        self.top_k = config.n_expert_per_token
        self.n_routed_experts = config.n_expert
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.n_embd)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Loss-free balancing parameters
        self.loss_free_balance = config.loss_free_balance
        if self.loss_free_balance:
            self.loss_free_balance_update_rate = config.loss_free_balance_update_rate
            self.register_buffer("expert_bias", torch.zeros(self.n_routed_experts))

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)

        if self.scoring_func != "softmax":
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )
        scores = logits.softmax(dim=-1, dtype=torch.float)

        # Apply loss-free balancing if enabled
        if self.loss_free_balance:
            # Apply expert bias to logits before topk selection
            biased_logits = logits + self.expert_bias
            _, topk_idx = torch.topk(biased_logits, k=self.top_k, dim=-1, sorted=False)
            # Use original scores (without bias) for weights
            topk_weight = torch.gather(scores, -1, topk_idx)

            # Update expert bias during training
            if self.training:
                with torch.no_grad():
                    # Count tokens per expert
                    tokens_per_expert = torch.bincount(
                        topk_idx.flatten(), minlength=self.n_routed_experts
                    )
                    # Calculate load error: average_load - actual_load
                    # Positive error means expert is underloaded, should increase bias
                    average_load = tokens_per_expert.float().mean()
                    load_error = average_load - tokens_per_expert.float()
                    # Update bias
                    self.expert_bias += (
                        load_error.sign() * self.loss_free_balance_update_rate
                    )

            # No auxiliary loss when using loss-free balancing
            aux_loss = None
        else:
            # Standard top-k selection without bias
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False
            )

            # Calculate auxiliary loss if enabled
            if self.training and self.alpha > 0.0:
                aux_loss = self._compute_aux_loss(
                    scores, topk_idx, bsz, seq_len, hidden_states.device
                )
            else:
                aux_loss = None

        # Normalize top-k weights if needed
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        return topk_idx, topk_weight, aux_loss

    def _compute_aux_loss(
        self,
        scores: torch.Tensor,
        topk_idx: torch.Tensor,
        bsz: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute auxiliary loss for load balancing."""
        topk_idx_for_aux_loss = topk_idx.view(bsz, -1)

        if self.seq_aux:
            scores_for_seq_aux = scores.view(bsz, seq_len, -1)
            ce = torch.zeros(bsz, self.n_routed_experts, device=device)
            ce.scatter_add_(
                1,
                topk_idx_for_aux_loss,
                torch.ones(bsz, seq_len * self.top_k, device=device),
            ).div_(seq_len * self.top_k / self.n_routed_experts)
            aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                dim=1
            ).mean() * self.alpha
        else:
            mask_ce = F.one_hot(
                topk_idx_for_aux_loss.view(-1),
                num_classes=self.n_routed_experts,
            )
            ce = mask_ce.float().mean(0)
            pi = scores.mean(0)
            fi = ce * self.n_routed_experts
            aux_loss = (pi * fi).sum() * self.alpha

        return aux_loss


class AddDeepseekAuxiliaryLoss(torch.autograd.Function):
    """Custom autograd function to add auxiliary loss to the computation graph."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class DeepseekMoE(nn.Module):
    """DeepSeek Mixture of Experts module."""

    def __init__(self, config: DeepseekConfig) -> None:
        super().__init__()
        self.config = config
        self.n_expert_per_token = config.n_expert_per_token
        self.experts = nn.ModuleList(
            LLaMAMLP(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.n_expert)
        )
        self.gate = DeepseekMoEGate(config)
        if config.n_shared_experts is not None and config.n_shared_experts > 0:
            shared_intermediate_size = (
                config.moe_intermediate_size * config.n_shared_experts
            )
            self.shared_experts = LLaMAMLP(
                config, intermediate_size=shared_intermediate_size
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        orig_shape = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            x = x.repeat_interleave(self.n_expert_per_token, dim=0)
            y = torch.empty_like(x)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
            if aux_loss is not None:
                y = AddDeepseekAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(
                *orig_shape
            )

        if hasattr(self, "shared_experts"):
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(
        self,
        x: torch.Tensor,
        flat_expert_indices: torch.Tensor,
        flat_expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.n_expert_per_token
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(
                0,
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce="sum",
            )
        return expert_cache

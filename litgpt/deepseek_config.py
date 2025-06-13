# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from dataclasses import dataclass
from typing import Literal

from litgpt.config import Config


@dataclass
class DeepseekConfig(Config):
    """Configuration for DeepSeek models with MoE-specific parameters."""

    # DeepseekMoE specific
    n_shared_experts: int = 0

    # MoE Gate parameters
    norm_topk_prob: bool = False
    scoring_func: str = "softmax"
    aux_loss_alpha: float = 0.001
    seq_aux: bool = False
    loss_free_balance: bool = False
    loss_free_balance_update_rate: float = 1e-3

    # Override mlp_class_name type to include DeepseekMoE
    mlp_class_name: Literal[
        "GptNeoxMLP", "LLaMAMLP", "GemmaMLP", "LLaMAMoE", "DeepseekMoE"
    ] = "GptNeoxMLP"


#################
# Deepseek MoE
#################
deepseek_configs = [
    # https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/main/config.json
    dict(
        name="Deepseek-MoE-16B",
        hf_config=dict(org="deepseek-ai", name="deepseek-moe-16b-base"),
        vocab_size=102400,
        padded_vocab_size=102400,
        n_layer=28,
        n_embd=2048,
        n_head=16,
        n_query_groups=16,
        parallel_residual=True,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-6,
        mlp_class_name="DeepseekMoE",
        intermediate_size=5632,  # 10944 in HF is a mistake
        # MoE
        n_expert=64,
        n_expert_per_token=6,
        # DeepseekMoE
        n_shared_experts=2,
        moe_intermediate_size=1408,
        # MoE Gate
        norm_topk_prob=True,
        rope_base=10000,
        rotary_percentage=1.0,
    ),
    dict(
        name="Deepseek-MoE-3B-lfb",
        hf_config=dict(org="local", name="Deepseek-MoE-3B-lfb"),
        vocab_size=102400,
        padded_vocab_size=102400,
        n_layer=14,
        n_embd=2048,
        n_head=16,
        n_query_groups=16,
        parallel_residual=True,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-6,
        mlp_class_name="DeepseekMoE",
        intermediate_size=5632,
        # MoE
        n_expert=32,
        n_expert_per_token=4,
        # DeepseekMoE
        n_shared_experts=1,
        moe_intermediate_size=1408,
        # MoE Gate
        norm_topk_prob=True,
        rope_base=10000,
        rotary_percentage=1.0,
        # Loss-free balancing
        loss_free_balance=True,
        aux_loss_alpha=0.0,
        loss_free_balance_update_rate=1e-3,
    ),
    dict(
        name="Deepseek-MoE-1B-lfb",
        hf_config=dict(org="local", name="Deepseek-MoE-1B-lfb"),
        vocab_size=102400,
        padded_vocab_size=102400,
        n_layer=8,
        n_embd=1024,
        n_head=8,
        n_query_groups=8,
        parallel_residual=True,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-6,
        mlp_class_name="DeepseekMoE",
        intermediate_size=2816,
        # MoE
        n_expert=16,
        n_expert_per_token=2,
        # DeepseekMoE
        n_shared_experts=1,
        moe_intermediate_size=704,
        # MoE Gate
        norm_topk_prob=True,
        rope_base=10000,
        rotary_percentage=1.0,
        # Loss-free balancing
        loss_free_balance=True,
        aux_loss_alpha=0.0,
        loss_free_balance_update_rate=1e-3,
    ),
]

# Dictionary for easy name-based lookup
deepseek_name_to_config = {config["name"]: config for config in deepseek_configs}

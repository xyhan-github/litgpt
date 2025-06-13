# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from dataclasses import dataclass
from typing import Literal, Type, Optional, Any

from litgpt.config import Config


@dataclass
class DeepseekConfig(Config):
    """Configuration for DeepSeek models with MoE-specific parameters."""

    # DeepseekMoE specific
    n_shared_experts: int = 0
    first_k_dense_replace: int = 0
    moe_layer_freq: int = 1

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

    @property
    def mlp_class(self) -> Type:
        """Override to handle DeepseekMoE which is in a different module."""
        if self.mlp_class_name == "DeepseekMoE":
            from litgpt.deepseek_model import DeepseekMoE

            return DeepseekMoE
        else:
            # Fall back to parent class implementation for other MLP types
            return super().mlp_class

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Optional["DeepseekConfig"]:
        """Create a DeepseekConfig from a model name."""
        if name in deepseek_name_to_config:
            conf_dict = deepseek_name_to_config[name].copy()
            conf_dict.update(kwargs)
            return cls(**conf_dict)
        return None


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
        intermediate_size=5632,
        # MoE
        n_expert=64,
        n_expert_per_token=6,
        # DeepseekMoE
        n_shared_experts=2,
        moe_intermediate_size=1408,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        # MoE Gate
        norm_topk_prob=True,
        scoring_func="softmax",
        rope_base=10000,
        rotary_percentage=1.0,
        block_size=4096,
    ),
    # Based on Table 5 of the Auxiliary-Loss-Free Load Balancing Strategy paper
    dict(
        name="Deepseek-MoE-3B-lfb",
        hf_config=dict(org="local", name="Deepseek-MoE-3B-lfb"),
        vocab_size=32064,
        padded_vocab_size=32064,
        n_layer=11,
        n_embd=1280,
        n_head=10,
        n_query_groups=10,
        parallel_residual=True,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-6,
        mlp_class_name="DeepseekMoE",
        intermediate_size=3520,
        # MoE
        n_expert=64,
        n_expert_per_token=6,
        # DeepseekMoE
        n_shared_experts=2,
        moe_intermediate_size=880,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        # MoE Gate
        norm_topk_prob=True,
        scoring_func="softmax",
        rope_base=10000,
        rotary_percentage=1.0,
        # Loss-free balancing (when enabled, aux_loss_alpha should be 0)
        loss_free_balance=True,
        aux_loss_alpha=0.0,
        loss_free_balance_update_rate=1e-3,
        block_size=4096,
    ),
    # Based on Table 5 of the Auxiliary-Loss-Free Load Balancing Strategy paper
    dict(
        name="Deepseek-MoE-1B-lfb",
        hf_config=dict(org="local", name="Deepseek-MoE-1B-lfb"),
        vocab_size=32064,
        padded_vocab_size=32064,
        n_layer=9,
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
        n_expert=64,
        n_expert_per_token=6,
        # DeepseekMoE
        n_shared_experts=2,
        moe_intermediate_size=528,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        # MoE Gate
        norm_topk_prob=True,
        scoring_func="softmax",
        rope_base=10000,
        rotary_percentage=1.0,
        # Loss-free balancing (when enabled, aux_loss_alpha should be 0)
        loss_free_balance=True,
        aux_loss_alpha=0.0,
        loss_free_balance_update_rate=1e-3,
        block_size=4096,
    ),
]

# Dictionary for easy name-based lookup
deepseek_name_to_config = {config["name"]: config for config in deepseek_configs}


def create_deepseek_model(name: str, **kwargs: Any):
    """Factory function to create a DeepSeek model."""
    from litgpt.deepseek_model import DeepseekGPT

    config = DeepseekConfig.from_name(name, **kwargs)
    if config is None:
        raise ValueError(f"{name!r} is not a valid DeepSeek model name")

    return DeepseekGPT(config)


# Export configs so they can be imported and integrated
__all__ = [
    "DeepseekConfig",
    "deepseek_configs",
    "deepseek_name_to_config",
    "create_deepseek_model",
]

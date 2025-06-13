# DeepSeek Models in LitGPT

This implementation adds support for DeepSeek MoE models with auxiliary-loss-free load balancing.

## Features

- **DeepSeek MoE architecture**: Mixture of Experts with shared experts
- **Auxiliary-loss-free load balancing**: Novel load balancing method that avoids gradient interference
- **Support for 3 model sizes**: 16B, 3B, and 1B parameters

## Available Models

1. `Deepseek-MoE-16B` - Original DeepSeek MoE 16B model
2. `Deepseek-MoE-3B-lfb` - 3B model with loss-free balancing
3. `Deepseek-MoE-1B-lfb` - 1B model with loss-free balancing

## Usage

Since DeepSeek models require special handling, use the dedicated factory function:

```python
from litgpt.deepseek_config import DeepseekConfig, create_deepseek_model

# Create a DeepSeek model
model = create_deepseek_model("Deepseek-MoE-1B-lfb")

# Or create a config first
config = DeepseekConfig.from_name("Deepseek-MoE-3B-lfb")
model = create_deepseek_model(config.name)
```

## Key Innovations

### 1. First Layer Dense MLP
The first transformer block uses a regular dense MLP instead of MoE, as specified in the DeepSeek architecture.

### 2. Auxiliary-Loss-Free Load Balancing
When `loss_free_balance=True`:
- Expert selection uses biased logits: `logits + expert_bias`
- Expert weights use original scores (without bias)
- Bias is updated based on load imbalance: `bias += sign(average_load - expert_load) * update_rate`
- No auxiliary loss is added to the computation graph

### 3. Shared Experts
Models include shared experts that process all tokens in addition to the routed experts.

## Configuration Parameters

Key parameters for DeepSeek models:

- `n_shared_experts`: Number of shared experts (default: 2)
- `loss_free_balance`: Enable loss-free load balancing (default: False)
- `loss_free_balance_update_rate`: Bias update rate (default: 1e-3)
- `aux_loss_alpha`: Auxiliary loss weight (set to 0 when using loss-free balancing)
- `norm_topk_prob`: Normalize top-k probabilities (default: True)

## Testing

Run the test script to verify the implementation:

```bash
python test_deepseek.py
```

## References

- [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066)
- [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) 
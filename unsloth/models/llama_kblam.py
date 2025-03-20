# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Union
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

from .kblam import KBLaMConfig

PADDING_VALUE = torch.finfo(torch.bfloat16).min


class KBLamLlamaAttention(nn.Module):
    """
    Llama attention mechanism adapted for KBLaM.
    Optimized for Unsloth's memory-efficient processing.
    """
    
    def __init__(self, config, layer_idx: Optional[int] = None):
        """
        Initialize KBLaM-adapted Llama attention.
        
        Args:
            config: Model configuration
            layer_idx: Layer index
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        # KBLaM-specific parameters
        self.score_shift = nn.Parameter(torch.zeros(self.num_heads, 1) - 3)
        
        # Projection matrices
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        
        # Additional query projection for KB queries if using separate query head
        self.q_proj_kb = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=config.attention_bias
        )
        
        # Initialize rotary embeddings
        self._init_rope()
    
    def _init_rope(self):
        """Initialize rotary position embeddings."""
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    
    def prune_key_value(self, query, kb_keys, kb_values, topk_size=20):
        """
        Prune KB key-value pairs based on attention scores to keep only top-k entries.
        
        Args:
            query: Query tensor
            kb_keys: KB keys tensor
            kb_values: KB values tensor
            topk_size: Number of top entries to keep
            
        Returns:
            Pruned keys, pruned values, and attention weights
        """
        batch_size, num_heads, kb_len, head_dim = kb_keys.shape
        
        # Compute attention scores
        attn_weights = torch.matmul(query, kb_keys.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )  # Batchsize, num_heads, query_size, key_size
        
        # If topk_size is larger than KB length, no need to prune
        if topk_size >= kb_len:
            return kb_keys, kb_values, attn_weights
        
        # Get top-k indices
        with torch.no_grad():
            top_idx = attn_weights.sum((1, 2)).topk(min(kb_len, topk_size), -1)[1]
            # Reshape for gather operation
            top_idx = top_idx.view(batch_size, -1, topk_size, 1).expand(
                batch_size, num_heads, topk_size, head_dim
            )
            # Gather top-k entries
            kb_keys = kb_keys.gather(-2, top_idx)
            kb_values = kb_values.gather(-2, top_idx)
        
        return kb_keys, kb_values, attn_weights[..., :topk_size]
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        kb_kvs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kb_config: Optional[KBLaMConfig] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for KBLaM-adapted Llama attention.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_value: Past key-value cache
            output_attentions: Whether to output attention weights
            use_cache: Whether to use key-value cache
            cache_position: Position in cache
            kb_kvs: Knowledge base key-value pairs
            kb_config: KBLaM configuration
            
        Returns:
            output hidden states, attention weights (optional), past key-value (optional)
        """
        bsz, q_len, _ = hidden_states.size()
        
        # Process input projections - with Unsloth optimizations
        # (Note: actual implementation would need to handle tensor parallelism)
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Repeat kv states if needed
        if self.num_key_value_groups > 1:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Apply rotary position embeddings
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        
        # Apply rotary embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Process past key-value cache if provided
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        # Calculate query-key attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Process KB if provided and if this is a KB-enabled layer
        kb_attn_weights = None
        if kb_kvs is not None and kb_config is not None:
            # Only process KB in specified layers
            if (self.layer_idx is not None and 
                self.layer_idx % kb_config.kb_layer_frequency == 0):
                
                # Get KB keys and values
                kb_keys, kb_values = kb_kvs
                kb_layer_idx = self.layer_idx // kb_config.kb_layer_frequency
                
                # Slice the KB embeddings for this layer
                kb_keys_slice = kb_keys[:, :, kb_layer_idx*self.head_dim:(kb_layer_idx+1)*self.head_dim]
                kb_values_slice = kb_values[:, :, kb_layer_idx*self.head_dim:(kb_layer_idx+1)*self.head_dim]
                
                # Reshape for attention
                kb_keys_slice = kb_keys_slice.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                kb_values_slice = kb_values_slice.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                
                # Use separate query head for KB if configured
                if kb_config.sep_query_head:
                    kb_query_states = self.q_proj_kb(hidden_states).view(
                        bsz, q_len, self.num_heads, self.head_dim
                    ).transpose(1, 2)
                    kb_query_states, _ = apply_rotary_pos_emb(kb_query_states, None, cos, sin, position_ids)
                else:
                    kb_query_states = query_states
                
                # Prune KB entries for efficiency if dynamic sparsification is enabled
                if kb_config.dynamic_sparsify:
                    kb_keys_slice, kb_values_slice, kb_attn_weights = self.prune_key_value(
                        kb_query_states, kb_keys_slice, kb_values_slice, kb_config.top_k_kb
                    )
                
                # Calculate KB attention scores
                if kb_attn_weights is None:
                    kb_attn_weights = torch.matmul(
                        kb_query_states, kb_keys_slice.transpose(2, 3)
                    ) / math.sqrt(self.head_dim)
                
                # Apply score shift to KB attention
                kb_attn_weights = kb_attn_weights + self.score_shift.unsqueeze(-1)
                
                # Create combined attention mask
                combined_attn_weights = torch.cat([attn_weights, kb_attn_weights], dim=-1)
                
                # Create combined key/value states
                key_states_combined = torch.cat([key_states, kb_keys_slice], dim=2)
                value_states_combined = torch.cat([value_states, kb_values_slice], dim=2)
                
                # Update variables for further processing
                attn_weights = combined_attn_weights
                key_states = key_states_combined
                value_states = value_states_combined
                
                # Update attention mask if provided
                if attention_mask is not None:
                    # Create mask for KB entries (all 0s to always attend)
                    kb_attention_mask = torch.zeros(
                        (bsz, 1, q_len, kb_keys_slice.size(2)),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    )
                    # Combine with original attention mask
                    attention_mask = torch.cat([attention_mask, kb_attention_mask], dim=-1)
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Apply softmax to get attention probabilities
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Apply attention dropout if training
        if self.training:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)
        
        # Calculate attention output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        # Project back to hidden size
        attn_output = self.o_proj(attn_output)
        
        # Return output and optionally attention weights and past key-value
        outputs = (attn_output, None, None)
        
        if output_attentions:
            outputs = (attn_output, attn_weights, None)
        
        if use_cache:
            outputs = (attn_output, attn_weights, past_key_value)
        
        return outputs 
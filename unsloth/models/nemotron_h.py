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

from .llama import FastLlamaModel, logger
from transformers import AutoConfig
import torch

__all__ = [
    "FastNemotronHModel",
]

class FastNemotronHModel(FastLlamaModel):
    """
    FastNemotronHModel supports NVIDIA Nemotron-H hybrid Mamba2+Transformer architecture.
    
    This model has a unique layer pattern:
    - 27 Mamba2 layers (state-space models)
    - 4 Attention layers (transformer attention)
    - 25 MLP layers (feed-forward)
    
    Due to the hybrid architecture, we target MLP layers for LoRA fine-tuning,
    as they are the most stable and widely distributed across the model.
    """
    
    @staticmethod
    def get_lora_targets(model_config):
        """
        Returns LoRA target modules specifically for Nemotron-H architecture.
        
        The model uses a specific layer structure:
        - backbone.layers.*.mixer.up_proj (MLP up projection)
        - backbone.layers.*.mixer.down_proj (MLP down projection)
        
        We target 25 MLP layers (50 projections total) for stable LoRA training.
        """
        # MLP layer indices in Nemotron-H architecture
        mlp_layers = [
            1, 3, 5, 8, 10, 12, 15, 17, 19, 22, 24, 26, 28,
            31, 33, 35, 37, 40, 42, 45, 47, 49, 51, 53, 55
        ]
        
        targets = []
        for layer_idx in mlp_layers:
            targets.extend([
                f"backbone.layers.{layer_idx}.mixer.up_proj",
                f"backbone.layers.{layer_idx}.mixer.down_proj"
            ])
        
        return targets
    
    @staticmethod
    def from_pretrained(
        model_name="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        token=None,
        device_map="sequential",
        rope_scaling=None,
        fix_tokenizer=True,
        trust_remote_code=True,
        use_gradient_checkpointing="unsloth",
        resize_model_vocab=None,
        revision=None,
        use_exact_model_name=False,
        *args,
        **kwargs,
    ):
        """
        Load a Nemotron-H model with Unsloth optimizations.
        
        Note: Nemotron-H requires mamba-ssm for full functionality.
        Ensure Python 3.11 environment with compatible mamba-ssm installation.
        """
        
        # Check if mamba-ssm is available
        try:
            import mamba_ssm
            logger.info("Nemotron-H: mamba-ssm detected, using full hybrid architecture support")
        except ImportError:
            logger.warning(
                "Nemotron-H: mamba-ssm not found. Model will load but SSM layers may use naive implementation.\n"
                "For optimal performance, install mamba-ssm in a Python 3.11 environment:\n"
                "  uv venv .venv311 --python 3.11\n"
                "  source .venv311/bin/activate\n"
                "  uv pip install mamba-ssm"
            )
        
        # Load config to verify it's a Nemotron-H model
        config = AutoConfig.from_pretrained(
            model_name,
            token=token,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        
        # Verify model type
        model_type = getattr(config, "model_type", None)
        if model_type not in ["nemotron_h", "nemotronh"]:
            logger.warning(
                f"Model type '{model_type}' may not be Nemotron-H. "
                "Proceeding with Nemotron-H optimizations anyway."
            )
        
        # Get LoRA targets specific to Nemotron-H
        lora_targets = FastNemotronHModel.get_lora_targets(config)
        
        # Override kwargs with Nemotron-H specific settings
        kwargs["lora_targets"] = lora_targets
        
        # Nemotron-H benefits from specific settings
        if "use_cache" not in kwargs:
            kwargs["use_cache"] = False  # Disable KV cache during training
        
        # Nemotron-H models use custom architectures that don't fit standard Llama patterns
        # We need to use the generic FastModel fallback for hybrid architectures
        from .llama import FastModel
        
        # Use generic model loader that supports custom architectures
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            full_finetuning=full_finetuning,
            token=token,
            device_map=device_map,
            rope_scaling=rope_scaling,
            fix_tokenizer=fix_tokenizer,
            trust_remote_code=trust_remote_code,
            use_gradient_checkpointing=use_gradient_checkpointing,
            resize_model_vocab=resize_model_vocab,
            revision=revision,
            return_logits=False,
            fullgraph=True,
            use_exact_model_name=use_exact_model_name,
            *args,
            **kwargs,
        )
        
        # Apply Nemotron-H specific optimizations
        if hasattr(model, "config"):
            # Ensure hybrid architecture is properly configured
            model.config.use_cache = False  # Disable during training
            
            # Log architecture info
            logger.info(
                f"Nemotron-H loaded: {len(lora_targets)} LoRA targets configured for MLP layers"
            )
        
        return model, tokenizer
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
import math
from transformers import PretrainedConfig
from transformers.utils import logging
from typing import List, Optional, Tuple, Union, Dict, Any

logger = logging.get_logger(__name__)

class KBLaMConfig(PretrainedConfig):
    """
    Configuration class for KBLaM integration with Unsloth.
    Based on Microsoft's KBLaM implementation but adapted for Unsloth.
    """
    def __init__(
        self,
        base_model_name_or_path: str = "",
        kb_layer_frequency: int = 3,
        kb_scale_factor: Optional[int] = None,
        top_k_kb: int = 100,
        dynamic_sparsify: bool = False,
        sep_query_head: bool = False,
        attn_implementation: str = "eager",
        **kwargs,
    ):
        self.base_model_name_or_path = base_model_name_or_path
        self.kb_layer_frequency = kb_layer_frequency
        self.kb_scale_factor = kb_scale_factor
        self.top_k_kb = top_k_kb
        self.dynamic_sparsify = dynamic_sparsify
        self.sep_query_head = sep_query_head
        self.attn_implementation = attn_implementation
        super().__init__(**kwargs)


class KBLaMProcessor:
    """
    Processor for KBLaM knowledge base input processing.
    Adapted for Unsloth's efficient processing pipeline.
    """
    def __init__(self, tokenizer, kb_encoder=None):
        self.tokenizer = tokenizer
        self.kb_encoder = kb_encoder
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def prepare_kb_inputs(self, knowledge_base):
        """
        Process knowledge base inputs into the format expected by KBLaM models.
        
        Args:
            knowledge_base: List of (key, value) tuples or pre-encoded embeddings
            
        Returns:
            Processed knowledge base in the format expected by the model
        """
        if self.kb_encoder is None:
            raise ValueError("KB encoder is not initialized. Cannot process knowledge base.")
            
        # Process the knowledge base
        if isinstance(knowledge_base[0][0], str):
            # Text knowledge base
            return self.kb_encoder.encode(knowledge_base)
        elif isinstance(knowledge_base[0][0], torch.Tensor):
            # Pre-encoded knowledge base
            return self.kb_encoder.encode_base_embeddings(knowledge_base)
        else:
            raise ValueError(f"Unsupported knowledge base format: {type(knowledge_base[0][0])}")


# Utility functions for KBLaM module
def get_kblam_projector(
    projector_type: str, 
    in_dim: int, 
    out_dim: int, 
    projector_kwargs: dict = {"mlp_depth": 1, "mlp_hidden_dim": 512}
) -> nn.Module:
    """
    Creates a projector module for KBLaM knowledge base encoding.
    Optimized for Unsloth's memory-efficient approach.
    
    Args:
        projector_type: Type of projector ("identity", "linear", or "mlp")
        in_dim: Input dimension 
        out_dim: Output dimension
        projector_kwargs: Additional parameters for projector configuration
        
    Returns:
        Projector module
    """
    if projector_type == "identity":
        return nn.Identity()
    elif projector_type == "linear":
        return nn.Linear(in_dim, out_dim)
    elif projector_type == "mlp":
        mlp_depth = projector_kwargs.get("mlp_depth", 1)
        mlp_hidden_dim = projector_kwargs.get("mlp_hidden_dim", 512)
        
        modules = [nn.Linear(in_dim, mlp_hidden_dim)]
        for _ in range(mlp_depth):
            modules.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim))
            modules.append(nn.GELU())
        modules.append(nn.Linear(mlp_hidden_dim, out_dim))
        return nn.Sequential(*modules)
    else:
        raise ValueError(f"Unsupported projector type: {projector_type}")


def prepare_model_for_kblam(
    model,
    kb_config: KBLaMConfig,
    kb_encoder = None,
):
    """
    Prepares a model for KBLaM integration by adding necessary components.
    
    Args:
        model: The base model to prepare
        kb_config: KBLaM configuration
        kb_encoder: Optional pre-initialized KB encoder
        
    Returns:
        model: The model with KBLaM components added
        processor: A KBLaM processor for handling knowledge base inputs
    """
    import inspect
    
    # Detect model type and apply appropriate patching
    model_class_name = model.__class__.__name__
    
    if "Llama" in model_class_name:
        # Import here to avoid circular imports
        from .llama import patch_llama_with_kblam
        model = patch_llama_with_kblam(model, kb_config)
        
        # Import specialized processor
        from ..kblam_utils.kb_processor import LlamaKBProcessor
        processor_class = LlamaKBProcessor
        
    elif "Phi" in model_class_name:
        # Future implementation - for now we'll raise an error
        raise NotImplementedError("Phi models with KBLaM are not yet supported")
        
        # This would be the implementation:
        # from .phi import patch_phi_with_kblam
        # model = patch_phi_with_kblam(model, kb_config)
        # from ..kblam_utils.kb_processor import PhiKBProcessor
        # processor_class = PhiKBProcessor
        
    else:
        # Fallback to generic processor
        from ..kblam_utils.kb_processor import KBProcessor
        processor_class = KBProcessor
        logger.warning(
            f"Model type {model_class_name} may not be fully supported for KBLaM integration. "
            f"Using generic KBProcessor. Results may be suboptimal."
        )
    
    # Create processor
    processor = processor_class(model.tokenizer, kb_encoder)
    
    # Add prepare_kb_inputs method to the model
    model.prepare_kb_inputs = processor.prepare_kb_inputs_for_generation
    
    logger.info(f"Model {model_class_name} prepared for KBLaM integration")
    return model, processor 
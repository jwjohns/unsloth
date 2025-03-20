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
from typing import List, Tuple, Dict, Union, Optional, Any
from transformers import BatchFeature
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

class KBProcessor:
    """
    Processor for handling knowledge base inputs for KBLaM models.
    Optimized for Unsloth's memory-efficient processing.
    """
    
    def __init__(self, tokenizer, kb_encoder=None):
        """
        Initialize the KB processor.
        
        Args:
            tokenizer: Tokenizer for processing text inputs
            kb_encoder: KB encoder for encoding knowledge base entries
        """
        self.tokenizer = tokenizer
        self.kb_encoder = kb_encoder
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __call__(
        self,
        knowledge_base: Optional[List[Tuple[str, str]]] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> BatchFeature:
        """
        Process knowledge base and text inputs for KBLaM models.
        
        Args:
            knowledge_base: Knowledge base entries as (key, value) pairs
            text: Text input for the model
            add_generation_prompt: Whether to add prompts for generation
            
        Returns:
            BatchFeature containing processed inputs
        """
        # Process the knowledge base if provided
        kb_kvs = None
        if knowledge_base is not None:
            if self.kb_encoder is None:
                raise ValueError("KB encoder not initialized. Cannot process knowledge base.")
            
            if isinstance(knowledge_base[0][0], torch.Tensor):
                kb_kvs = self.kb_encoder.encode_base_embeddings(knowledge_base)
            else:
                kb_kvs = self.kb_encoder.encode(knowledge_base)
        
        # Process the text
        if text is not None:
            # Format the input text based on the model's requirements
            # This varies by model, but here's a generic approach
            if add_generation_prompt:
                if isinstance(text, list):
                    text = [self._format_input_text(t) for t in text]
                else:
                    text = self._format_input_text(text)
            
            text_inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                **kwargs
            ).to(self.device)
            
            # Combine text inputs and knowledge base
            return BatchFeature(data={**text_inputs, "kb_kvs": kb_kvs})
        else:
            # Return only knowledge base
            return BatchFeature(data={"kb_kvs": kb_kvs})
    
    def _format_input_text(self, text: str) -> str:
        """
        Format input text for the model.
        
        Args:
            text: Input text
            
        Returns:
            Formatted text
        """
        # Default formatting (override in model-specific implementations)
        return text
    
    def prepare_kb_inputs_for_generation(
        self,
        knowledge_base: List[Tuple[str, str]],
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare knowledge base inputs for generation.
        
        Args:
            knowledge_base: Knowledge base entries as (key, value) pairs
            
        Returns:
            Processed knowledge base inputs
        """
        if self.kb_encoder is None:
            raise ValueError("KB encoder not initialized. Cannot process knowledge base.")
            
        return self.kb_encoder.encode(knowledge_base)
    
    def batch_decode(self, *args, **kwargs):
        """
        Decode a batch of token IDs to strings.
        
        Args:
            *args: Arguments for the tokenizer's batch_decode method
            **kwargs: Keyword arguments for the tokenizer's batch_decode method
            
        Returns:
            Decoded strings
        """
        return self.tokenizer.batch_decode(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        """
        Decode token IDs to a string.
        
        Args:
            *args: Arguments for the tokenizer's decode method
            **kwargs: Keyword arguments for the tokenizer's decode method
            
        Returns:
            Decoded string
        """
        return self.tokenizer.decode(*args, **kwargs)


class LlamaKBProcessor(KBProcessor):
    """
    KB processor for Llama models using KBLaM.
    """
    
    def _format_input_text(self, text: str) -> str:
        """
        Format input text for Llama models.
        
        Args:
            text: Input text
            
        Returns:
            Formatted text
        """
        # Format for Llama (may vary by version)
        if not text.startswith("<|start_header_id|>") and not text.startswith("<s>"):
            return (
                "<|start_header_id|>user<|end_header_id|> "
                + text
                + "<|eot_id|>"
                + "<|start_header_id|>assistant<|end_header_id|>"
            )
        return text


class PhiKBProcessor(KBProcessor):
    """
    KB processor for Phi models using KBLaM.
    """
    
    def _format_input_text(self, text: str) -> str:
        """
        Format input text for Phi models.
        
        Args:
            text: Input text
            
        Returns:
            Formatted text
        """
        # Format for Phi (may vary by version)
        if not text.startswith("<|user|>") and not text.startswith("<s>"):
            return "<|user|>" + text + "<|assistant|>"
        return text 
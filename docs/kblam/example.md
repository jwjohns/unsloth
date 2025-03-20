# KBLaM + Unsloth Integration Example

This document demonstrates how to use Microsoft's Knowledge Base augmented Language Models (KBLaM) with Unsloth for efficient training and inference.

## Installation

```python
# Install dependencies
pip install transformers==4.43.2 accelerate==0.30.1 peft==0.8.2 trl==0.7.10 bitsandbytes==0.43.0 sentence-transformers huggingface_hub
pip install -e .
```

## Loading a Model with KBLaM Support

We'll load a Llama model with KBLaM support to demonstrate the functionality.

```python
# Import libraries
import torch
from unsloth import FastLanguageModel
from huggingface_hub import login

# Log in to Hugging Face (needed for accessing Llama models)
login(token="your_token_here")

# Load model with KBLaM support
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",  # You can use other supported models
    max_seq_length=2048,
    load_in_4bit=True,
    
    # KBLaM parameters
    enable_kblam=True,
    kb_layer_frequency=3,  # Apply KB in every 3rd layer
    kb_scale_factor=None,
    top_k_kb=100,  # Top-k entries to keep from KB
    kb_encoder_name="all-MiniLM-L6-v2",  # Encoder for KB entries
    kb_projector_type="mlp",  # Projector type for KB adapters
    kb_dynamic_sparsify=True,  # Use dynamic sparsification
    kb_sep_query_head=True,  # Use separate query head for KB
)

print(f"Model loaded with KBLaM support!")
```

## Creating a Knowledge Base

Let's create a simple knowledge base to use with our model.

```python
# Create a simple knowledge base
knowledge_base = [
    ("Unsloth", "Unsloth is a library that optimizes LLM fine-tuning with reduced VRAM usage through 4-bit quantization and LoRA/QLoRA techniques."),
    ("KBLaM", "KBLaM (Knowledge Base augmented Language Models) is a method developed by Microsoft for augmenting language models with external knowledge without retrieval modules."),
    ("LoRA", "Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique that reduces memory requirements by adapting low-rank matrices instead of full model weights."),
    ("QLoRA", "QLoRA (Quantized Low-Rank Adaptation) combines quantization with LoRA to further reduce memory requirements for fine-tuning large language models."),
    ("Llama", "Llama is a family of large language models developed by Meta AI (formerly Facebook AI Research)."),
    ("Phi", "Phi is a family of small language models developed by Microsoft Research, known for their strong performance despite smaller parameter counts."),
]

# Process knowledge base for the model
kb_inputs = model.prepare_kb_inputs(knowledge_base)

print(f"Knowledge base with {len(knowledge_base)} entries prepared!")
```

## Inference with KBLaM

Now we can perform inference with the knowledge base.

```python
# Generate text with knowledge base augmentation
questions = [
    "What is Unsloth and how does it help with LLM fine-tuning?",
    "Can you explain what KBLaM is and how it works?",
    "What's the difference between LoRA and QLoRA?",
    "Tell me about the Llama language model family.",
    "What is something you don't know? (This should fallback to the base model)"
]

for question in questions:
    # Format the input
    inputs = tokenizer(question, return_tensors="pt").to("cuda")
    
    # Generate with knowledge base
    outputs = model.generate(
        **inputs,
        kb_kvs=kb_inputs,  # Pass the knowledge base
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
    )
    
    # Decode the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Question: {question}")
    print(f"Response: {response}")
    print("---")
```

## Fine-tuning with KBLaM

Now we'll demonstrate how to fine-tune a model with KBLaM support using Unsloth's efficient training.

```python
# Create a simple dataset for fine-tuning
from datasets import Dataset
import pandas as pd

# Sample dataset with knowledge base references
data = [
    {
        "text": "User: What is Unsloth?\nAssistant: Unsloth is a library that optimizes LLM fine-tuning with reduced VRAM usage through 4-bit quantization and LoRA/QLoRA techniques.",
        "kb": [("Unsloth", "Unsloth is a library that optimizes LLM fine-tuning with reduced VRAM usage through 4-bit quantization and LoRA/QLoRA techniques.")]
    },
    {
        "text": "User: Explain KBLaM.\nAssistant: KBLaM is a method developed by Microsoft for augmenting language models with external knowledge without retrieval modules.",
        "kb": [("KBLaM", "KBLaM (Knowledge Base augmented Language Models) is a method developed by Microsoft for augmenting language models with external knowledge without retrieval modules.")]
    },
    {
        "text": "User: What's the difference between LoRA and QLoRA?\nAssistant: LoRA is a parameter-efficient fine-tuning technique that reduces memory requirements by adapting low-rank matrices instead of full model weights. QLoRA combines quantization with LoRA to further reduce memory requirements.",
        "kb": [("LoRA", "Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique that reduces memory requirements by adapting low-rank matrices instead of full model weights."),
               ("QLoRA", "QLoRA (Quantized Low-Rank Adaptation) combines quantization with LoRA to further reduce memory requirements for fine-tuning large language models.")]
    },
]

# Create a dataset
dataset = Dataset.from_pandas(pd.DataFrame(data))
print(f"Dataset created with {len(dataset)} examples")
```

```python
# Prepare for QLoRA fine-tuning
from peft import LoraConfig
from unsloth.trainer import SFTTrainer
from transformers import TrainingArguments

# Prepare model for LoRA fine-tuning
peft_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=16,
    lora_dropout=0.05,
    # Target the attention projection matrices and KBLaM components
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "q_proj_kb", "score_shift"],
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA to the model
model = FastLanguageModel.get_peft_model(model, peft_config)
print("Model prepared with LoRA")
```

```python
# Function to process dataset with KB
def kb_dataset_preparation_function(examples):
    # Process text inputs
    model_inputs = tokenizer(examples["text"], truncation=True, padding=False, return_tensors=None)
    
    # Process KB inputs - we just pass them through as-is for now
    # The KB will be processed during training
    model_inputs["kb"] = examples["kb"]
    
    return model_inputs

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./kblam_unsloth_model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=100,
    logging_steps=10,
    save_steps=50,
    fp16=True,
)

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    dataset_kb_field="kb",  # Field for KB data
    dataset_preparation_function=kb_dataset_preparation_function,
)

# For demonstration, we won't run the actual training
# To train, uncomment this:
# trainer.train()
```

## Saving the Model

After training, you can save the model with KBLaM components.

```python
# Save the trained model
output_dir = "./kblam_unsloth_model_final"

# Save model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
```

## Conclusion

In this document, we demonstrated how to:

1. Load a model with KBLaM support using Unsloth
2. Prepare a knowledge base and use it for inference
3. Set up fine-tuning with KBLaM and Unsloth's QLoRA optimizations

This integration brings together Microsoft's KBLaM knowledge augmentation approach with Unsloth's memory-efficient training techniques, allowing for powerful knowledge-enhanced language models that can be trained with minimal computational resources. 
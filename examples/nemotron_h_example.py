"""
Example: Fine-tuning NVIDIA Nemotron-H with Unsloth

This example demonstrates how to use Unsloth's optimizations for fine-tuning
NVIDIA's Nemotron-H hybrid Mamba2+Transformer architecture.

Requirements:
- Python 3.11 (for mamba-ssm compatibility)
- CUDA 12.1+ 
- mamba-ssm package installed
"""

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Load Nemotron-H model with Unsloth optimizations
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    max_seq_length=4096,
    dtype=None,  # Auto-detect best dtype
    load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
    trust_remote_code=True,  # Required for Nemotron-H custom code
)

# Add LoRA adapters specifically for Nemotron-H architecture
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=[
        # Nemotron-H uses backbone.layers.*.mixer for MLP projections
        "backbone.layers.1.mixer.up_proj", "backbone.layers.1.mixer.down_proj",
        "backbone.layers.3.mixer.up_proj", "backbone.layers.3.mixer.down_proj",
        "backbone.layers.5.mixer.up_proj", "backbone.layers.5.mixer.down_proj",
        "backbone.layers.8.mixer.up_proj", "backbone.layers.8.mixer.down_proj",
        "backbone.layers.10.mixer.up_proj", "backbone.layers.10.mixer.down_proj",
        "backbone.layers.12.mixer.up_proj", "backbone.layers.12.mixer.down_proj",
        "backbone.layers.15.mixer.up_proj", "backbone.layers.15.mixer.down_proj",
        "backbone.layers.17.mixer.up_proj", "backbone.layers.17.mixer.down_proj",
        "backbone.layers.19.mixer.up_proj", "backbone.layers.19.mixer.down_proj",
        "backbone.layers.22.mixer.up_proj", "backbone.layers.22.mixer.down_proj",
        "backbone.layers.24.mixer.up_proj", "backbone.layers.24.mixer.down_proj",
        "backbone.layers.26.mixer.up_proj", "backbone.layers.26.mixer.down_proj",
        "backbone.layers.28.mixer.up_proj", "backbone.layers.28.mixer.down_proj",
        "backbone.layers.31.mixer.up_proj", "backbone.layers.31.mixer.down_proj",
        "backbone.layers.33.mixer.up_proj", "backbone.layers.33.mixer.down_proj",
        "backbone.layers.35.mixer.up_proj", "backbone.layers.35.mixer.down_proj",
        "backbone.layers.37.mixer.up_proj", "backbone.layers.37.mixer.down_proj",
        "backbone.layers.40.mixer.up_proj", "backbone.layers.40.mixer.down_proj",
        "backbone.layers.42.mixer.up_proj", "backbone.layers.42.mixer.down_proj",
        "backbone.layers.45.mixer.up_proj", "backbone.layers.45.mixer.down_proj",
        "backbone.layers.47.mixer.up_proj", "backbone.layers.47.mixer.down_proj",
        "backbone.layers.49.mixer.up_proj", "backbone.layers.49.mixer.down_proj",
        "backbone.layers.51.mixer.up_proj", "backbone.layers.51.mixer.down_proj",
        "backbone.layers.53.mixer.up_proj", "backbone.layers.53.mixer.down_proj",
        "backbone.layers.55.mixer.up_proj", "backbone.layers.55.mixer.down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
    random_state=42,
)

# Load a sample dataset
dataset = load_dataset("imdb", split="train[:1000]")  # Small subset for demo

def formatting_func(examples):
    """Format dataset for training"""
    texts = []
    for text in examples["text"]:
        # Simple formatting - adjust based on your needs
        texts.append(f"Review: {text[:500]}... Sentiment: ")
    return {"text": texts}

# Prepare training arguments
training_args = TrainingArguments(
    output_dir="./outputs/nemotron_h_unsloth",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    warmup_steps=10,
    max_steps=100,  # Short run for demo
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
)

# Create trainer with Unsloth optimizations
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=4096,
    formatting_func=formatting_func,
    args=training_args,
)

# Start training
print("Starting Nemotron-H fine-tuning with Unsloth optimizations...")
trainer.train()

# Save the fine-tuned model
model.save_pretrained("nemotron_h_lora_model")
tokenizer.save_pretrained("nemotron_h_lora_model")

# Optional: Merge LoRA weights back to base model for inference
# model = FastLanguageModel.merge_and_unload(model)
# model.save_pretrained("nemotron_h_merged")

print("Training complete! Model saved to ./nemotron_h_lora_model")
print("\nNemotron-H specific notes:")
print("- 50 MLP target modules were fine-tuned (25 layers × 2 projections)")
print("- Hybrid architecture preserves SSM efficiency while training MLP layers")
print("- Use Python 3.11 environment for best mamba-ssm compatibility")
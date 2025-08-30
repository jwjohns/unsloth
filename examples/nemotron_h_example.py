from unsloth import FastLanguageModel
from unsloth.models.nemotron_h import FastNemotronHModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Load NVIDIA Nemotron-H with Unsloth optimizations
# Nemotron-H is a hybrid Mamba2+Transformer architecture (27M + 4A + 25MLP layers)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    max_seq_length = 2048,
    load_in_4bit = True,
    trust_remote_code = True,
)

# Get optimal LoRA targets for Nemotron-H (all 25 MLP layers = 50 projections)
nemotron_targets = FastNemotronHModel.get_lora_targets(model.config)

# Setup LoRA for Nemotron-H
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = nemotron_targets,
    lora_alpha = 32,
    lora_dropout = 0.1,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 42,
)

# Load dataset
dataset = load_dataset("imdb", split = "train[:1000]")

def formatting_prompts_func(examples):
    texts = []
    for text, label in zip(examples["text"], examples["label"]):
        sentiment = "positive" if label == 1 else "negative"
        text = f"Review: {text}\nSentiment: {sentiment}"
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched = True)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    dataset_num_proc = 2,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 100,
        learning_rate = 2e-4,
        fp16 = True,
        bf16 = False,
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "outputs",
    ),
)

trainer.train()

# Save model
model.save_pretrained("nemotron_h_lora")
tokenizer.save_pretrained("nemotron_h_lora")
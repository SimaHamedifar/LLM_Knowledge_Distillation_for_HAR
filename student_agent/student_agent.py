import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# 1. Load your Generated Teacher Data
dataset = load_dataset("json", data_files="student_ready_data.jsonl", split="train")

# 2. Setup LoRA Config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Adapting attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 3. Load Student Model (e.g., Qwen-7B or Llama-3-8B) in 4-bit
model_id = "Qwen/Qwen2.5-7B" # A perfect student for a 32B Qwen teacher
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True, # Saves VRAM on your A5000s
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 4. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./student_model_har",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4, # Effectively batch size 16
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True, # Use half-precision for speed
)

# 5. Initialize Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="output", # It will learn to generate the reasoning + label
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
)

# 6. Train!
trainer.train()

# 7. Save the LoRA adapters
trainer.model.save_pretrained("./final_student_lora")

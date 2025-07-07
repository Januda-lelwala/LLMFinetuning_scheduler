import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer

# Model and dataset paths
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
DATASET_NAME = "your_dataset_name"  # Replace with your dataset
OUTPUT_DIR = "./results"

# QLoRA parameters
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

# Training parameters
training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=500,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="wandb"
)

def load_model():
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def create_peft_model(model):
    # Prepare for LoRA
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

def train():
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    # Prepare PEFT model
    model = create_peft_model(model)
    
    # Load dataset from local JSONL file
    dataset = load_dataset('json', data_files='dataset.jasonl', split='train')
    
    # Format the dataset for instruction fine-tuning
    def format_instruction(example):
        # Get the instruction, input, and output from the example
        instruction = example['instruction']
        input_text = example.get('input', '')  # Handle cases where input might be empty
        output = example['output']
        
        # Create a structured prompt with both instruction and input if available
        if input_text:
            text = f"""### Instruction: {instruction}
### Input:
{input_text}
### Response:
{output}"""
        else:
            text = f"""### Instruction: {instruction}
### Response:
{output}"""
        
        return {"text": text}
    
    dataset = dataset.map(format_instruction)
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_arguments,
    )
    
    # Train model
    trainer.train()
    
    # Save the fine-tuned model
    trainer.model.save_pretrained("./llama3-7b-scheduler")

if __name__ == "__main__":
    train()

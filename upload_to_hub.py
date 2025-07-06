import os
from pathlib import Path
from huggingface_hub import HfApi, ModelCard, create_repo
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import PeftModel
import torch

def upload_to_hub(
    model_name: str,
    output_dir: str = "./results",
    repo_name: str = "your-username/your-model-name",  # Replace with your HF username and desired model name
    commit_message: str = "Upload trained model",
    private: bool = True,
    push_to_hub: bool = True,
):
    """
    Upload a trained model to the Hugging Face Hub.
    
    Args:
        model_name (str): The base model name or path.
        output_dir (str): Directory where the trained model is saved.
        repo_name (str): Name of the repository on the Hugging Face Hub.
        commit_message (str): Commit message for the upload.
        private (bool): Whether the model should be private on the Hub.
        push_to_hub (bool): Whether to push the model to the Hub.
    """
    # Load the base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Configure 4-bit quantization for inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    
    # Load the base model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load the trained adapter
    model = PeftModel.from_pretrained(model, output_dir)
    
    # Merge the adapter with the base model
    model = model.merge_and_unload()
    
    # Create a model card
    model_card = f"""---
language: en
license: mit
library_name: transformers
---

# {repo_name.split('/')[-1]}

This is a fine-tuned version of {model_name} using QLoRA.

## Training procedure

The model was fine-tuned using QLoRA with 4-bit quantization.

### Training hyperparameters
- Learning rate: 2e-4
- Batch size: 4
- Epochs: 3
- Optimizer: paged_adamw_32bit
- LR scheduler: cosine
- Weight decay: 0.001
"""

    # Save the model and tokenizer
    save_dir = Path("final_model")
    save_dir.mkdir(exist_ok=True)
    
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # Save model card
    with open(save_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(model_card)
    
    if push_to_hub:
        # Create repository if it doesn't exist
        api = HfApi()
        api.create_repo(
            repo_id=repo_name,
            private=private,
            exist_ok=True,
            repo_type="model"
        )
        
        # Upload the model to the Hub
        api.upload_folder(
            folder_path=str(save_dir),
            repo_id=repo_name,
            commit_message=commit_message
        )
        
        print(f"Model successfully uploaded to: https://huggingface.co/{repo_name}")
    else:
        print(f"Model saved locally at: {save_dir.absolute()}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload a trained model to Hugging Face Hub")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B",
                       help="Base model name or path")
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Directory where the trained model is saved")
    parser.add_argument("--repo_name", type=str, required=True,
                       help="Name of the repository on the Hugging Face Hub (e.g., 'username/model-name')")
    parser.add_argument("--commit_message", type=str, default="Upload trained model",
                       help="Commit message for the upload")
    parser.add_argument("--private", action="store_true",
                       help="Make the repository private")
    parser.add_argument("--no_upload", action="store_false", dest="push_to_hub",
                       help="Don't upload to Hub, just save locally")
    
    args = parser.parse_args()
    
    upload_to_hub(
        model_name=args.model_name,
        output_dir=args.output_dir,
        repo_name=args.repo_name,
        commit_message=args.commit_message,
        private=args.private,
        push_to_hub=args.push_to_hub
    )

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig

def load_model(model_path, base_model_name="meta-llama/Meta-Llama-3-8B"):
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load the fine-tuned LoRA model
    model = PeftModel.from_pretrained(model, model_path)
    model = model.merge_and_unload()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_response(prompt, model, tokenizer, max_length=1024):
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True
    )
    
    # Format the prompt with clear JSON structure
    formatted_prompt = f"""### Instruction: {prompt}
### Response:
{{"schedule": [
  {{
    "date": "2025-07-04",
    "time": "09:00 - 10:00",
    "name": "Task Name",
    "priority": "high",
    "recurrence": "once",
    "description": "Task description"
  }}
]}}"""

    # Generate response
    result = pipe(
        formatted_prompt,
        max_new_tokens=600,
        pad_token_id=tokenizer.eos_token_id
    )[0]['generated_text']
    
    # Extract the JSON part of the response
    try:
        # Find the start and end of the JSON response
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            return "Error: Could not parse response as JSON"
            
        json_str = result[json_start:json_end]
        
        # Parse and pretty print the JSON
        import json
        response = json.loads(json_str)
        return json.dumps(response, indent=2)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print("Raw response:", result)
        return "Error: Could not parse the schedule. Please try again."
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "An unexpected error occurred. Please try again."

if __name__ == "__main__":
    # Load your fine-tuned model
    MODEL_PATH = "./llama3-7b-scheduler"  # Update this path
    model, tokenizer = load_model(MODEL_PATH)
    
    print("AI Scheduler Assistant (type 'quit' to exit)")
    print("Example: Schedule a meeting with John tomorrow at 2pm for 1 hour")
    print("-" * 50)
    
    # Test the model
    while True:
        user_input = input("\nYour scheduling request: ")
        if user_input.lower() == 'quit':
            break
            
        response = generate_response(user_input, model, tokenizer)
        print("\nScheduled Tasks:")
        print("-" * 30)
        print(response)
        print("-" * 30)

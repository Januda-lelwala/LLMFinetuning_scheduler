import torch
import json
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# Pydantic models for request/response validation
class ScheduleItem(BaseModel):
    time: str = Field(..., description="Time range in 'HH:MM - HH:MM' format")
    name: str = Field(..., description="Name of the scheduled item")
    priority: str = Field("medium", description="Priority: 'high', 'medium', or 'low'")
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"), 
                     description="Date in 'YYYY-MM-DD' format")
    recurrence: str = Field("once", description="Recurrence: 'once', 'daily', 'weekly', or 'monthly'")
    description: Optional[str] = Field(None, description="Optional description")

    @validator('time')
    def validate_time_format(cls, v):
        try:
            start, end = v.split(' - ')
            datetime.strptime(start, '%H:%M')
            datetime.strptime(end, '%H:%M')
            return v
        except ValueError:
            raise ValueError("Time must be in 'HH:MM - HH:MM' format")

    @validator('priority')
    def validate_priority(cls, v):
        if v.lower() not in ['high', 'medium', 'low']:
            raise ValueError("Priority must be 'high', 'medium', or 'low'")
        return v.lower()

    @validator('recurrence')
    def validate_recurrence(cls, v):
        if v.lower() not in ['once', 'daily', 'weekly', 'monthly']:
            raise ValueError("Recurrence must be 'once', 'daily', 'weekly', or 'monthly'")
        return v.lower()

class ScheduleRequest(BaseModel):
    prompt: str = Field(..., description="User's scheduling request")
    existing_schedule: List[ScheduleItem] = Field(default_factory=list, 
                                               description="List of existing schedule items")
    max_retries: int = Field(3, ge=1, le=5, 
                            description="Maximum number of retries for conflict resolution")

class ScheduleResponse(BaseModel):
    schedule: List[ScheduleItem] = Field(..., description="Generated schedule items")
    conflicts: List[Dict[str, Any]] = Field(default_factory=list, 
                                         description="List of detected conflicts")
    resolution_details: List[Dict[str, Any]] = Field(default_factory=list,
                                                  description="Details of conflict resolutions")
    success: bool = Field(..., description="Whether the scheduling was successful")
    attempts: int = Field(..., description="Number of attempts made")

# Initialize FastAPI app
app = FastAPI(
    title="AI Scheduler API",
    description="API for generating and managing schedules with conflict resolution",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def generate_schedule(
    request: ScheduleRequest,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_length: int = 1024
) -> ScheduleResponse:
    """
    Generate a schedule based on the request.
    
    Args:
        request: ScheduleRequest object with prompt and existing schedule
        model: The language model
        tokenizer: The model's tokenizer
        max_length: Maximum length of the generated response
        
    Returns:
        ScheduleResponse with generated schedule and conflict info
    """
    existing_schedule = [item.dict() for item in request.existing_schedule]
    existing_schedule_str = format_schedule_for_prompt(existing_schedule)
    
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
    
    formatted_prompt = f"""### Instruction: {request.prompt}

{existing_schedule_str}

Please generate a schedule in JSON format with the following fields for each item:
- date (YYYY-MM-DD)
- time (HH:MM - HH:MM)
- name (string)
- priority (high/medium/low)
- recurrence (once/daily/weekly/monthly)
- description (string)

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

    # Try multiple times to generate a valid schedule
    for attempt in range(request.max_retries):
        try:
            # Generate response
            result = pipe(
                formatted_prompt,
                max_new_tokens=500,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Extract the generated text
            generated_text = result[0]['generated_text']
            
            # Extract just the JSON part from the response
            json_start = generated_text.find('{')
            json_end = generated_text.rfind('}') + 1
            json_str = generated_text[json_start:json_end]
            
            # Parse the JSON
            response = json.loads(json_str)
            
            # Validate the schedule format
            if 'schedule' not in response or not isinstance(response['schedule'], list):
                raise ValueError("Invalid schedule format: missing 'schedule' array")
                
            # Convert to ScheduleItem objects for validation
            schedule_items = [ScheduleItem(**item) for item in response['schedule']]
            
            # Check for conflicts with existing schedule
            conflicts = find_conflicts([item.dict() for item in schedule_items], existing_schedule)
            
            if conflicts and attempt < request.max_retries - 1:
                # If there are conflicts, update the prompt and try again
                conflict_messages = [c.get('message', 'Unknown conflict') for c in conflicts]
                request.prompt = f"{request.prompt}\n\nPlease avoid these scheduling conflicts: {', '.join(conflict_messages)}"
                continue
                
            # If we get here, either there are no conflicts or we've reached max retries
            return ScheduleResponse(
                schedule=schedule_items,
                conflicts=conflicts,
                resolution_details=[],
                attempts=attempt + 1,
                success=len(conflicts) == 0 or attempt == request.max_retries - 1
            )
            
        except Exception as e:
            if attempt == request.max_retries - 1:
                error_msg = f"Failed to generate valid schedule: {str(e)}"
                return ScheduleResponse(
                    schedule=[],
                    conflicts=[],
                    resolution_details=[{"error": error_msg}],
                    attempts=attempt + 1,
                    success=False
                )
    
    return ScheduleResponse(
        schedule=[],
        conflicts=[],
        resolution_details=[{"error": "Failed to generate schedule after multiple attempts"}],
        attempts=request.max_retries,
        success=False
    )

# Initialize FastAPI endpoints
@app.post("/generate-schedule", response_model=ScheduleResponse)
async def generate_schedule_endpoint(request: ScheduleRequest):
    """
    Generate a schedule based on the user's request and existing commitments.
    
    Example request body:
    ```json
    {
        "prompt": "Schedule a team meeting and code review",
        "existing_schedule": [
            {
                "time": "11:00 - 12:00",
                "name": "Client Call",
                "priority": "high",
                "recurrence": "weekly"
            }
        ],
        "max_retries": 3
    }
    ```
    """
    try:
        # Load model if not already loaded
        if 'model' not in generate_schedule_endpoint.__dict__:
            MODEL_PATH = "./llama3-7b-scheduler"  # Update this path
            generate_schedule_endpoint.model, generate_schedule_endpoint.tokenizer = load_model(MODEL_PATH)
        
        # Generate the schedule
        response = generate_schedule(
            request=request,
            model=generate_schedule_endpoint.model,
            tokenizer=generate_schedule_endpoint.tokenizer
        )
        
        # Attempt to resolve any remaining conflicts
        if response.conflicts:
            resolution = adjust_schedule(
                [item.dict() for item in response.schedule],
                [item.dict() for item in request.existing_schedule]
            )
            response.resolution_details = resolution.get('resolution_details', [])
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

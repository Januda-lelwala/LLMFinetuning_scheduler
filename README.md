# Llama 3 7B Scheduler Fine-tuning

This project demonstrates how to fine-tune the Llama 3 7B model for a scheduling assistant application.

## Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended)
- Hugging Face account with access to Llama 3

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Hugging Face authentication:
   ```bash
   huggingface-cli login
   ```

## Dataset Preparation

### Option 1: Generate Synthetic Data

Use the provided script to generate a synthetic dataset:

```bash
python create_dataset.py
```

This will create a `scheduler_dataset.jsonl` file with examples in the required format.

### Option 2: Use Custom Data

Create a dataset in the following JSONL format (one JSON object per line):

```json
{
  "instruction": "Create a schedule for my day with the following tasks: Team Meeting, Code Review, Lunch Break. Here are my existing commitments: [{\"date\":\"2025-07-04\",\"time\":\"11:00 - 12:00\",\"name\":\"Client Call\",\"priority\":\"high\"}]. Please avoid scheduling conflicts.",
  "response": {
    "schedule": [
      {
        "date": "2025-07-04",
        "time": "09:00 - 10:00",
        "name": "Team Meeting",
        "priority": "high",
        "recurrence": "weekly",
        "description": "Weekly team sync"
      },
      {
        "date": "2025-07-04",
        "time": "10:15 - 10:45",
        "name": "Code Review",
        "priority": "medium",
        "recurrence": "daily",
        "description": "Review pull requests"
      },
      {
        "date": "2025-07-04",
        "time": "12:15 - 12:45",
        "name": "Lunch Break",
        "priority": "low",
        "recurrence": "daily",
        "description": "Lunch break"
      }
    ]
  }
}
```

### Dataset Fields

- `instruction`: User's scheduling request
- `response`: JSON object containing:
  - `schedule`: Array of scheduled tasks with:
    - `date`: Date in YYYY-MM-DD format
    - `time`: Time range in HH:MM - HH:MM format
    - `name`: Task name
    - `priority`: Task priority (high/medium/low)
    - `recurrence`: Recurrence pattern (daily/weekly/once)
    - `description`: Task description

## Training

1. Update `train.py` with your dataset name and paths
2. Run training:
   ```bash
   python train.py
   ```

## Inference

To test your fine-tuned model:

```bash
python inference.py
```

## Model Details

- Base Model: meta-llama/Meta-Llama-3-8B
- Fine-tuning Method: QLoRA
- Training Parameters:
  - Learning Rate: 2e-4
  - Batch Size: 4 (effective batch size: 16)
  - Epochs: 3
  - Sequence Length: 2048

## Hardware Requirements

- GPU: NVIDIA GPU with at least 24GB VRAM (for 7B model)
- RAM: 32GB+
- Disk: ~30GB free space

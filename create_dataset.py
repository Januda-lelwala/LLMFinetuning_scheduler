import json
from datetime import datetime, timedelta
import random

# Example tasks with variations
tasks = [
    {"name": "Team Meeting", "duration": 60, "priority": "high", "recurrence": "weekly"},
    {"name": "Code Review", "duration": 90, "priority": "medium", "recurrence": "daily"},
    {"name": "Lunch Break", "duration": 30, "priority": "low", "recurrence": "daily"},
    {"name": "Project Planning", "duration": 120, "priority": "high", "recurrence": "weekly"},
    {"name": "Exercise", "duration": 60, "priority": "medium", "recurrence": "daily"},
    {"name": "Learning Session", "duration": 90, "priority": "medium", "recurrence": "daily"},
]

def generate_datetime():
    # Generate a random time between 8 AM and 6 PM
    hour = random.randint(8, 17)
    minute = random.choice([0, 15, 30, 45])
    return datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)

def generate_schedule_example():
    # Select random tasks (2-4 tasks per day)
    num_tasks = random.randint(2, 4)
    selected_tasks = random.sample(tasks, num_tasks)
    
    # Sort tasks by priority (high to low)
    priority_order = {"high": 0, "medium": 1, "low": 2}
    selected_tasks.sort(key=lambda x: priority_order[x["priority"]])
    
    # Generate schedule
    schedule = []
    current_time = generate_datetime()
    
    for task in selected_tasks:
        end_time = current_time + timedelta(minutes=task["duration"])
        
        schedule.append({
            "date": current_time.strftime("%Y-%m-%d"),
            "time": f"{current_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}",
            "name": task["name"],
            "priority": task["priority"],
            "recurrence": task["recurrence"],
            "description": f"{task['name']} session"
        })
        
        # Add some buffer time between tasks
        current_time = end_time + timedelta(minutes=random.choice([15, 30, 45]))
    
    return schedule

def generate_dataset(num_examples=100, output_file="scheduler_dataset.jsonl"):
    with open(output_file, 'w') as f:
        for _ in range(num_examples):
            schedule = generate_schedule_example()
            task_names = ", ".join([task['name'] for task in schedule])
            
            example = {
                "instruction": f"Create a schedule for my day with the following tasks: {task_names}",
                "response": {
                    "schedule": schedule
                }
            }
            
            f.write(json.dumps(example) + "\n")

if __name__ == "__main__":
    print("Generating dataset...")
    generate_dataset(num_examples=100, output_file="scheduler_dataset.jsonl")
    print("Dataset generated successfully as 'scheduler_dataset.jsonl'")
    print("To use this dataset for training, upload it to Hugging Face Hub or use it locally.")

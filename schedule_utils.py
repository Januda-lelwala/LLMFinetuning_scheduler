from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json

def parse_time(time_str: str) -> tuple[datetime, datetime]:
    """Parse time string in format 'HH:MM - HH:MM' into datetime objects"""
    try:
        start_str, end_str = time_str.split(' - ')
        date = datetime.now().date()
        start_time = datetime.strptime(start_str, '%H:%M').time()
        end_time = datetime.strptime(end_str, '%H:%M').time()
        return (
            datetime.combine(date, start_time),
            datetime.combine(date, end_time)
        )
    except ValueError:
        raise ValueError(f"Invalid time format: {time_str}. Expected 'HH:MM - HH:MM'")

def has_time_overlap(time1: str, time2: str) -> bool:
    """Check if two time ranges overlap"""
    try:
        start1, end1 = parse_time(time1)
        start2, end2 = parse_time(time2)
        
        # Check if one range starts before the other ends and vice versa
        return (start1 < end2) and (start2 < end1)
    except ValueError:
        return False

def find_conflicts(new_schedule: List[Dict[str, Any]], 
                  existing_schedule: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Find scheduling conflicts between new and existing schedules.
    Returns a list of conflicts with details.
    """
    conflicts = []
    
    for new_item in new_schedule:
        for existing_item in existing_schedule:
            # Only check items on the same date
            if new_item.get('date') == existing_item.get('date'):
                if has_time_overlap(new_item['time'], existing_item['time']):
                    conflicts.append({
                        'new_item': new_item,
                        'existing_item': existing_item,
                        'conflict_type': 'time_overlap',
                        'message': f"'{new_item['name']}' conflicts with existing '{existing_item['name']}'"
                    })
    
    return conflicts

def adjust_schedule(proposed_schedule: List[Dict[str, Any]], 
                   existing_schedule: List[Dict[str, Any]],
                   work_hours: tuple[str, str] = ('09:00', '18:00')) -> Dict[str, Any]:
    """
    Adjust proposed schedule to avoid conflicts with existing schedule.
    Returns a dictionary with the adjusted schedule and conflict resolution details.
    """
    conflicts = find_conflicts(proposed_schedule, existing_schedule)
    adjusted_schedule = proposed_schedule.copy()
    resolution_details = []
    
    for conflict in conflicts:
        new_item = conflict['new_item']
        existing_item = conflict['existing_item']
        
        # Try to reschedule the new item after the conflicting item
        _, existing_end = parse_time(existing_item['time'])
        new_start = (existing_end + timedelta(minutes=15)).strftime('%H:%M')
        new_end = (existing_end + timedelta(hours=1)).strftime('%H:%M')
        
        # Check if the new time is within work hours
        work_start, work_end = work_hours
        if new_start < work_start or new_end > work_end:
            resolution_details.append({
                'conflict': conflict,
                'resolution': 'Could not reschedule within work hours',
                'status': 'unresolved'
            })
            continue
            
        # Update the time for the new item
        new_item['time'] = f"{new_start} - {new_end}"
        resolution_details.append({
            'conflict': conflict,
            'resolution': f"Rescheduled to {new_item['time']}",
            'status': 'resolved'
        })
    
    return {
        'schedule': adjusted_schedule,
        'conflicts': [c for c in conflicts if c not in [d['conflict'] for d in resolution_details if d['status'] == 'resolved']],
        'resolution_details': resolution_details
    }

def format_schedule_for_prompt(schedule: List[Dict[str, Any]]) -> str:
    """Format schedule as a string for inclusion in the prompt"""
    if not schedule:
        return "No existing commitments."
        
    formatted = ["Existing commitments:"]
    for item in schedule:
        formatted.append(
            f"- {item.get('time', '')} | {item.get('name', '')} "
            f"(Priority: {item.get('priority', 'medium')})"
        )
    return "\n".join(formatted)

def validate_schedule(schedule: List[Dict[str, Any]]) -> List[str]:
    """Validate schedule items and return a list of errors"""
    errors = []
    
    for i, item in enumerate(schedule, 1):
        if not item.get('name'):
            errors.append(f"Item {i}: Missing 'name' field")
            
        if not item.get('time'):
            errors.append(f"Item {i}: Missing 'time' field")
        else:
            try:
                parse_time(item['time'])
            except ValueError as e:
                errors.append(f"Item {i}: {str(e)}")
                
        if item.get('priority', '').lower() not in ['high', 'medium', 'low']:
            errors.append(f"Item {i}: Invalid priority '{item.get('priority')}'. Must be 'high', 'medium', or 'low'")
    
    return errors

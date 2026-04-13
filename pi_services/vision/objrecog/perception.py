import random

def interpret_objects(detections, action="detected"):
    """Generate natural language descriptions for detected objects"""
    if not detections:
        return None
    
    # Handle both old format (list of strings) and new format (list of dicts)
    if isinstance(detections[0], str):
        object_names = detections
    else:
        object_names = [det['name'] if isinstance(det, dict) else det for det in detections]
    
    if not object_names:
        return None
    
    # Object-specific responses
    responses = {
        'apple': ["I see an apple 🍎", "There's an apple here", "Found an apple!"],
        'banana': ["A banana appeared 🍌", "I spot a banana", "There's a banana"],
        'bottle': ["I see a bottle", "There's a bottle here", "Found a bottle"],
        'cup': ["I see a cup ☕", "There's a cup", "Found a cup"],
        'laptop': ["I see a laptop 💻", "There's a laptop", "Found a laptop"],
        'cell phone': ["I see a phone 📱", "There's a phone", "Found a phone"],
        'book': ["I see a book 📚", "There's a book", "Found a book"],
        'chair': ["I see a chair", "There's a chair", "Found a chair"],
        'tv': ["I see a TV 📺", "There's a TV", "Found a TV"],
        'clock': ["I see a clock 🕐", "There's a clock", "Found a clock"],
        'keyboard': ["I see a keyboard ⌨️", "There's a keyboard", "Found a keyboard"],
        'mouse': ["I see a mouse 🖱️", "There's a mouse", "Found a mouse"],
    }
    
    # Action-specific prefixes
    action_prefixes = {
        'appeared': ["Hey, ", "Oh, ", "Look, ", ""],
        'disappeared': ["The ", "Hmm, the ", "Oh, the ", ""],
        'detected': ["", "I can see ", "I notice ", ""]
    }
    
    # Action-specific suffixes
    action_suffixes = {
        'appeared': [" just appeared!", " showed up!", " is here now!", "!"],
        'disappeared': [" is gone now", " disappeared", " left", " vanished"],
        'detected': ["", " in view", " here", ""]
    }
    
    results = []
    
    for obj_name in object_names[:3]:  # Limit to 3 objects to avoid spam
        # Get specific response or generic one
        if obj_name in responses:
            base_response = random.choice(responses[obj_name])
        else:
            base_response = f"I see a {obj_name.replace('_', ' ')}"
        
        # Add action context
        if action in action_prefixes:
            prefix = random.choice(action_prefixes[action])
            suffix = random.choice(action_suffixes[action])
            
            if action == 'disappeared':
                response = f"{prefix}{obj_name.replace('_', ' ')}{suffix}"
            else:
                response = f"{prefix}{base_response}{suffix}"
        else:
            response = base_response
        
        results.append(response)
    
    # Combine multiple objects naturally
    if len(results) == 1:
        return results[0]
    elif len(results) == 2:
        return f"{results[0]} and {results[1].lower()}"
    else:
        return f"{', '.join(results[:-1])}, and {results[-1].lower()}"

def get_object_category(obj_name):
    """Categorize objects for better understanding"""
    categories = {
        'food': ['apple', 'banana', 'orange', 'sandwich', 'pizza', 'cake', 'donut', 'hot dog'],
        'drink': ['bottle', 'cup', 'wine glass'],
        'tech': ['laptop', 'cell phone', 'tv', 'keyboard', 'mouse', 'remote'],
        'furniture': ['chair', 'couch', 'bed', 'dining table'],
        'kitchen': ['microwave', 'oven', 'toaster', 'sink', 'refrigerator'],
        'personal': ['backpack', 'handbag', 'umbrella', 'tie', 'suitcase'],
        'sports': ['frisbee', 'sports ball', 'tennis racket', 'baseball bat']
    }
    
    for category, items in categories.items():
        if obj_name in items:
            return category
    return 'object'


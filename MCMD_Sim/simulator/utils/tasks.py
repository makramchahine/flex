import random
import numpy as np


task_2_delta = {
    'towards': np.array([-1.0, 0.0, 0.0]),
    'left': np.array([0.0, 1.0, 0.0]),
    'right': np.array([0.0, -1.0, 0]),
    'above': np.array([0, 0, 0.7]),
    'below': np.array([0, 0, -0.7]),
    'behind': np.array([1.0, 0, 0])
}
def get_task_delta(direction, target_idx = None):
     delta = task_2_delta[direction]
     if (target_idx == 0 and direction == 'right') or (target_idx == 1 and direction == 'left'):
          return delta * random.uniform(0.25, 0.35)
     return delta * random.uniform(0.45, 0.55)

def generate_instruction(color = '', direction = 'towards', obj_type = None) -> str:
    if obj_type is None:
        return f"{direction}---{color} ball"
    if color in {'colorless', '', None}:
         return f"{direction}---{obj_type}"
    return f"{direction}---{color} {obj_type}"

def generate_instruction_structured(color = '', direction = 'towards', obj_type = None) -> str:
    action_verbs= [
        "Navigate", "Move", "Go", "Rush", "Travel", "Migrate", "Zoom", 
        "Journey", "Advance", "Approach", "Proceed", "Venture", "Head"
    ]
    obj_words = [
        "target", "spot", "signal", "symbol", "goal", "marker",
        "location", "beacon", "destination"
    ]
    direction_phrases = {
        "left": ["to the left of", "on the left of"],
        "right": ["to the right of", "on the right of"],
        "towards": ["towards the", "to the", "in the direction of the"],
        "above": ["above the", "on top of", "over the"],
        "below": [ "under the", "underneath the", "beneath the"],
        "behind": ["behind the", "on the back of"]
    }

    verb = random.choice(action_verbs)
    dir_phrase = random.choice(direction_phrases.get(direction, ["in some direction of the"]))
    obj = random.choice(obj_words)

    if color in {'colorless', '', None}:
         return f"{verb} {dir_phrase} {obj_type}."

    return f"{verb} {dir_phrase} {color} {obj}."

def generate_instruction_old(color: str, obj_type: str, direction: str) -> str:
    action_verbs = ["Fly", "Navigate", "Move", "Steer", "Guide", "Drift", "Glide", "Circle"]

    templates = {
        "left": [
            lambda v: f"{v} to the left of the {color} {obj_type}",
            lambda v: f"Avoid the {color} {obj_type} by shifting left",
            lambda v: f"{v} away from the {color} {obj_type} by going left",
            lambda v: f"Stay safe by passing to the left of the {color} {obj_type}",
            lambda v: f"Carefully drift to the left of the {color} {obj_type}",
            # lambda v: f"Circle around the {color} {obj_type} to the left"
        ],
        "right": [
            lambda v: f"{v} to the right of the {color} {obj_type}",
            lambda v: f"Avoid the {color} {obj_type} by shifting right",
            lambda v: f"{v} away from the {color} {obj_type} by going right",
            lambda v: f"Stay safe by passing to the right of the {color} {obj_type}",
            lambda v: f"Carefully drift to the right of the {color} {obj_type}",
            # lambda v: f"Circle around the {color} {obj_type} to the right"
        ],
        "towards": [
            lambda v: f"{v} directly towards the {color} {obj_type}",
            lambda v: f"Head straight to the {color} {obj_type}",
            lambda v: f"Guide yourself towards the {color} {obj_type}",
            lambda v: f"Steer directly to the {color} {obj_type}",
            lambda v: f"Glide steadily towards the {color} {obj_type}",
            # lambda v: f"Carefully approach the {color} {obj_type}"
        ]
    }

    verb = random.choice(action_verbs)
    return random.choice(templates.get(direction, [lambda v: "Invalid direction"]))(verb)

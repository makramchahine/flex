import random
import numpy as np


task_2_delta = {
    'towards': np.array([-1.0, 0.0, 0.0]),
    'left': np.array([0.0, 1.0, 0.0]),
    'right': np.array([0.0, -1.0, 0]),
    'up': np.array([0, 0, 1.0]),
    'down': np.array([0, 0, -1.0]),
    'behind': np.array([1.0, 0, 0])
}
def get_task_delta(direction, target_idx = None):
     delta = task_2_delta[direction]
     if (target_idx == 0 and direction == 'right') or (target_idx == 1 and direction == 'left'):
          return delta * random.uniform(0.25, 0.35)
     return delta* random.uniform(0.45, 0.55)

def generate_instruction(color: str, obj_type: str, direction: str) -> str:
    action_verbs = ["Fly", "Navigate", "Move", "Steer", "Guide", "Drift", "Glide", "Circle"]

    templates = {
        "left": [
            lambda v: f"{v} to the left of the {color} {obj_type}",
            lambda v: f"Avoid the {color} {obj_type} by shifting left",
            lambda v: f"{v} away from the {color} {obj_type} by going left",
            lambda v: f"Stay safe by passing to the left of the {color} {obj_type}",
            lambda v: f"Carefully drift to the left of the {color} {obj_type}",
            lambda v: f"Circle around the {color} {obj_type} to the left"
        ],
        "right": [
            lambda v: f"{v} to the right of the {color} {obj_type}",
            lambda v: f"Avoid the {color} {obj_type} by shifting right",
            lambda v: f"{v} away from the {color} {obj_type} by going right",
            lambda v: f"Stay safe by passing to the right of the {color} {obj_type}",
            lambda v: f"Carefully drift to the right of the {color} {obj_type}",
            lambda v: f"Circle around the {color} {obj_type} to the right"
        ],
        "towards": [
            lambda v: f"{v} directly towards the {color} {obj_type}",
            lambda v: f"Head straight to the {color} {obj_type}",
            lambda v: f"Guide yourself towards the {color} {obj_type}",
            lambda v: f"Steer directly to the {color} {obj_type}",
            lambda v: f"Glide steadily towards the {color} {obj_type}",
            lambda v: f"Carefully approach the {color} {obj_type}"
        ]
    }

    verb = random.choice(action_verbs)
    return random.choice(templates.get(direction, [lambda v: "Invalid direction"]))(verb)

from enum import Enum

from utils import generate_init_conditions_closed_loop_inference_2choice, generate_init_conditions_closed_loop_inference_3choice_random

class Task(Enum):
    RB_2CHOICE = "2rb"
    COLOR_5 = "2colors"
    R_SHAPE = "2rshape"
    M_SHAPE = "2mshape"
    OPEN_DICT = "3open_dict"

task = Task("2colors")
env_name = "arena"

if task == Task.RB_2CHOICE:
    objects = ["red ball", "blue ball"]
    targets = ["red ball", "blue ball"]
    num_obj = 2
    generate_init_conditions = generate_init_conditions_closed_loop_inference_2choice
elif task == Task.COLOR_5:
    objects = ["red ball", "blue ball", "yellow ball", "green ball", "purple ball"]
    targets = ["red ball", "blue ball", "yellow ball", "green ball", "purple ball"]
    num_obj = 2
    generate_init_conditions = generate_init_conditions_closed_loop_inference_2choice
elif task == Task.R_SHAPE:
    objects = ["red ball", "red cube", "red pyramid"]
    targets = ["ball", "cube", "pyramid"]
    num_obj = 2
    generate_init_conditions = generate_init_conditions_closed_loop_inference_2choice
elif task == Task.M_SHAPE:
    objects = ["red ball", "blue ball", "yellow ball", "green ball", "purple ball", "red cube", "blue cube", "yellow cube", "green cube", "purple cube", "red pyramid", "blue pyramid", "yellow pyramid", "green pyramid", "purple pyramid"]
    targets = ["ball", "cube", "pyramid"]
    num_obj = 2
    generate_init_conditions = generate_init_conditions_closed_loop_inference_2choice
elif task == Task.OPEN_DICT:
    objects = ["red ball", "blue ball", "jeep", "horse", "dog", "palmtree", "watermelon", "rocket"]
    targets = ["red ball", "blue ball", "jeep", "horse", "dog", "palmtree", "watermelon", "rocket"]
    num_obj = 3
    generate_init_conditions = generate_init_conditions_closed_loop_inference_3choice_random
else:
    raise ValueError("Invalid task")
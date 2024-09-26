import subprocess
import numpy as np
from datetime import datetime
from enum import Enum

from utils import generate_init_conditions_closed_loop_inference_2choice, generate_init_conditions_closed_loop_inference_3choice_random

from argparse import ArgumentParser

def main(configurations):
    for config in configurations:
        try:
            command = [
                "python",
                "runner.py",
                "--cfg_path", config["cfg_path"],
                "--closed_loop_save_path", config["closed_loop_save_path"],
                "--objects_color", *config["objects_color"],
                "--text_instr", config["text_instr"],
                "--selected_index", str(config["selected_index"])
            ]

            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running the command: {e}")


if __name__ == "__main__":

    parser = ArgumentParser()
    # get task name
    parser.add_argument("--task", type=str, required=True)
    # get environment name
    parser.add_argument("--env", type=str, required=True)

    args = parser.parse_args()


    class Task(Enum):
        RB_2CHOICE = "2rb"
        COLOR_5 = "2colors"
        R_SHAPE = "2rshape"
        M_SHAPE = "2mshape"
        OPEN_DICT = "3open_dict"


    task = Task(args.task)
    env_name = args.env

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
        objects = ["red ball", "blue ball", "yellow ball", "green ball", "purple ball", "red cube", "blue cube",
                   "yellow cube", "green cube", "purple cube", "red pyramid", "blue pyramid", "yellow pyramid",
                   "green pyramid", "purple pyramid"]
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


    # choose num_obj objects from the list of objects and shuffle them
    object_colors = np.random.choice(objects, num_obj, replace=False).tolist()
    obj_idx = [i for i in range(num_obj)]
    # shuffle the indices
    np.random.shuffle(obj_idx)
    # reorder the objects based on the shuffled indices
    object_colors = [object_colors[i] for i in obj_idx]


    def gen_custom_tag(model_type, patch_size=None):
        """
        Generate a custom tag based on model type and patch size.
        """
        model_tag = model_type
        patch_tag = f"patchsize{patch_size}" if patch_size else ""

        custom_tag = f"{model_tag}_{patch_tag}" if patch_tag else model_tag

        return custom_tag


    # List of models and their corresponding patch sizes (if applicable)
    models = [
        # {"model_type": "conv", "patch_size": None},
        # {"model_type": "full_img_all_dim_trans", "patch_size": None},
        {"model_type": "full_img_first_dim_trans", "patch_size": None},
        # {"model_type": "linear", "patch_size": None},
        # {"model_type": "simplevit", "patch_size": None},
        {"model_type": "simplevit_2x2", "patch_size": 2},
        {"model_type": "simplevit_4x4_1", "patch_size": 4},
        {"model_type": "simplevit_8x8", "patch_size": 8},
    ]

    # make a string with current date and time
    # this will be used to create a unique save path
    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Generate configurations
    configurations = [
        {
            "cfg_path": f"/home/makramchahine/repos/flex/local/train_flight/{model['model_type']}/config",
            "closed_loop_save_path": "results/" + gen_custom_tag(model['model_type'], model[
                'patch_size']) + "_" + task.value + "_" + env_name + "_" + "_".join(object_colors) + "_" + dt,
            "objects_color": object_colors,
            "text_instr": f"Navigate to the {object_colors[0]}",
            "selected_index": 0,
            "custom_tag": gen_custom_tag(model['model_type'], model['patch_size']),
        }
        for model in models
    ]



    main(configurations)
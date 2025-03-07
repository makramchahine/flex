import subprocess
from config import objects, num_obj, task, env_name
import numpy as np
from datetime import datetime
import sys

# choose num_obj objects from the list of objects and shuffle them
object_colors = np.random.choice(objects, num_obj, replace=False).tolist()
obj_idx = [i for i in range(num_obj)]
# shuffle the indices
np.random.shuffle(obj_idx)
# reorder the objects based on the shuffled indices
object_colors = [object_colors[i] for i in obj_idx]

# make a string with current date and time
# this will be used to create a unique save path
dt = datetime.now().strftime("%Y-%m-%d")

# def gen_custom_tag(model_type, patch_size=None):
#     """
#     Generate a custom tag based on model type and patch size.
#     """
#     model_tag = model_type
#     patch_tag = f"patchsize{patch_size}" if patch_size else ""

#     custom_tag = f"{model_tag}_{patch_tag}" if patch_tag else model_tag

#     return custom_tag




# Generate configurations
configurations = [
    {
        "closed_loop_save_path": "results/" + dt + "/" + task.value + "_" + env_name + "_" + "_".join(object_colors),
        "objects_color": object_colors,
        "text_instr": f"Navigate to the {object_colors[0]}",
        "selected_index": 0,
        "custom_tag": dt,
    }
]


def main():
    from joblib import Parallel, delayed

    def run_command(config):
        try:
            command = [
                "python",
                "runner.py",
                "--closed_loop_save_path", config["closed_loop_save_path"],
                "--objects_color", *config["objects_color"],
                "--text_instr", config["text_instr"],
                "--selected_index", str(config["selected_index"])
            ]

            subprocess.run(command, check=True, stdout=sys.stdout, stderr=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running the command: {e}")

    Parallel(n_jobs=3)(delayed(run_command)(config) for config in configurations)


if __name__ == "__main__":
    main()

import subprocess
from config import objects, num_obj, task, env_name
import numpy as np
from datetime import datetime

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
    #{"model_type": "full_img_first_dim_trans", "patch_size": None},
    # {"model_type": "linear", "patch_size": None},
    # {"model_type": "resnet", "patch_size": None},
    #{"model_type": "simplevit", "patch_size": None},
    #{"model_type": "simplevit_2x2", "patch_size": 2},
    #{"model_type": "simplevit_4x4_1", "patch_size": 4},
    # {"model_type": "simplevit_8x8", "patch_size": 8},
    # {"model_type": "sv2O1C", "patch_size": None},
    # {"model_type": "sv1O1C", "patch_size": None},
    # {"model_type": "sv1OMC", "patch_size": None},
    # {"model_type": "lstm_seq_patch2", "patch_size": 2},
    # {"model_type": "2025-02-26/18-35-10", "patch_size": 2}
    {"model_type": "2025-04-09/15-36-07", "patch_size": 2}
]

# make a string with current date and time
# this will be used to create a unique save path
dt = datetime.now().strftime("%Y_%m_%d_%H")

# Generate configurations
configurations = [
    {
        "cfg_path": f"/home/alex/flex/local/train_flight/{model['model_type']}",
        "closed_loop_save_path": f"results/{dt}", # + gen_custom_tag(model['model_type'], model['patch_size'])+ "_" + task.value + "_" + env_name + "_" + "_".join(object_colors),
        "objects_color": object_colors,
        "selected_index": 0,
        "custom_tag": gen_custom_tag(model['model_type'], model['patch_size']),
    }
    for model in models
]


def main():
    from joblib import Parallel, delayed

    def run_command(config):
        try:
            command = [
                "python",
                "runner.py",
                "--cfg_path", config["cfg_path"],
                "--closed_loop_save_path", config["closed_loop_save_path"],
                "--objects_color", *config["objects_color"],
                "--selected_index", str(config["selected_index"])
            ]

            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running the command: {e}")

    Parallel(n_jobs=3)(delayed(run_command)(config) for config in configurations)


if __name__ == "__main__":
    main()

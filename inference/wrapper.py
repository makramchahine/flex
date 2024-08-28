import subprocess
from config import objects, num_obj, task, env_name
import numpy as np

# choose num_obj objects from the list of objects
object_colors = np.random.choice(objects, num_obj, replace=False).tolist()


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
    {"model_type": "conv", "patch_size": None},
    {"model_type": "full_img_all_dim_trans", "patch_size": None},
    {"model_type": "full_img_first_dim_trans", "patch_size": None},
    {"model_type": "linear", "patch_size": None},
    {"model_type": "simplevit", "patch_size": None},
    {"model_type": "simplevit_2x2", "patch_size": 2},
    {"model_type": "simplevit_4x4_1", "patch_size": 4},
    {"model_type": "simplevit_8x8", "patch_size": 8},
]

# Generate configurations
configurations = [
    {
        "cfg_path": f"/home/makramchahine/repos/flex/local/train_flight/{model['model_type']}/config",
        "closed_loop_save_path": "results/" + gen_custom_tag(model['model_type'], model['patch_size'])+ "_" + task.value + "_" + env_name + "_" + "_".join(object_colors),
        "objects_color": object_colors,
        "text_instr": f"Navigate to the {object_colors[0]}",
        "selected_index": 0,
        "custom_tag": gen_custom_tag(model['model_type'], model['patch_size']),
    }
    for model in models
]


def main():
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
    main()

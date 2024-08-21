import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy.printing.pretty.pretty_symbology import line_width

# get colormap of greens
cmap = plt.get_cmap('winter')

# Define the configurations and texts
cfgs = ["simplevit", "conv", "linear", "full_img_first_dim_trans", "full_img_all_dim_trans"]
texts = ["fly to the blue object", "fly to the blue target", "reach the blue goal", "atteins la cible bleue"]

# Load the saved outputs
outputs = np.load("open_loop_outputs.npy")

# Load the ground truth values from the CSV file
ground_truth = pd.read_csv("/home/makramchahine/repos/flex/BLIP2_DATASET/eval/save-flight-02.13.2024_21.41.47.286898/data_out.csv").values
# convert to numpy array and transpose
ground_truth = ground_truth[:, :-1]
ground_truth = np.concatenate(([ground_truth[0, :]], ground_truth), axis=0)


print(f"Outputs shape: {outputs.shape}")
print(f"Ground truth shape: {ground_truth.shape}")


# Check dimensions
assert outputs.shape[-1] == ground_truth.shape[-1], f"Mismatch in the number of outputs. Expected {ground_truth.shape[1]}, got {outputs.shape[1]}"

# Number of configurations, texts, and images
num_configs = outputs.shape[0]
num_texts = outputs.shape[1]
num_images = outputs.shape[2]

# Create subplots
fig, axes = plt.subplots(4, num_configs, figsize=(24, 20), sharex=True)
x = range(num_images)

# Titles and labels for subplots
for i in range(num_configs):
    for j in range(4):  # For each of vx, vy, vz, and yaw
        axes[j, i].plot(x, ground_truth[:, j], label='Ground Truth', color='black', linewidth=0.5)
        for k in range(num_texts):
            axes[j, i].plot(x, outputs[i, k, :, j], label=texts[k], color=cmap(k/num_texts), alpha=0.7)
        if j==0:
            axes[j, i].set_title(cfgs[i])
        if j==3:
            axes[j, i].set_xlabel('Image Index')
        if i==0:
            axes[j, i].set_ylabel(f'{["vx", "vy", "vz", "yaw"][j]}')
# add one legend for all subplots outside the subplots/ above the subplots
axes[0, 0].legend(loc='upper right', bbox_to_anchor=(4.2, 1.3), shadow=True, ncol=5)

# adapt the y limits per row
for j in range(4):
    miin = np.min(outputs[:, :, :, j])
    maax = np.max(outputs[:, :, :, j])
    print(f"Min: {miin}, Max: {maax}")
    for i in range(num_configs):
        axes[j, i].set_ylim([miin, maax])

# Overall title
plt.suptitle('Model Predictions vs Ground Truth', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make space for the main title
plt.show()

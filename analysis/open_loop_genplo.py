import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from networkx.algorithms.bipartite import color
from tqdm import tqdm
from assets.instructions import texts_dict

cmap = plt.get_cmap('autumn')

# Load the saved outputs
results_dir = "results"
data_root = "/home/makramchahine/repos/fm_flight/analysis/activations_50_fix/run_0"
config_names = ["simplevit", "simplevit_2x2", "simplevit_4x4_1", "simplevit_8x8", "linear", "conv", "full_img_first_dim_trans", "full_img_all_dim_trans"]
outputs = {cfg: np.load(os.path.join(results_dir, f"{cfg}.npy"), allow_pickle=True).item() for cfg in config_names}

# Directories and metrics
dirs = list(outputs[config_names[0]].keys())
metrics = ["vx", "vy", "vz", "yaw"]

# Create plots for each directory
for dir_name in tqdm(dirs):
    fig, axes = plt.subplots(len(metrics)+1, len(config_names), figsize=(18, 12))

    # on the first row, plot sample images from the directory, equally spaced between 0 and 150
    # take len(config_names) images to fit the number of columns
    # get png files from the directory
    images = [f for f in os.listdir(f"{data_root}/{dir_name}/pybullet_pics0") if f.endswith('.png')]
    images = sorted(images)
    # get indices of images to plot
    idx = np.linspace(0, 120, len(config_names)).astype(int)
    images = [images[i] for i in idx]
    for i, img in enumerate(images):
        img = plt.imread(f"{data_root}/{dir_name}/pybullet_pics0/{img}")
        axes[0, i].imshow(img)
        axes[0, i].axis('off')


    # Load the ground truth values from the CSV file
    ground_truth = pd.read_csv(
        f"{data_root}/{dir_name}/sim_vel.csv").values
    # convert to numpy array and transpose
    ground_truth = ground_truth[:150]

    x = range(150)  # Assuming the number of images is the same across configs

    # Iterate over metrics and configurations
    for metric_idx, metric in enumerate(metrics):
        for cfg_idx, cfg in enumerate(config_names):
            ax = axes[metric_idx+1, cfg_idx]

            # Plot ground truth
            ax.plot(x, ground_truth[:, metric_idx], label='Ground Truth', color='black', linewidth=0.5)

            # Plot each text prompt
            for text_idx, text in enumerate(
                    texts_dict[dir_name.split('_')[1]]):  # Get the correct text prompt for the dir
                y = outputs[cfg][dir_name][text_idx, :150, metric_idx]

                # median filter y
                y = np.convolve(y, np.ones(10) / 10, mode='same')

                ax.plot(x, y, label=f"{text}", alpha=0.7, color=cmap(text_idx/5))

            # Set titles and labels
            if metric_idx == 0:
                ax.set_title(f"{cfg}", fontsize=12)
            if cfg_idx == 0:
                ax.set_ylabel(metric.upper(), fontsize=12)
            if metric_idx == len(metrics) - 1:
                ax.set_xlabel('Image Index', fontsize=12)

            # Add legend to the first subplot only
            if metric_idx == 0 and cfg_idx == 0:
                # put it on top outside the subplots
                ax.legend(loc='upper right', bbox_to_anchor=(4.4, 2.5), shadow=True, ncol=1
                + len(texts_dict[dir_name.split('_')[1]]))

    # fix the y limits for all subplots
    # for each metric, find the min and max across all configurations
    for metric_idx in range(len(metrics)):
        miin = np.min([np.min(outputs[cfg][dir_name][:, :150, metric_idx]) for cfg in config_names])
        maax = np.max([np.max(outputs[cfg][dir_name][:, :150, metric_idx]) for cfg in config_names])
        for cfg_idx in range(len(config_names)):
            axes[metric_idx+1, cfg_idx].set_ylim([miin, maax])


    # Save the plot
    if not os.path.exists('comparison_plots'):
        os.makedirs('comparison_plots')
    plt.savefig(f'comparison_plots/{dir_name}_comparison.png')
    plt.close()

print("Plotting complete.")

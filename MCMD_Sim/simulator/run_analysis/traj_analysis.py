import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import cv2
import time
import io
from IPython import display
from PIL import Image
from tqdm import tqdm

import os
import matplotlib.pyplot as plt
from PIL import Image, ImageChops

def crop_white_border(pil_img, threshold=250):
    bg = Image.new(pil_img.mode, pil_img.size, (255,) * len(pil_img.getbands()))
    diff = ImageChops.difference(pil_img, bg)
    bbox = diff.getbbox()
    if bbox:
        return pil_img.crop(bbox)
    else:
        return pil_img

def show_all_plots(plots_base, plot_tag='sim_pos_xy', filters = None, square_size=256):
    plots_folders = sorted([f for f in os.listdir(plots_base) if os.path.isdir(os.path.join(plots_base, f))])
    if filters is not None:
        plots_folders = [
            pfldr for pfldr in plots_folders
            if all([
                any([
                    fld_detail.startswith(f) or fld_detail.endswith(f)
                    for fld_detail in pfldr.split('_')
                ])
                for f in filters
            ])
        ]
    for i in range(0, len(plots_folders), 25):
        fig, axes = plt.subplots(5, 5, figsize=(20, 20))
        for j in range(5):
            for k in range(5):
                if i+j*5+k >= len(plots_folders):
                    break
                run = plots_folders[i+j*5+k]
                splots = [sp for sp in os.listdir(os.path.join(plots_base, run)) if sp.startswith(plot_tag)]
                if not splots:
                    print(f"Skipping {run}")
                    continue
                im_path = os.path.join(plots_base, run, splots[0])
                try:
                    img = Image.open(im_path).convert('RGB')
                    img = crop_white_border(img)
                    img = img.resize((square_size, square_size))
                    axes[j, k].imshow(img)
                    run_ = run[:5] + run[13:]
                    axes[j, k].set_title(run_[:min(32, len(run))])
                    axes[j, k].axis('off')
                except Exception as e:
                    print(f"Error loading image from {im_path}: {e}")
                    axes[j, k].axis('off')
                # axes[j, k].imshow(plt.imread(im_path))
                # run_ = run[:7] + run[15:]
                # axes[j, k].set_title(run_[:min(32, len(run))])
                # axes[j, k].axis('off')
        plt.show()


def delete_datas(dataset_base, plots_base, bad_runs):
    for run in bad_runs:
        run = '1'+run
        run_type = run[1:].split('_')
        run_type = run_type[0] + '_' + run_type[1][1:]
        if os.path.exists(os.path.join(plots_base, run)):
            shutil.rmtree(os.path.join(plots_base, run))
        if os.path.exists(os.path.join(dataset_base, 'train', run_type, run[1:])):
            shutil.rmtree(os.path.join(dataset_base, 'train', run_type, run[1:]))
        elif os.path.exists(os.path.join(dataset_base, 'eval', run_type, run[1:])):
            shutil.rmtree(os.path.join(dataset_base, 'eval', run_type, run[1:]))
        feature_base = dataset_base.replace('BLIP2_DATASET', 'BLIP2_Features')
        if os.path.exists(os.path.join(feature_base, 'train', run_type, run[1:])):
            shutil.rmtree(os.path.join(feature_base, 'train', run_type, run[1:]))
        elif os.path.exists(os.path.join(feature_base, 'eval', run_type, run[1:])):
            shutil.rmtree(os.path.join(feature_base, 'eval', run_type, run[1:]))

        print(f"Deleted {run}")


def watch_videos(
    plots_base, 
    speed=16.0, 
    wait_time=0.7, 
    resize=(320, 320), 
    skip_every=1, 
    filters = None, 
    thresh = 0.5, 
    step_thresh=0.05
):
    plots_folders = sorted([f for f in os.listdir(plots_base) if os.path.isdir(os.path.join(plots_base, f))])
    if filters is not None:
        plots_folders = [
            pfldr for pfldr in plots_folders
            if all([
                any([
                    fld_detail.startswith(f) or fld_detail.endswith(f)
                    for fld_detail in pfldr.split('_')
                ])
                for f in filters
            ])
        ]
    plots_folders = plots_folders[::skip_every]  # Take every `step`-th folder

    # def get_cmd_path(pfldr, access_mode):
    #     if access_mode == 'tra'
    result_path = os.path.join(plots_base, 'result.csv')
    score_exists = False
    if os.path.exists(result_path):
        score_exists = True
        results = pd.read_csv(result_path)

    for pi, pfldr in enumerate(plots_folders):
        video_name = [f for f in os.listdir(os.path.join(plots_base, pfldr)) if f.endswith('.mp4')]
        if not video_name:
            print(f"Skipping {pfldr}")
            continue

        video_path = os.path.join(plots_base, pfldr, video_name[0])
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            continue

        # vel_path = os.path.join(plots_base, pfldr, 'data', 'data_out.csv')
        # if not os.path.exists(vel_path):
        #     vel_path = os.path.join()
        stop_path = os.path.join(plots_base, pfldr, 'stop_pred.txt')
        stop_arr = None
        if os.path.exists(stop_path):
            stop_arr = np.loadtxt(os.path.join(plots_base, pfldr, 'stop_pred.txt'))

        command = 'Not found in Logs'
        with open(os.path.join(plots_base, pfldr, 'logs.txt'), 'r') as log_file:
            logs = log_file.read()
            logs = logs.split('\n')
            for line in logs:
                if line.strip().startswith('Text Command'):
                    command = line.split(':')[1].strip()
                    break

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for fc in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Create a black banner for text
            h, w, _ = frame.shape
            banner_height = 85
            banner = np.ones((banner_height, w, 3), dtype=np.uint8) * 255

            # Write text on banner
            cv2.putText(banner, f'Frame: {fc + 1}/{total_frames}', (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(banner, f'Video: {pi + 1}/{len(plots_folders)}', (125, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(banner, f'Run: {pfldr}', (5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            cv2.putText(banner, f'Command: {command}', (5, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            if stop_arr is not None:
                stop_val = 0 if fc >= len(stop_arr) else stop_arr[fc] if len(stop_arr.shape) == 1 else stop_arr[fc, 0]
                step_val = 1.0 if fc >= len(stop_arr) or len(stop_arr.shape) < 2 else stop_arr[fc, 1]
                stop_color = (0, 255, 0) if stop_val < thresh else (0, 0, 255)
                step_color = (0, 255, 0) if step_val >= step_thresh else (0, 0, 255)
                cv2.putText(banner, f'Stop: {stop_val:.2f}', (5, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, stop_color, 1)

                cv2.putText(banner, f'Progress: {1 - step_val:.2f}', (125, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, step_color, 1)
                if stop_val >= thresh: frame = np.clip(frame * np.array([0.9, 0.9, 1.2]), 0, 255).astype(np.uint8)
                if step_val < step_thresh: frame = np.clip(frame * np.array([1.2, 0.9, 0.9]), 0, 255).astype(np.uint8)
            
            if score_exists:
                dfr = results.loc[results['run'] == os.path.join('results_test', plots_base.split('/')[-1], pfldr), ['score', 'cos_sim']]
                score, cos_sim = dfr.iloc[0].values
                cv2.putText(banner, f'Score: {score:.2f}', (5, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                cv2.putText(banner, f'Cos Sim: {cos_sim:.2f}', (125, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # Combine and resize
            frame_with_banner = np.vstack((banner, frame))
            frame_resized = cv2.resize(frame_with_banner, (resize[0], resize[1] + banner_height//2))

            # Display frame
            _, buffer = cv2.imencode('.jpeg', frame_resized)
            image = Image.open(io.BytesIO(buffer))
            display.display(image)
            display.clear_output(wait=True)
            time.sleep(1 / speed)

        time.sleep(wait_time)
        cap.release()


def count_frames(database_path):
    all_counts = {'train': {}, 'eval': {}}
    for subdir in ['train', 'eval']:
        for run_type in os.listdir(os.path.join(database_path, subdir)):
            for run in os.listdir(os.path.join(database_path, subdir, run_type)):
                run_path = os.path.join(database_path, subdir, run_type, run)
                feat_path = run_path.replace('BLIP2_DATASET', 'BLIP2_Features')
                img_count = len([f for f in os.listdir(run_path) if f.endswith('.png')])
                feat_count = len([f for f in os.listdir(feat_path) if f.endswith('.pt')])
                if run.startswith('save'):
                    label_count = len(pd.read_csv(os.path.join(run_path, 'data_out.csv')))
                else:
                    label_count = len(pd.read_csv(os.path.join(run_path, 'data_out.csv'), skiprows=1))
                all_counts[subdir][run] = (img_count - 1, label_count, feat_count - 1)
    return all_counts


def count_test(database_path, seq_length=32):
    all_counts = count_frames(database_path)
    for k, v in all_counts['train'].items():
        if v[2] != v[1] or v[1] < seq_length:
            print(k, v)
    for k, v in all_counts['eval'].items():
        if v[2] != v[1] or v[1] < seq_length:
            print(k, v)

def get_all_texts(data_base, run_types=None):
    all_texts = {'train': {}, 'eval': {}}
    if run_types is None: run_types = os.listdir(os.path.join(data_base, 'train'))
    for run_for in ['train', 'eval']:
        for run_type in run_types:
            for run in os.listdir(os.path.join(data_base, run_for, run_type)):
                with open(os.path.join(data_base, run_for, run_type, run, 'label.txt'), 'r') as f:
                    text = f.read()
                all_texts[run_for][run] = text
    return all_texts


def change_run_text(data_base, runs = None, command = None, color = None):
    from ..utils.tasks import generate_instruction
    if runs is None: runs = os.listdir(data_base)
    for run in runs:
        command, color = None, None
        run_path = os.path.join(data_base, run)
        if not os.path.exists(run_path):
            run_type = run.split('_')
            command, color = run_type[0], run_type[1][1:]
            run_type = command + '_' + color
            run_path = os.path.join(data_base, run_type, run)
        with open(os.path.join(run_path, 'label.txt'), 'r') as f:
            text = f.read()
        text = text.split(' ')
        if command is None:
            command = 'towards'
            if 'right' in text: command = 'right'
            if 'left' in text: command = 'left'
        if color is None:
            color = 'blue'
            if 'red' in text: color = 'red'
            if 'green' in text: color = 'green'
        new_text = generate_instruction(color, command)
        with open(os.path.join(run_path, 'label.txt'), 'w') as f:
            f.write(new_text)
        print(f"Changed {run} from {text} to {new_text}")
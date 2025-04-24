# this script reads the results of the inference experiments and makes square videos by model/environment/task
# it saves in a folder called "videos" in the root of the repository

# this is a script to make a video of the images after resizing them to 224x224
import os
import cv2
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import pandas as pd


# Set up command line argument parser
parser = ArgumentParser()
# get task name
parser.add_argument("--task", type=str, required=True)
# get environment name
parser.add_argument("--env", type=str, required=True)

args = parser.parse_args()

# path to the folders
path = "/home/alex/flex/inference/results/"

# list the folders and keep track of their names
folders = os.listdir(path)
# keep only the folders that have the task and environment in their name
folders = [fold for fold in folders if args.task in fold and args.env in fold]
# sort the folders
folders.sort()

# find the unique models tested
models_np = set([fold.split("_")[0] for fold in folders if not "patchsize" in fold and not "full_img" in fold])
models_pp = set(["_".join(fold.split("_")[:2]) for fold in folders if "patchsize" in fold])
models_fi = set(["_".join(fold.split("_")[:4]) for fold in folders if "full_img" in fold])
# combine the three sets
models = models_np.union(models_pp).union(models_fi)
# sort the models
models = list(models)
models.sort()


# hack for single model at a time
models = ["resnet"]

SR = np.zeros((len(models)))

def save_video(models, folders):
    # if no out_video folder exists, create it
    if not os.path.exists("out_video"):
        os.makedirs("out_video")

    for model in models:
        # create the video writer in mp4 format and 10 fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # save video in the out_video folder
        out = cv2.VideoWriter(f"out_video/{model}_{args.task}_{args.env}.mp4", fourcc, 20, (224, 224))

        # filter the folders that correspond to the specific model
        if model == "simplevit":
            runs = [fold for fold in folders if model in fold and not "patchsize" in fold]
        else:
            runs = [fold for fold in folders if model in fold]

        suc = 0

        for run in runs:
            last_view = os.path.join("results", run, "last_view.txt")
            instruction = os.path.join("results", run, "instruction_text.txt")
            imdir = os.path.join("results", run, "pybullet_pics0")
            # list the images under the run
            images = os.listdir(imdir)
            images.sort()

            with open(last_view, "r") as f:
                first_line = f.readline().strip()
                if first_line == "fail":
                    string = "F"
                elif first_line == "success":
                    string = "S"
                    suc += 1

            with open(instruction, "r") as f:
                text = f.readline().strip()

            for i, image in enumerate(images):
                if i < 8000:
                    image = os.path.join(imdir, image)
                    # load the image
                    img = Image.open(image)
                    # make the image have 3 channels
                    img = img.convert('RGB')
                    # resize the image to 224x224
                    img = img.resize((224, 224))
                    # convert the image to a numpy array
                    img = np.array(img)
                    # convert the numpy array to a BGR image
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    # add text to the image with the name of the folder, have it in the top left corner in black color
                    # readable size for 224x224 images
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # smaller text for 224x224 images
                    # cv2.putText(img, f"{string}{fold}", (10, 18), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(img, f"{model}", (10, 18), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    # put instruction text in the bottom center
                    cv2.putText(img, f"{text}", (10, 210), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    # write the image to the video
                    out.write(img)

        # release the video writer with video name as the initial folder name
        out.release()
        print(f"Video {model}_{args.task}_{args.env} saved!")

        SR[models.index(model)] = suc/len(runs)
        print(f"Success rate for {model}: {SR[models.index(model)]}")

    # save SR to a file
    # make the correct headers for the csv file by model
    db = pd.DataFrame(SR, index=models, columns=["SR"])
    db.to_csv(f"out_video/SR_{args.task}_{args.env}.csv")


if __name__ == "__main__":
    save_video(models, folders)

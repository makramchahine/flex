from pprint import pprint
import os
import tqdm
import argparse
import random
import pandas as pd
from collections import OrderedDict
import multiprocessing
import trimesh
import objaverse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-objects", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--keywords", nargs="+", type=str,
                        default=["vehicle", "car"])
    parser.add_argument("--metainfo-csv", type=str, default=None)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    if args.metainfo_csv is not None:
        df = pd.read_csv(args.metainfo_csv)
        download_uids = df["uid"].values.tolist()
    else:
        # load annotations
        print("Loading annotations...")
        if args.n_objects > 0:
            uids = objaverse.load_uids()
            uids = random.sample(uids, args.n_objects)
            annotations = objaverse.load_annotations(uids)
        else:
            annotations = objaverse.load_annotations()
        n_annotations = len(annotations)
        print(f"Loaded {n_annotations} annotations")
        
        # obtain a set of uids by filtering annotation
        print("Filtering annotations...")
        filtered_annotations = dict()
        pbar = tqdm.tqdm(annotations.items(), total=n_annotations)
        for uid, annotation in pbar:
            pbar.set_description(f"{uid}")         
            if include_by_annotation(args, annotation):
                filtered_annotations[uid] = annotation
        
        if args.verbose:
            print_annotations(filtered_annotations)
        
        n_filtered_annotations = len(filtered_annotations)
        print(f"There are {n_filtered_annotations} filtered annotations")
            
        df = OrderedDict(uid=[], name=[], categories=[], viewerUrl=[])
        for uid, annotation in filtered_annotations.items():
            df["uid"].append(uid)
            df["name"].append(annotation["name"])
            df["categories"].append(",".join([v["name"] for v in annotation["categories"]]))
            df["viewerUrl"].append(annotation["viewerUrl"])
        df = pd.DataFrame(df)
        metainfo_path = os.path.join(objaverse.BASE_PATH, "metainfo.csv")
        df.to_csv(metainfo_path)
        print(f"Save metainfo to {metainfo_path}")
        
        download_uids = list(filtered_annotations.keys())
    
    # download objects (objaverse load already check if the object already exist)
    if args.download:
        print("Downloading objects")
        processes = multiprocessing.cpu_count()
        objects = objaverse.load_objects(
            uids=download_uids,
            download_processes=processes
        )


def print_annotations(annotations):
    print("")
    for uid, annotation in annotations.items():
        print(f"================ {uid} =================")
        print_keys = ["name", "categories", "description", "viewerUrl"]
        pprint({k: v for k, v in annotation.items() if k in print_keys})
        print("")


def include_by_annotation(args, annotation):
    counter = 0
    counter_include = 0
    
    # TODO: can use llm for mode advanced retrieval 
    
    for category_info in annotation["categories"]:
        for keyword in args.keywords:
            if keyword in category_info["name"]:
                counter_include += 1
            counter += 1
    
    for keyword in args.keywords:
        if keyword in annotation["name"]:
            counter_include += 1
        counter += 1
            
    include_this = (float(counter_include) / counter) > 0
    
    return include_this


def test2():
    processes = multiprocessing.cpu_count()
    uids = objaverse.load_uids()
    annotations = objaverse.load_annotations(uids[0:10])
    tgt_uids = [uids[2]] # c786b97d08b94d02a1fa3b87d2e86cf1
    objects = objaverse.load_objects(
        uids=tgt_uids,
        download_processes=processes
    )
    tmesh = trimesh.load(objects[tgt_uids[0]])


def test1():
    uids = objaverse.load_uids()
    print(f"There are {len(uids)} objects in total.") # 798759
    
    """ Annotation key
     ['uri', 'uid', 'name', 'staffpickedAt', 'viewCount', 'likeCount', 
      'animationCount', 'viewerUrl', 'embedUrl', 'commentCount', 'isDownloadable', 
      'publishedAt', 'tags', 'categories', 'thumbnails', 'user', 'description', 
      'faceCount', 'createdAt', 'vertexCount', 'isAgeRestricted', 'archives', 'license']
    """
    annotations = objaverse.load_annotations(uids[0:1])
    pprint(annotations)


def load_vista_meshlib(mesh_dir="../../../data/vista/carpack01/"):
    from vista.entities.sensors.MeshLib import MeshLib
    meshlib = MeshLib(mesh_dir)
    
    return meshlib


if __name__ == "__main__":
    main()

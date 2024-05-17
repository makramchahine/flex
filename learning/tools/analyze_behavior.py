import os
import argparse
import tqdm
import pickle
from pprint import pprint
import numpy as np
from enum import Enum
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from src.utils import read_h5


SMALL_SIZE = 20
MEDIUM_SIZE = 40
BIGGER_SIZE = 50

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def main():
    # load data
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--low-dim", type=int, default=10)
    parser.add_argument("--use-soft-clustering-label", action="store_true")
    parser.add_argument("--projection", type=str, default="kmeans",
                        choices=["kmeans", "pca"])
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-traj-train", type=int, default=10)
    parser.add_argument("--classifier-type", type=str, default="linear_svc",
                        choices=["linear_svc"])
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--topk-similarity", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot-dir", type=str, default=None)
    parser.add_argument("--concept-file", type=str, default=None)
    # NOTE: should be aligned with feats.pkl
    parser.add_argument("--feats-write-freq", type=int, default=20) 
    parser.add_argument("--featmap-size", type=int, default=256)
    args = parser.parse_args()
    
    np.random.seed(args.random_seed)
    
    if args.verbose:
        print("Load data")
    info_path = os.path.join(args.root_dir, "info.pkl")
    with open(info_path, "rb") as f:
        info = pickle.load(f)
        
    feats_path = os.path.join(args.root_dir, "feats.pkl")
    dataset_feats = read_h5(feats_path, mode="dataset_read_once", verbose=args.verbose)
    if args.verbose:
        print(f"Dataset size: {len(dataset_feats)}") # N
        
    if args.plot_dir is not None:
        os.makedirs(args.plot_dir, exist_ok=True)
    
    # get behavior
    if args.verbose:
        print("Get behavior")
    behavior = get_behavior(info)
    
    # grouping NOTE: no reset at trials
    if args.verbose:
        print("Clustering to obtain quantized features")
    X1 = []
    for i in range(len(dataset_feats)):
        feats_i = dataset_feats[i]
        X1.append(feats_i)
    X1 = np.concatenate(X1, axis=0)
    
    concepts = ["TBU"] * args.low_dim
    if args.projection == "kmeans":
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=args.low_dim, random_state=args.random_seed, n_init="auto")
        
        if args.use_soft_clustering_label:
            kmeans.fit(X1)
            X1_in_dist_space = kmeans.transform(X1)
            if False: # old
                spectrum_len = X1_in_dist_space.shape[1]
                spectrum = (np.arange(spectrum_len) / spectrum_len - 0.5) * 2
                Y1 = (X1_in_dist_space * spectrum[None,:]).sum(axis=1)
            else:
                Y1 = X1_in_dist_space
        else:
            Y1 = kmeans.fit_predict(X1)
    elif args.projection == "pca":
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=args.low_dim, random_state=args.random_seed)
        pca.fit(X1)
        Y1 = pca.transform(X1)
    else:
        raise ValueError(f"Unrecognized projection {args.projection}")
    
    if args.concept_file is not None:
        import torch
        from lavis.models import load_model_and_preprocess

        assert os.path.exists(args.concept_file)
        with open(args.concept_file, "r") as f:
            lines = f.readlines()
        lines = [v for v in lines if "#" != v[0]]
        concepts_text = np.array([v.replace("\n", "") for v in lines])
        
        device = "cuda" if torch.cuda.is_available else "cpu" # set to cpu for now
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="blip2_feature_extractor",
            model_type="pretrain",
            is_eval=True,
            device=device,
        )
        
        concepts_feat = []
        text_bsize = min(len(concepts_text), 100)
        for text_batch in np.split(concepts_text, len(concepts_text) // text_bsize):
            text_inputs = [txt_processors["eval"](v) for v in text_batch]
            sample = {"text_input": text_inputs}
            with torch.no_grad():
                text_feat = model.extract_features(sample, mode="text")
            concepts_feat.append(text_feat.text_embeds_proj[:, 0, :]) # drop qformer-like embeddings
        concepts_feat = torch.cat(concepts_feat, dim=0).data.cpu().numpy() # K x C
        
        if args.projection == "pca":
            similarity = pca.components_ @ concepts_feat.T
        elif args.projection == "kmeans":
            similarity = kmeans.cluster_centers_ @ concepts_feat.T
        else:
            raise ValueError(f"Unrecognized projection {args.projection} at concept feature comparison")
        best_match_idcs = similarity.argmax(axis=1)
        best_similarity = np.array([similarity[i,v] for i, v in enumerate(best_match_idcs)])
        concepts = concepts_text[best_match_idcs]
        
        pprint({k: v for k, v in zip(concepts, best_similarity)})
        
        # reorder concepts
        if args.topk_similarity > 0:
            sorted_idcs = best_similarity.argsort()[::-1]
            sorted_idcs = sorted_idcs[:args.topk_similarity]
            concepts = concepts[sorted_idcs]
            Y1 = Y1[:, sorted_idcs]
            
            args.low_dim = args.topk_similarity
    
    ## multiple trial runs
    accuracy_train_all_trials = []
    accuracy_val_all_trials = []
    
    print("\n============= Run behavior classifier ===============")
    BehaviorCriteria.print()
    pprint(vars(args))
    for trial_i in range(args.n_trials):
        # prepare train/val data for behavior classifier
        ep_idcs_all = np.arange(len(behavior))
        ep_idcs_train = np.random.choice(ep_idcs_all, args.n_traj_train)
        ep_idcs_val = np.array([v for v in ep_idcs_all if v not in ep_idcs_train])
        
        X2_train, Y2_train = [], []
        X2_val, Y2_val = [], []
        
        global_i = 0
        feats_i = 0
        for ep_i, behavior_ep in enumerate(behavior):
            for step_i, behavior_step in enumerate(behavior_ep):
                if global_i % args.feats_write_freq == 0:
                    feats_label = Y1[feats_i*args.featmap_size:(feats_i + 1)*args.featmap_size]
                    
                    feats_label = feats_label.flatten()
                    
                    if ep_i in ep_idcs_train:
                        X2_train.append(feats_label.copy())
                        Y2_train.append(behavior_step.value)
                    else:
                        X2_val.append(feats_label.copy())
                        Y2_val.append(behavior_step.value)
                    feats_i += 1
                global_i += 1
                
        # train and test behavior classifier
        if args.classifier_type == "linear_svc":
            from sklearn.svm import LinearSVC
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            clf = make_pipeline(StandardScaler(), LinearSVC(dual="auto", random_state=0, tol=1e-5))
            clf.fit(X2_train, Y2_train)
            Y2_hat_train = clf.predict(X2_train)
            Y2_hat_val = clf.predict(X2_val)
            
            if args.plot_dir is not None:
                fH, fW = 16, 16
                assert args.featmap_size == (fH * fW)
                if args.projection == "pca":
                    coef = clf.named_steps['linearsvc'].coef_.reshape(-1, fH, fW, args.low_dim)
                elif args.projection == "kmeans":
                    assert args.use_soft_clustering_label
                    coef = clf.named_steps['linearsvc'].coef_.reshape(-1, fH, fW, args.low_dim)
                else:
                    raise ValueError(f"Unrecognized projection {args.projection} at plotting coefficient")
                
                plot_mode = 3
                nrows, ncols = coef.shape[0], coef.shape[-1]
                if args.topk_similarity > 0:
                    ncols = args.topk_similarity
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*(ncols + 0.2), 7*nrows))
                for i in range(nrows):
                    for j in range(ncols):
                        ax = axes[i, j]
                        ax.set_xticks([])
                        ax.set_yticks([])
                        if plot_mode in [2, 3, 4]:
                            if plot_mode == 2:
                                vmag = max(coef.min(), coef.max())
                            elif plot_mode == 3:
                                vmag = max(coef[i].min(), coef[i].max())
                            else:
                                vmag = max(coef[...,j].min(), coef[...,j].max())
                            vmin, vmax = -vmag, vmag
                            cmap = mpl.colormaps['bwr']
                            coef_normed = (coef[i, ..., j] + vmag) / (2 * vmag)
                            im = ax.imshow(cmap(coef_normed))
                        elif plot_mode == 1:
                            ax.imshow(coef[i, ..., j], vmin=coef[i].min(), vmax=coef[i].max())
                        else:
                            ax.imshow(coef[i, ..., j], vmin=coef.min(), vmax=coef.max())
                            
                    if plot_mode == 3:
                        class RemainderFixed(axes_size.Scaled):
                            def __init__(self, xsizes, ysizes, divider):
                                self.xsizes =xsizes
                                self.ysizes =ysizes
                                self.div = divider

                            def get_size(self, renderer):
                                xrel, xabs = axes_size.AddList(self.xsizes).get_size(renderer)
                                yrel, yabs = axes_size.AddList(self.ysizes).get_size(renderer)
                                bb = Bbox.from_bounds(*self.div.get_position()).transformed(self.div._fig.transFigure)
                                w = bb.width/self.div._fig.dpi - xabs
                                h = bb.height/self.div._fig.dpi - yabs
                                return 0, min([w,h])
                
                        def make_square_axes_with_colorbar(ax, size=0.1, pad=0.1):
                            """ Make an axes square, add a colorbar axes next to it, 
                                Parameters: size: Size of colorbar axes in inches
                                            pad : Padding between axes and cbar in inches
                                Returns: colorbar axes
                            """
                            divider = make_axes_locatable(ax)
                            margin_size = axes_size.Fixed(size)
                            pad_size = axes_size.Fixed(pad)
                            xsizes = [pad_size, margin_size]
                            yhax = divider.append_axes("right", size=margin_size, pad=pad_size)
                            divider.set_horizontal([RemainderFixed(xsizes, [], divider)] + xsizes)
                            divider.set_vertical([RemainderFixed(xsizes, [], divider)])
                            return yhax
                        # ax = axes[i, j+1]
                        # divider = make_axes_locatable(ax)
                        # cax = divider.append_axes('right', size='5%', pad=0.2)
                        bbox = ax.get_position()
                        cax = fig.add_axes([
                            bbox.xmax + 0.1 * (bbox.xmax - bbox.xmin),
                            bbox.ymin,
                            0.10 * (bbox.xmax - bbox.xmin), 
                            bbox.ymax-bbox.ymin,
                        ]) # (left, bottom, width, height)
                        # cax = make_square_axes_with_colorbar(ax, size=0.15, pad=0.1)
                        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                        cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                                        norm=norm,
                                                        orientation='vertical')
                        # fig.colorbar(cb1, cax=cax, orientation='vertical')
                
                conf_mat = confusion_matrix(Y2_val, Y2_hat_val, normalize="true")
                axes[0, 0].set_ylabel(f"Lane Stable \n (Acc. = {conf_mat[0,0]*100:.2f}%)")
                axes[1, 0].set_ylabel(f"Avoidance \n (Acc. = {conf_mat[1,1]*100:.2f}%)")
                axes[2, 0].set_ylabel(f"Recovery \n (Acc. = {conf_mat[2,2]*100:.2f}%)")
                
                for k in range(ncols):
                    axes[0, k].set_title(concepts[k])
                            
                # fig.tight_layout()
                fig.savefig(f"{args.plot_dir}/coef_trial_{trial_i:02d}.png", bbox_inches='tight')
                fig.savefig(f"{args.plot_dir}/coef_trial_{trial_i:02d}.pdf", bbox_inches='tight')
                
                plt.close("all")
        else:
            raise ValueError(f"Unrecognized classifier type {args.classifier_type}")
        
        accuracy_train = accuracy_score(Y2_train, Y2_hat_train)
        accuracy_val = accuracy_score(Y2_val, Y2_hat_val)
        accuracy_train_all_trials.append(accuracy_train)
        accuracy_val_all_trials.append(accuracy_val)
        
        print(f"\n========== Trial {trial_i:02d} ============")
        print(f"Train accuracy: {accuracy_train}")
        print(f"Val accuracy: {accuracy_val}")
        
    print(f"\n========== Average of {args.n_trials:02d} trials ============")
    print(f"Train accuracy: {np.mean(accuracy_train_all_trials)} +- {np.std(accuracy_train_all_trials)}")
    print(f"Val accuracy: {np.mean(accuracy_val_all_trials)} +- {np.std(accuracy_val_all_trials)}")

    
def matched_stepping(args, behavior, dataset_feats):
    # an example of how behavior and dataset_feats match
    global_i = 0
    feats_i = 0
    for ep_i, behavior_ep in tqdm.tqdm(enumerate(behavior), total=len(behavior), desc="Loop through steps"):
        for step_i, behavior_step in enumerate(behavior_ep):
            if global_i % args.feats_write_freq == 0:
                feats_step = dataset_feats[feats_i]
                feats_i += 1
            global_i += 1


class Behavior(Enum):
    LANE_FOLLOWING = 0
    AVOIDANCE = 1
    RECOVERY = 2
    

class BehaviorCriteria:
    DISTANCE_THRESHOLD = 10
    # DISTANCE_THRESHOLD_LANE_FOLLOWING_AFTER_RECOVERY = 12 # must set to the same as distance threshold otherwise will cause unhandle case
    MAX_EPISODE_LEN = 200

    @classmethod
    def print(cls):
        print_str = f"=== {cls.__name__} ===\n"
        for attribute, value in cls.__dict__.items():
            if not attribute.startswith('__') and isinstance(value, (int, float, list, dict)):
                print_str += f"{attribute} = {value}\n"
        print(print_str)


def get_behavior(info):
    out = []
    for info_ep in info:
        out.append([])
        behavior_prev = Behavior.LANE_FOLLOWING
        for info_step in info_ep:
            ego_agent_id = info_step["ego_agent_id"]
            info_step_ego = info_step[ego_agent_id]
            
            ado_agent_id = [k for k in info_step.keys() if k not in [info_step["ego_agent_id"], "ego_agent_id"]][0]
            info_step_ado = info_step[ado_agent_id]
            
            ego_is_front = info_step_ego["frame_number"] > info_step_ado["frame_number"]
            dist = np.linalg.norm(info_step_ado["ego_dynamics"][:2] - info_step_ego["ego_dynamics"][:2])
            is_close = dist <= BehaviorCriteria.DISTANCE_THRESHOLD
            # is_close_after_recovery = dist <= BehaviorCriteria.DISTANCE_THRESHOLD_LANE_FOLLOWING_AFTER_RECOVERY
            if ego_is_front and is_close and (behavior_prev in [Behavior.AVOIDANCE, Behavior.RECOVERY]):
                behavior = Behavior.RECOVERY
            elif is_close and (behavior_prev in [Behavior.LANE_FOLLOWING, Behavior.AVOIDANCE]):
                behavior = Behavior.AVOIDANCE
            elif (not ego_is_front) and (not is_close) and (behavior_prev in [Behavior.LANE_FOLLOWING]):
                behavior = Behavior.LANE_FOLLOWING # lane following before recovery
            elif ego_is_front and (not is_close) and (behavior_prev in [Behavior.RECOVERY, Behavior.LANE_FOLLOWING]):
                behavior = Behavior.LANE_FOLLOWING # lane following after recovery
            else:
                raise ValueError(f"Unhandled case: ego_is_front={ego_is_front}, is_close={is_close}, behavior_prev={behavior_prev}")
            out[-1].append(behavior)
            
            behavior_prev = behavior

    return out


if __name__ == "__main__":
    main()

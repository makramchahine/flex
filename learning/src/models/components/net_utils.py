import os
from omegaconf import DictConfig
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn as nn


def get_norm(norm_cfg: DictConfig, num_features: int):
    if norm_cfg.type in [None, "None"]:
        norm = None
    elif norm_cfg.type == "gn":
        norm = nn.GroupNorm(
            num_groups=math.ceil(num_features/float(norm_cfg.group_size)),
            num_channels=num_features,
            eps=norm_cfg.get("eps", 1e-5),
            affine=norm_cfg.get("affine", True),
        )
    elif norm_cfg.type == "bn":
        norm_cls = nn.BatchNorm2d if norm_cfg.dim == 2 else nn.BatchNorm1d
        norm = norm_cls(
            num_features=num_features,
            eps=norm_cfg.get("eps", 1e-5),
            momentum=norm_cfg.get("momentum", 0.1),
            affine=norm_cfg.get("affine", True),
            track_running_stats=norm_cfg.get("track_running_stats", True),
        )
    else:
        raise ValueError(f"Unrecognized normalization {norm_cfg.type}")     
        
    return norm


def get_activation(act_cfg: DictConfig, num_features: int):
    if act_cfg.type in [None, "None"]:
        act = None
    elif act_cfg.type == "relu":
        act = nn.ReLU(
            inplace=act_cfg.get("inplace", False),
        )
    else:
        raise ValueError(f"Unrecognized activation {act_cfg.type}")

    return act


class ImageToTextFeatModel:
    DEFAULT_TEXT_POOL = ["car", "road", "tree", "sky", "dark"]

    def __init__(self, extract_text_feat, text_pool=None, min_similarity=0., replaceable_concepts=None, replace_prob=0.,
                 feat_enhancement_ratio=0.):
        self.extract_text_feat = extract_text_feat
        if isinstance(text_pool, str) and os.path.exists(text_pool):
            with open(text_pool, "r") as f:
                lines = f.readlines()
            lines = [v for v in lines if "#" != v[0]]
            self.text_pool = [v.replace("\n", "") for v in lines]
        else:
            self.text_pool = list(text_pool if text_pool is not None else self.DEFAULT_TEXT_POOL)
        
        self.text_pool_feats = self.extract_text_feat(self.text_pool) # N x 256
        
        self.min_similarity = min_similarity
        self.replace_prob = replace_prob
        self.feat_enhancement_ratio = feat_enhancement_ratio
        
        if replaceable_concepts is not None:
            with open(replaceable_concepts, "r") as f:
                lines = f.readlines()
            lines = [v for v in lines if "#" != v[0]]
            self.replaceable_concepts = dict()
            curr_keys = []
            prev_is_key = False
            for line in lines:
                line = line.replace("\n", "")
                if "=" == line[0]:
                    curr_key = line.replace("= ", "").replace("=", "")
                    if not prev_is_key:
                        curr_keys = [curr_key]
                    else:
                        curr_keys.append(curr_key)
                    for curr_key in curr_keys:
                        self.replaceable_concepts[curr_key] = []
                    prev_is_key = True
                else:
                    assert len(curr_keys) > 0, "No reference concept to be replaced"
                    for curr_key in curr_keys:
                        self.replaceable_concepts[curr_key].append(line)
                    prev_is_key = False
            
            self.replaceable_concepts_idcs = dict()
            self.replaceable_concepts_feats = dict()
            for key, val in self.replaceable_concepts.items():
                assert key in self.text_pool, f"{key} not in the text pool"
                self.replaceable_concepts_idcs[key] = self.text_pool.index(key)
                self.replaceable_concepts_feats[key] = self.extract_text_feat(val)
        else:
            self.replaceable_concepts = replaceable_concepts

    def __call__(self, img_feats, img_raw=None, model_is_training=False):
        B, C, H, W = img_feats.shape
        assert C == self.text_pool_feats.shape[-1]
        
        if self.text_pool_feats.device != img_feats.device:
            self.text_pool_feats = self.text_pool_feats.to(img_feats)
        
        similarities = torch.einsum("bchw,nc->bnhw", img_feats, self.text_pool_feats)
        
        text_feats = self.text_pool_feats.clone()
        sim_max, sim_max_idcs = similarities.max(dim=1)
        text_feats = text_feats[sim_max_idcs].permute(0, 3, 1, 2)
        
        if False: # getattr(self, "stop", False): # DEV feature enhancement
            self._visualize(similarities, img_raw)
            
            text_idx = 0 # 0 (car)
            n_repeat = similarities.shape[1]
            prior_subspace_feats = torch.einsum("bnhw,nc->bchw", similarities, self.text_pool_feats[text_idx:text_idx+1].repeat(n_repeat, 1)) / n_repeat
            orth_subspace_feats = img_feats - prior_subspace_feats
            orth_subspace_similarities = torch.einsum("bchw,nc->bnhw", orth_subspace_feats, self.text_pool_feats)
            # self._visualize(orth_subspace_similarities, img_raw)

            self.min_similarity = 0.05
            mask = (sim_max > self.min_similarity) & (sim_max_idcs == text_idx)
            prior_enhanced_feats = img_feats + 0.5 * prior_subspace_feats * mask
            prior_enhanced_similarities = torch.einsum("bchw,nc->bnhw", prior_enhanced_feats, self.text_pool_feats)
            # self._visualize(prior_enhanced_similarities, img_raw)
            
            # text_replaced_mask = torch.zeros_like(similarities, dtype=bool)
            # text_replaced_mask[:, text_idx].data = mask.data
            text_replaced_mask = mask[:,None]
            text_replaced_feats = torch.where(text_replaced_mask, text_feats, img_feats)
            text_replaced_similarities = torch.einsum("bchw,nc->bnhw", text_replaced_feats, self.text_pool_feats)
            self._visualize(text_replaced_similarities, img_raw)
            import ipdb; ipdb.set_trace()
        
        if self.replaceable_concepts is not None: 
            if model_is_training: # NOTE: only apply during training
                for key, val in self.replaceable_concepts_feats.items():
                    if val.device != img_feats.device:
                        self.replaceable_concepts_feats[key] = val.to(img_feats)
        
                out_feats = img_feats
                for key, replaceable_idx in self.replaceable_concepts_idcs.items():
                    is_replaced = sim_max_idcs == replaceable_idx
                    replaced_feats_pool = self.replaceable_concepts_feats[key]
                    n_possible_replacements = replaced_feats_pool.shape[0]
                    rand_idcs = torch.randint(n_possible_replacements, (is_replaced.shape.numel(),))
                    replaced_feats = self.replaceable_concepts_feats[key][rand_idcs].reshape(text_feats.shape)
                    
                    mask = is_replaced
                    if self.min_similarity > 0.:
                        sim_above_thresh = (sim_max > self.min_similarity)
                        mask = mask & sim_above_thresh
                    if self.replace_prob > 0.:
                        accept = torch.rand(*mask.shape, device=mask.device) <= self.replace_prob
                        mask = mask & accept
                    out_feats = torch.where(mask[:,None], replaced_feats, out_feats)
            else:
                out_feats = img_feats
        else:
            if self.min_similarity > 0.:
                mask = sim_max > self.min_similarity
                out_feats = torch.where(mask[:,None], text_feats, img_feats)
            else:
                out_feats = text_feats
                
        if self.feat_enhancement_ratio > 0.:
            n_repeat = similarities.shape[1]
            prior_enhanced_feats = out_feats
            for text_idx in range(len(self.text_pool)):
                mask = (sim_max > self.min_similarity) & (sim_max_idcs == text_idx) # NOTE: the sim_max_idcs masking leads to top-1-take-all
                prior_subspace_feats = torch.einsum("bnhw,nc->bchw", similarities, self.text_pool_feats[text_idx:text_idx+1].repeat(n_repeat, 1)) / n_repeat
                prior_enhanced_feats += self.feat_enhancement_ratio * prior_subspace_feats * mask # TODO: may not work well with multiple concepts
            out_feats = prior_enhanced_feats

        return out_feats

    def _visualize(self, similarities, img_raw=None, out_dir="../local/misc/"):
        ## plot 0
        if img_raw is not None:
            cv2.imwrite(os.path.join(out_dir, "image_to_text_0.png"), img_raw)
        
        ## plot 1
        n_subplots = len(self.text_pool)
        fig, axes = plt.subplots(1, n_subplots, figsize=(4.8*n_subplots, 4.8))
        
        vmin, vmax = similarities[0].min().data.cpu().item(), similarities[0].max().data.cpu().item()
        for i, ax in enumerate(axes):
            similarity = similarities[0, i].data.cpu().numpy()
            ax.set_title(self.text_pool[i])
            ax.imshow(similarity, vmin=vmin, vmax=vmax)
        
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "image_to_text_1.png"))
        
        del fig
        del axes
        
        ## plot 2
        fig, ax = plt.subplots(1, 1)
        
        text_idcs = similarities.argmax(dim=1)[0].data.cpu().numpy()
        cmap = plt.cm.jet  # define the colormap
        cmaplist = np.array([cmap(i) for i in range(cmap.N)])
        cmap_idcs = np.linspace(0, cmaplist.shape[0]-1, len(self.text_pool)).astype(int)
        text_idcs_vis = cmaplist[cmap_idcs][text_idcs]
        ax.imshow(text_idcs_vis)
        
        bounds = np.linspace(0, cmaplist.shape[0] + 1, cmaplist.shape[0])
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        ticklabels = [self.text_pool[list(cmap_idcs).index(i)] if i in cmap_idcs else None for i in range(cmaplist.shape[0])]
        ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
        cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
            spacing='proportional', ticks=bounds, boundaries=bounds)
        cb.set_ticklabels(ticklabels)

        fig.savefig(os.path.join(out_dir, "image_to_text_2.png"), bbox_inches='tight')
        
        del fig
        del ax

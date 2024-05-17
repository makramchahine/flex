from typing import Optional, List
import copy
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import clip

from src.models.components.extractors.base import BaseExtractor

## for debugging only
DEBUG = False
DEBUG_DIR = "../local/misc/fastsam_dev"
if DEBUG:
    import os
    os.makedirs(DEBUG_DIR, exist_ok=True)
    
    # only for visualization
import cv2
import matplotlib
cmap = matplotlib.cm.get_cmap("jet")


class CLIPExtractor(BaseExtractor):
    def __init__(
        self,
        name: str,
        jit: bool,
        download_root: str,
        freeze_clip: bool,
        last_linear_layer: Optional[List[int]] = None,
        use_per_pixel_feature_with_sam: Optional[bool] = False,
        segmentor: Optional[bool] = "fastsam",
        segmentor_ckpt: Optional[str] = None,
        fastsam_imgsz: Optional[int] = 320,
        fastsam_iou: Optional[float] = 0.7,
        fastsam_conf: Optional[float] = -0.25,
        sam_sort_by: Optional[str] = "area",
        fastsam_min_area_size: Optional[float] = 100,
        aggregate_by_avg_features: Optional[bool] = False,
        fastsam_fill_missing_masks: Optional[bool] = False,
        roi_method: Optional[str] = "crop", # crop|masked|crop_masked 
        # TODO [fastsam]: fill_missing_mask, apply_concept_fusion
        spatial_subsample_size: Optional[List[int]] = None,
        use_visual_encoder_only: Optional[bool] = False,
    ):
        super().__init__()
        
        device = "cpu" # set to cpu for now
        model, preprocess = clip.load(
            name=name,
            device=device,
            jit=jit,
            download_root=download_root,
        )
        self.model = model
        
        if freeze_clip:
            for param in self.model.parameters():
                param.requires_grad = False
        self.freeze_clip = freeze_clip
        
        if use_per_pixel_feature_with_sam:
            if segmentor == "fastsam":
                from fastsam import FastSAM
                assert segmentor_ckpt is not None
                self.sam = FastSAM(segmentor_ckpt)
                
                self.fastsam_cfg = {
                    "retina_masks": True,
                    "imgsz": fastsam_imgsz,
                    "iou": fastsam_iou,
                    "conf": fastsam_conf,
                    "verbose": False, # set to True to print inference time
                }
                self.fastsam_min_area_size = fastsam_min_area_size
                self.aggregate_by_avg_features = aggregate_by_avg_features
                self.fastsam_fill_missing_masks = fastsam_fill_missing_masks
                self.roi_method = roi_method
            else:
                raise ValueError(f"Unrecognized segmentor {segmentor}")
            self.segmentor = segmentor
            self.sam_sort_by = sam_sort_by
        self.use_per_pixel_feature_with_sam = use_per_pixel_feature_with_sam
        
        if spatial_subsample_size is not None:
            self.spatial_subsample = nn.AdaptiveAvgPool2d(output_size=spatial_subsample_size)
        else:
            self.spatial_subsample = None
            
        if use_visual_encoder_only:
            self.visual_encoder = self.model.visual
            self.visual_encoder.ln_post = nn.Identity() # drop layer norm post to compare with BLIP visual encoder
            self.visual_encoder.proj = None # to avoid reducing spatial dimension
        self.use_visual_encoder_only = use_visual_encoder_only
        
        # instantiate last layer
        if last_linear_layer is not None:
            last_linear_layer = nn.Conv2d(*last_linear_layer, 1, 1, 0)
        self.last_linear_layer = last_linear_layer

    def forward(self, x):
        super().forward(x)
        img = x["image"]
        device = img.device
        
        if self.get_clip_device() != device: # avoid setting device as it takes some time (~1e-3s)
            self.model.to(device)
            self.model.eval()
            
        if "text" in x.keys():
            text = clip.tokenize(x["text"]).to(device)
            text_feature = self.model.encode_text(text)
            # TODO: untested and not yet merged with image feature
        
        if self.use_per_pixel_feature_with_sam:
            if self.segmentor == "fastsam":
                # preprocessing
                assert list(img.shape[-2:]) == [200, 320]
                n_px = 224
                img = TF.resize(img, n_px, interpolation=InterpolationMode.BICUBIC, antialias=True)
                img = TF.center_crop(img, n_px)

                img_sam = torch.clamp(img * 255, 0, 255) # fastsam takes image in range 0-255
                img_sam = img_sam.cpu().numpy().astype(np.uint8)[0].transpose(1, 2, 0) # HACK: otherwise segmentation mask will be of wrong shape
                # NOTE: while img is in BGR and fastsam takes in RGB as input, the preprocessing includes BGR to RGB
                #       please check ultralytics/yolo/engine/predictor.py preprocess
                if DEBUG:
                    import cv2
                    cv2.imwrite(f"{DEBUG_DIR}/sam_inp.png", img_sam)
                
                img_clip = TF.normalize(img, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                
                if DEBUG:
                    ## For debugging only. Comment out during actual runs
                    img_clip = img_clip.cuda()
                    device = img_clip.device
                    self.model.to(device)
                    self.last_linear_layer.to(device)
                    ######
                
                # run fastsam
                try:
                    everything_results = self.sam(img_sam, device=device, **self.fastsam_cfg)
                except:
                    import cv2
                    cv2.imwrite(f"{DEBUG_DIR}/sam_inp_problematic.png", img_sam)
                    exit(9)
                bbox_tmp = everything_results[0].boxes.xywh
                masks_tmp = everything_results[0].masks.data
                masks = []
                for idx, mask_tmp in enumerate(masks_tmp):
                    bool_mask = mask_tmp.detach().bool()
                    area = bool_mask.sum() 
                    if area < self.fastsam_min_area_size: continue
                
                    masks.append({})
                    masks[-1]['segmentation'] = bool_mask
                    masks[-1]['segmentation_np'] = bool_mask.cpu().numpy() # for visualization and missing masks only
                    masks[-1]['bbox'] = (bbox_tmp[idx].detach())
                    masks[-1]['area'] = area
                    ### old: directly using bbox from fastsam, which is buggy
                    # masks[-1]['segmentation'] = mask_tmp.detach().to(bool)
                    # masks[-1]['segmentation_np'] = mask_tmp.detach().cpu().numpy().astype(bool) # for visualization only
                    # masks[-1]['bbox'] = (bbox_tmp[idx].detach().cpu().numpy())
                    # masks[-1]['area'] = bbox_tmp[idx][-1].detach().cpu().numpy() * bbox_tmp[idx][-2].detach().cpu().numpy()
                    #######
                masks = sorted(masks, key=(lambda x: x[self.sam_sort_by]), reverse=True)
                
                if self.fastsam_fill_missing_masks:
                    _, masks_in_one_matrix = get_vis_anns(masks, img_sam)
                    
                    thresh = copy.deepcopy(masks_in_one_matrix)
                    thresh[masks_in_one_matrix == 0] = 255
                    thresh[masks_in_one_matrix != 0] = 0
                    thresh = thresh.astype("uint8")
                    kernel = np.ones((5,5),np.uint8)
                    erosion = cv2.erode(thresh,kernel,iterations = 1)
                    missing_masks = []
                    marker_count, contours2 = cv2.connectedComponents(erosion)
                    for label_for_detected_obj in range(1,marker_count):
                        bool_mask = contours2==label_for_detected_obj
                        mask = np.where(bool_mask, np.uint8(255), np.uint8(0))
                        x,y,w,h = cv2.boundingRect(mask)
                        area = cv2.countNonZero(mask[y:y+h,x:x+w])
                        if area < self.fastsam_min_area_size or w < 1 or h <1: continue
                        
                        missing_masks.append({})
                        missing_masks[-1]['segmentation'] = copy.copy(bool_mask)
                        missing_masks[-1]['bbox'] = (x,y,w,h)
                        missing_masks[-1]['area'] = area
                    
                    masks = missing_masks + masks 
                    masks = sorted(masks, key=(lambda x: x[self.sort_by]), reverse=True)
                
                if DEBUG:
                    masks_of_sam, masks_in_one_matrix = get_vis_anns(masks, img_sam) # visualization
                    cv2.imwrite(f"{DEBUG_DIR}/sam_mask.png", masks_of_sam*255)
                
                # run clip
                clip_inp = []
                for maskidx in range(0, len(masks)):
                    seg = masks[maskidx]["segmentation"]
                    x_min, y_min, x_max, y_max= tuple(compute_bbox(seg))
                    if y_max - y_min <1 or x_max - x_min <1: 
                        continue
                    if self.roi_method == "crop":
                        img_clip_roi = img[:, :, int(y_min) :  int(y_max), int(x_min) : int(x_max)]
                        ### old: directly using bbox from fastsam, which is buggy
                        # _x, _y, _w, _h = tuple(masks[maskidx]["bbox"])  # xywh bounding box
                        # img_clip_roi = img_clip[:, :, int(_y) : int(_y) + int(_h), int(_x) : int(_x) + int(_w)]
                        #######
                        img_clip_roi = TF.resize(img_clip_roi, (n_px, n_px), interpolation=InterpolationMode.BICUBIC, antialias=True)
                    elif self.roi_method == "masked":
                        img_clip_roi = img.clone()
                        img_clip_roi[:,:,~seg] = 0
                    elif self.roi_method == "crop_masked":
                        tmp = img.clone()
                        tmp[:,:,~seg] = 0
                        img_clip_roi = tmp[:,:,int(y_min) :  int(y_max), int(x_min) : int(x_max)]
                        img_clip_roi = TF.resize(img_clip_roi, (n_px, n_px), interpolation=InterpolationMode.BICUBIC, antialias=True)
                    else:
                        raise ValueError(f"Unrecognized roi method {self.roi_method}")
                    clip_inp.append(img_clip_roi)
                clip_inp = torch.cat(clip_inp, dim=0)
                
                if DEBUG:
                    import cv2
                    roi_img_show = []
                    for maskidx in range(0, len(masks)):
                        seg = masks[maskidx]["segmentation"]
                        if self.roi_method == "masked":
                            roi = img_sam.copy()
                            roi[~seg] = 0
                        elif self.roi_method == "crop_masked":
                            x_min, y_min, x_max, y_max= tuple(compute_bbox(seg))
                            if y_max - y_min <1 or x_max - x_min <1: 
                                continue
                            tmp = img_sam.copy()
                            tmp[~seg] = 0
                            roi = tmp[int(y_min) :  int(y_max), int(x_min) : int(x_max)].copy()
                            roi = cv2.resize(roi, (n_px, n_px))
                        elif self.roi_method == "crop":
                            x_min, y_min, x_max, y_max= tuple(compute_bbox(seg))
                            if y_max - y_min <1 or x_max - x_min <1: 
                                continue
                            roi = img_sam[int(y_min) :  int(y_max), int(x_min) : int(x_max)].copy()
                            roi = cv2.resize(roi, (n_px, n_px))
                        else:
                            raise ValueError(f"Unrecognized roi method {self.roi_method}")
                        roi_img_show.append(roi)
                    
                    masks_of_sam, masks_in_one_matrix = get_vis_anns(masks, img_sam) # visualization
                    roi_img_show.insert(0, (masks_of_sam*255).astype(np.uint8))
                    
                    import matplotlib.pyplot as plt
                    nrow = 2
                    ncol = int(np.ceil(len(roi_img_show) / nrow))
                    fig, axes = plt.subplots(nrow, ncol, figsize=(6*ncol, 6*nrow))
                    for i, (ax, roi_show) in enumerate(zip(axes.flatten(), roi_img_show)):
                        ax.imshow(roi_show[...,::-1])
                        ax.set_xticks([])
                        ax.set_yticks([])
                    fig.tight_layout()
                    fig.savefig(f"{DEBUG_DIR}/fastsam_roi_{self.roi_method}.png")
                
                if self.freeze_clip:
                    with torch.no_grad():
                        img_roi_feature = self.model.encode_image(clip_inp)
                else:
                    img_roi_feature = self.model.encode_image(clip_inp)
                
                feat_dim = img_roi_feature.shape[-1]
                bs, _, img_h, img_w = img.shape
                out = torch.zeros((bs, feat_dim, img_h, img_w)).to(device)
                for maskidx in range(len(masks)):
                    _weighted_feat = img_roi_feature[maskidx]
                    _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
                    mask_bool = masks[maskidx]["segmentation"]
                    if self.aggregate_by_avg_features:
                        out[:, :, mask_bool[:, 0], mask_bool[:, 1]] += _weighted_feat[None, :, None]
                    else:
                        out[:, :, mask_bool[:, 0], mask_bool[:, 1]] = _weighted_feat[None, :, None]
                    out[:, :, mask_bool[:, 0], mask_bool[:, 1]] = torch.nn.functional.normalize(
                        out[:, :, mask_bool[:, 0], mask_bool[:, 1]], dim=1,
                    )
                
                # NOTE: subsample in spatial dimension otherwise policy need to process a very large feature map
                if self.spatial_subsample is not None:
                    out = self.spatial_subsample(out)
            else:
                raise ValueError(f"Unrecognized segmentor {self.segmentor}")
        elif self.use_visual_encoder_only:
            def _infer_visual_encoder(x):
                # follow clip/model.py VisionTransformer
                x = self.visual_encoder.conv1(x)  # shape = [*, width, grid, grid]
                x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
                x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
                x = torch.cat([self.visual_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                x = x + self.visual_encoder.positional_embedding.to(x.dtype)
                x = self.visual_encoder.ln_pre(x)
                
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.visual_encoder.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD
                
                ## avoid dropping spatial dimension
                # x = self.visual_encoder.ln_post(x[:, 0, :])
                # if self.visual_encoder.proj is not None:
                #     x = x @ self.visual_encoder.proj
                return x
            
            if self.freeze_clip:
                with torch.no_grad():
                    img_feature = _infer_visual_encoder(img)
            else:
                img_feature = _infer_visual_encoder(img)
            
            fH, fW = 7, 7 # HACK
            assert (img_feature.shape[1] - 1) == (fH * fW)
            out = img_feature[:,:-1].view(img_feature.shape[0], fH, fW, img_feature.shape[-1])
            out = out.permute(0, 3, 1, 2)
        else:
            if self.freeze_clip:
                with torch.no_grad():
                    img_feature = self.model.encode_image(img)
            else:
                img_feature = self.model.encode_image(img)

            out = img_feature[..., None, None] # add H, W dimension
        
        if self.last_linear_layer is not None:
            out = self.last_linear_layer(out)
        
        if DEBUG:
            out = out.cpu()

        return out
    
    def get_clip_device(self):
        if not hasattr(self, "_clip_param"):
            self._clip_param =  next(self.model.parameters())
        return self._clip_param.device
    
    def set_backbone(self, lit_module):
        lit_module.backbone = self.model

    @property
    def modalities(self):
        return ["image", "text", "image_raw"]


def multiclass_vis(class_labels, img_to_viz, num_of_labels, np_used = False,alpha = 1):
    _overlay = img_to_viz.astype(float) / 255.0
    if np_used:
         viz = cmap(class_labels/num_of_labels)[..., :3]
    else:
         class_labels = class_labels.detach().cpu().numpy().astype(float)
         viz = cmap((class_labels/num_of_labels))[..., :3]
    _overlay =  alpha * viz + (1-alpha) * _overlay 
    s_overlay = cv2.cvtColor(np.float32(_overlay), cv2.COLOR_BGR2RGB)  

    return _overlay


def get_vis_anns(anns,img_to_viz):
   
    count = 1
    dum = anns[0]['segmentation_np']
    img = np.zeros((dum.shape[0], dum.shape[1]))
    for ann in anns:
        m = ann['segmentation_np']
        img[m] = count
        count+=1
    _overlay = multiclass_vis(img, img_to_viz, count, np_used = True)

    return _overlay,img


def compute_bbox(np_seg):
    segmentation = torch.where(np_seg == True)

    # Bounding Box
    bbox = 0, 0, 0, 0
    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = int(torch.min(segmentation[1]))
        x_max = int(torch.max(segmentation[1]))
        y_min = int(torch.min(segmentation[0]))
        y_max = int(torch.max(segmentation[0]))

        bbox = x_min, y_min, x_max, y_max
    return bbox

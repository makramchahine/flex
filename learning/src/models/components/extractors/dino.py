from typing import Optional, List
import os
import sys
import torch
import torch.nn as nn

from src.models.components.extractors.base import BaseExtractor

DINO_SAM_ROOT = os.environ.get("DINO_SAM_ROOT", None)
if DINO_SAM_ROOT is None:
    raise ValueError(f"Please set environment variable DINO_SAM_ROOT")
# sys.path.append(f"{DINO_SAM_ROOT}/") # NOTE: already set in macro otherwise won't go through multi-processing
# sys.path.append(f"{DINO_SAM_ROOT}/Segment-and-Track-Anything")
# sys.path.append(f"{DINO_SAM_ROOT}/Segment-and-Track-Anything/aot")

from model_args import aot_args, sam_args, segtracker_args
from DINO.dino_wrapper import get_detector_model, preprocess_frame
from sam.segment_anything import sam_model_registry, SamPredictor
from SegTracker import SegTracker


class DINOExtractor(BaseExtractor):
    def __init__(
        self,
        use_sam: bool,
        freeze_sam: bool,
        dino_strides: int,
        desired_height: int,
        desired_width: int,
        use_16bit: bool,
        use_traced_model: bool,
        freeze_dino: bool,
        last_linear_layer: Optional[List[int]] = None,
        dino_version: Optional[str] = "in_house",
        model_type: Optional[str] = "dino_vits8",
        hub_dir: Optional[str] = "facebookresearch/dino:main",
    ):
        super().__init__()
        
        assert dino_version in ["in_house", "official"]
        
        self.dino_version = dino_version
        self.freeze_dino = freeze_dino
        
        if self.dino_version == "official":
            self.dino = torch.hub.load(hub_dir, model_type)

            self.dino.patch_embed.proj.stride = (dino_strides, dino_strides)
            
            if self.freeze_dino:
                for name, param in self.dino.named_parameters():
                    param.requires_grad = False
        elif self.dino_version == "in_house":
            self.use_sam = use_sam
            
            # instantiate SAM
            if self.use_sam:
                sam_args["sam_checkpoint"] = os.path.join(DINO_SAM_ROOT, sam_args["sam_checkpoint"])
                aot_args["model_path"] = os.path.join(DINO_SAM_ROOT, aot_args["model_path"])
                self.sam = SegTracker(segtracker_args, sam_args, aot_args)
                self.sam.restart_tracker() # TODO: maybe we should reset for every frame?!
                if freeze_sam:
                    pass # TODO
                raise NotImplementedError # TODO
            
            # instantiate DINO
            # hub_dir = os.path.join(DINO_SAM_ROOT, torch.hub.get_dir())
            # torch.hub.set_dir(hub_dir)
            dino_cfg = {
                "dino_strides": dino_strides,
                "desired_height": desired_height,
                "desired_width": desired_width,
                "use_16bit": use_16bit,
                "use_traced_model": use_traced_model,
                "model_type": model_type,
            }
            device = torch.device("cpu") # set to cpu for now
            self.dino = get_detector_model(cfg=dino_cfg, device=device)
            self.dino_cfg = dino_cfg
            
            if self.freeze_dino:
                for name, param in self.dino.extractor.model.named_parameters():
                    param.requires_grad = False
        else:
            raise ValueError(f"Invalid DINO version {self.dino_version}")
        
        # instantiate last layer
        if last_linear_layer is not None:
            last_linear_layer = nn.Conv2d(*last_linear_layer, 1, 1, 0)
        self.last_linear_layer = last_linear_layer

    def forward(self, x):
        img = x["image"]
        device = img.device

        if self.dino_version == "official":
            pass
        elif self.dino_version == "in_house":
            if self.use_sam:
                raise NotImplementedError # TODO
            
            if self.get_dino_device() != device: # avoid setting device as it takes some time (~1e-3s)
                self.dino.to(device)
                self.dino.extractor.model.to(device)
        else:
            raise ValueError(f"Invalid DINO version {self.dino_version}")

        if self.freeze_dino:
            self.dino.eval()
            with torch.no_grad(): # runtime is roughly 1/10x of simple cnn (~6e-3)
                out = self.dino.forward(img)
        else:
            out = self.dino.forward(img)
            
        if self.dino_version == "official":
            out = out[..., None, None] # add H, W dimension
        
        if self.last_linear_layer is not None:
            out = self.last_linear_layer(out)

        return out
    
    def get_dino_device(self):
        if not hasattr(self, "_dino_param"):
            self._dino_param = next(self.dino.extractor.model.parameters())
        return self._dino_param.device
    
    def set_backbone(self, lit_module):
        if self.dino_version == "official":
            lit_module.backbone = self.dino
        elif self.dino_version == "in_house":
            if self.use_sam:
                raise NotImplementedError
            lit_module.backbone = self.dino.extractor.model
        else:
            raise ValueError(f"Invalid DINO version {self.dino_version}")
    
    @property
    def modalities(self):
        return ["image"]

from typing import Optional, List, Tuple
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.modules.utils as nn_utils
import types
from omegaconf import OmegaConf
from lavis.models import load_model_and_preprocess
from lavis.common.registry import registry
from lavis.models.blip_models.blip_outputs import BlipOutputFeatures

from src.utils.utils import get_square_indices
from src.models.components.extractors.base import BaseExtractor
from transformers import BertTokenizer

class BLIPExtractor(BaseExtractor):
    def __init__(
        self,
        name: str,
        model_type: str,
        freeze_blip: bool,
        use_low_dim_feature: Optional[bool] = False,
        use_masked_patch_wise_feature: Optional[bool] = True,
        use_visual_encoder_only: Optional[bool] = False,
        append_global_features: Optional[bool] = False,
        last_linear_layer: Optional[List[int]] = None,
        checkpoint: Optional[str] = None,
        use_continuous_pe: Optional[bool] = False,
        stride: Optional[bool] = 14,
        all_q_dims: Optional[bool] = False,
        patch_size: Optional[int] = 1,
    ):
        super().__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        ##### The following code may use internet connection/ model downloaded to torch hub directory
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name=name,
            model_type=model_type,
            is_eval=True,
            device=device,
        )

        if use_continuous_pe:
            model = patch_vit_resolution(model, stride=stride)
            self.extract_features = self.custom_extract_features
        else:
            self.extract_features = self.custom_extract_features

        self.model = model
        self.vis_processors = vis_processors
        self.txt_processors = txt_processors

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})

        if freeze_blip:
            for param in self.model.parameters():
                param.requires_grad = False
        self.freeze_blip = freeze_blip

        self.use_low_dim_feature = use_low_dim_feature
        self.use_masked_patch_wise_feature = use_masked_patch_wise_feature
        self.use_visual_encoder_only = use_visual_encoder_only
        self.mode = "image" if use_visual_encoder_only else "multimodal"
        self.append_global_features = append_global_features
        self.use_continuous_pe = use_continuous_pe
        self.all_q_dims = all_q_dims
        if isinstance(stride, int):
            stride = [stride] * 2
        self.stride = stride
        self.patch_size = patch_size
        
        # instantiate last layer
        if last_linear_layer is not None:
            last_linear_layer = nn.Conv2d(*last_linear_layer, 1, 1, 0, device=self.device)
        self.last_linear_layer = last_linear_layer
        
    def forward(self, x):
        super().forward(x)

        device = x["image"][0].device
        if self.get_model_device() != device: # avoid setting device as it takes some time (~1e-3s)
            self.model.eval()
            self.model.to(device)

        if self.use_masked_patch_wise_feature:
            if self.freeze_blip:
                with torch.no_grad():
                    raw_out = self.extract_features(x)
            else:
                raw_out = self.extract_features(x)
            features = raw_out.multimodal_embeds[:, 0] if not self.use_visual_encoder_only else raw_out.image_embeds[:, 0]

            if self.use_continuous_pe:
                def _to_size(_stride): # HACK
                    if _stride == 14:
                        return 224 // _stride
                    elif _stride == 7:
                        return 224 // _stride - 1
                    else:
                        raise ValueError
                fH, fW = _to_size(self.stride[0]), _to_size(self.stride[1])
            else:
                fH, fW = 16//self.patch_size, 16//self.patch_size # Lazy hack
            # assert features.shape[0] == self.time_seq * (fH * fW + 1)

            # extract global features
            global_features = features[::17] 
            final_indices = [i for i in range(features.size(0)) if (i + 1) % 17 != 0]
            final_features = features[final_indices]
            num_time_seq_images = features.shape[0] // 17
            out = final_features.view(1, fH * num_time_seq_images, fW, features.shape[-1])
            out = out.permute(0, 3, 1, 2)
            
            if self.append_global_features:
                out = torch.cat([out, global_features[None, :, None, None].repeat(out.shape[0], 1, out.shape[2], out.shape[3])], dim=1)
        
        else: # embed the entire image without masking patches
            # annoying rename issue I don't want to trace back to the source
            x = {"image": x["image"], "text_input": x["text"]}
            if self.freeze_blip:
                with torch.no_grad():
                    features = self.model.extract_features(x, mode=self.mode)
            else:
                features = self.model.extract_features(x, mode=self.mode)

            if self.use_visual_encoder_only:
                features = features.image_embeds if not self.use_low_dim_feature else features.image_embeds_proj
            else:
                features = features.multimodal_embeds

            if not self.all_q_dims:
                features = features[:, 0]
                out = features.view(1, 1, 1, features.shape[-1])
            else:
                # reappend the first query token to reach dimension of 36 (32 + 4, is a perfect square)
                # torch.Size([1, 32, d]) -> torch.Size([1, 36, d]) by repeating the first query token 4 times
                features = torch.cat([features[:, :1].repeat(1, 4, 1), features], dim=1)
                out = features.view(1, 6, 6, features.shape[-1])

            out = out.permute(0, 3, 1, 2)

        if self.last_linear_layer is not None:
            out = self.last_linear_layer(out)
            
        return out

    def get_model_device(self):
        if not hasattr(self, "_model_param"):
            self._model_param =  next(self.model.parameters())
        return self._model_param.device

    def set_backbone(self, lit_module):
        lit_module.backbone = self.model

    def custom_extract_features(self, sample):
        model = self.model

        image_set = sample["image"]
        image_embeds_results = []
        image_embed_proj_results = []
        multimodal_embeds_results = []

        for image in image_set:
            with model.maybe_autocast():
                image_embeds_frozen = model.ln_vision(model.visual_encoder(image))

            image_embeds_frozen = image_embeds_frozen.float().to(self.device)  # (1, 257, 1408)
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)  # (1, 257)
            query_tokens = model.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            ).to(self.device)  # (1, 32, 768)

            # NOTE: prepare input for leave-some-out patch feature
            n_patches_frozen = (1 + (224 - 14) // self.stride[0]) * (1 + (224 - 14) // self.stride[1])

            # don't mess with the stride as it is used to train the model
            assert image_embeds_frozen.shape[1] == (n_patches_frozen + 1)

            # instead increase the patch size through masking squares
            n_patches = n_patches_frozen // (self.patch_size**2)

            query_tokens_pp = query_tokens.repeat(n_patches + 1, 1, 1)
            image_embeds_frozen_pp = image_embeds_frozen.repeat(n_patches + 1, 1, 1)
            image_atts_pp = image_atts.repeat(n_patches + 1, 1, 1)

            # function format too lazy to refactor
            index_list = get_square_indices(np.sqrt(n_patches_frozen).astype(int), [self.patch_size])
            index_list = index_list[self.patch_size]

            assert len(index_list) == n_patches, f"Expected {n_patches} patches, got {len(index_list)}"

            # first patch is the global feature
            image_atts_pp[0, :] = 1
            # rest are patched by squares of size patch_size
            for i in range(n_patches):
                # compute indices corresponding to the square of size patch_size
                image_atts_pp[i+1, :] = 0
                image_atts_pp[i, :, index_list[i]] = 1

            # multimodal feature extraction
            if not self.use_visual_encoder_only:
                caption = sample["text"]
                # if caption is wrapped in a list, take the first element
                if isinstance(caption, list):
                    caption = caption[0]

                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                    self.device
                )
                text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                    self.device
                )
                att_mask = torch.cat([query_atts, text.attention_mask], dim=1).to(
                    self.device
                )

                att_mask_pp = att_mask.repeat(n_patches + 1, 1, 1)
                text_input_pp = text.input_ids.repeat(n_patches + 1, 1)

                query_output = model.Qformer.bert(
                    text_input_pp,
                    query_embeds=query_tokens_pp,
                    attention_mask=att_mask_pp,
                    encoder_hidden_states=image_embeds_frozen_pp,
                    encoder_attention_mask=image_atts_pp,
                    return_dict=True,
                )

                multimodal_embeds = query_output.last_hidden_state[:, : query_tokens.size(1), :]

                multimodal_embeds_results.append(multimodal_embeds)

            # visual feature extraction only
            else:
                query_output = model.Qformer.bert(
                    query_embeds=query_tokens_pp,
                    encoder_hidden_states=image_embeds_frozen_pp,
                    encoder_attention_mask=image_atts_pp,
                    return_dict=True,
                )

                image_embeds = query_output.last_hidden_state

                if self.use_low_dim_feature:
                    image_features = F.normalize(model.vision_proj(image_embeds), dim=-1)  # (1, 32, 256)
                    image_embeds_results.append(image_embeds)
                    image_embed_proj_results.append(image_features)
                else:
                    image_embeds_results.append(image_embeds)
        
        if not self.use_visual_encoder_only:
            multimodal_embeds_final = torch.cat(multimodal_embeds_results, dim=0)
            out = BlipOutputFeatures(
                    multimodal_embeds=multimodal_embeds_final,
                  )
        else:
            image_embeds_final = torch.cat(image_embeds_results, dim=0)
            if self.use_low_dim_feature:
                image_embed_proj_final = torch.cat(image_embed_proj_results, dim=0)
                out = BlipOutputFeatures(
                    image_embeds=image_embeds_final,
                    image_embeds_proj=image_embed_proj_final,
                )
            else:
                out = BlipOutputFeatures(
                        image_embeds=image_embeds_final,
                    )
        return out

    def extract_text_feat(self, texts, eval=True):
        assert isinstance(texts, list)
        for v in texts:
            assert isinstance(v, str)

        text_inputs = [self.txt_processors["eval"](v) for v in texts]
        sample = {"text_input": text_inputs}

        if eval:
            with torch.no_grad():
                feat = self.model.extract_features(sample, mode="text")
        else:
            feat = self.model.extract_features(sample, mode="text")
        feat = feat.text_embeds_proj[:, 0, :] # drop qformer-like embeddings

        return feat

    @property
    def intermediate_features(self):
        return self._intermediate_features

    @property
    def modalities(self):
        return ["image", "text", "image_raw"]

def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
    """
    Creates a method for position encoding interpolation.
    :param patch_size: patch size of the model.
    :param stride_hw: A tuple containing the new height and width stride respectively.
    :return: the interpolation method
    """

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        # compute number of tokens taking stride into account
        w0 = 1 + (w - patch_size) // stride_hw[1]
        h0 = 1 + (h - patch_size) // stride_hw[0]
        assert (
            w0 * h0 == npatch
        ), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                        stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
            align_corners=False,
            recompute_scale_factor=True,
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    return interpolate_pos_encoding


def patch_vit_resolution(model, stride):
    """
    change resolution of model output by changing the stride of the patch extraction.
    :param model: the model to change resolution for.
    :param stride: the new stride parameter.
    :return: the adjusted model
    """
    
    patch_size = model.visual_encoder.patch_embed.patch_size
    if stride == patch_size:  # nothing to do
        return model

    stride = nn_utils._pair(stride)
    if type(patch_size) is tuple: 
        assert (patch_size[0] == patch_size[1])
        patch_size = patch_size[0]
    assert all(
        [(patch_size // s_) * s_ == patch_size for s_ in stride]
    ), f"stride {stride} should divide patch_size {patch_size}"
    # fix the stride
    model.visual_encoder.patch_embed.proj.stride = stride
    # fix the positional encoding code
    model.visual_encoder.patched_func_stride_hw = stride # HACK: to hint non-local process to use patched function
    # model.visual_encoder.interpolate_pos_encoding = types.MethodType(
    #     _fix_pos_enc(patch_size, stride), model.visual_encoder
    # ) # NOTE: this won't work with multiprocessing; we directly include this function in eva_vit.py
    # model.visual_encoder.interpolate_pos_encoding = _fix_pos_enc(patch_size, stride) # cannot pickle
    return model
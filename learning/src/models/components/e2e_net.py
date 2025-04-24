from typing import List, Dict, Optional
from omegaconf import DictConfig # type: ignore
import torch
from .base import BaseNet
from .extractors.base import BaseExtractor
from .policies.base import BasePolicy
from .policies.stop_policies import select_stop_flagger


class E2ENet(BaseNet):
    def __init__(
        self,
        extractor: BaseExtractor,
        policy: BasePolicy,
        output_names: List[str],
        stop_flagger_cfg: Optional[DictConfig] = None
    ):
        super().__init__()

        self.extractor = extractor
        self.policy = policy
        self._output_names = output_names
        self.stop_flagger = None
        self.stop_only = False
        if stop_flagger_cfg is not None:
            in_dim = stop_flagger_cfg.in_dim
            hidden_dim = stop_flagger_cfg.hidden_dim
            self.stop_only = stop_flagger_cfg.stop_only
            self.stop_flagger = select_stop_flagger(stop_flagger_cfg.name)(in_dim, hidden_dim, cfg=stop_flagger_cfg)
            if not stop_flagger_cfg.get('step_remain_flag', False):
                self._output_names = self._output_names[:-1]
        if self.stop_only:
            for param in self.policy.parameters():
                param.requires_grad = False
            for param in self.extractor.last_linear_layer.parameters():
                param.requires_grad = False

    def get_obj_dir_text(self, texts: List[str], meta = None) -> Dict[str, List[str]]:
        # TODO More concrete usage of high level Planner here
        dirs, objs = [], []
        if meta is None: meta = texts[::]
        for i, text in enumerate(texts):
            parts = text.split('---')
            if len(parts) != 2: raise ValueError(f"Expected format 'dir---obj', but got: {text} {texts[i]} with meta {meta[i]} ")
            dirs.append(parts[0])
            objs.append(parts[1])
        return {'dir': dirs, 'obj': objs}

    def forward(self, x):
        txt_ = self.get_obj_dir_text(x['text'])
        x = {'image': x['image'], 'text':x['text']}
        if self.stop_only:
            with torch.no_grad():
                z = self.extractor({'image': x['image'], 'text': txt_['obj']})
                print(f"z: {z.shape}")
                return {
                    'vx': torch.tensor([0.0], device=x['image'].device),
                    'vy': torch.tensor([0.0], device=x['image'].device),
                    'vz': torch.tensor([0.0], device=x['image'].device),
                    'yaw': torch.tensor([0.0], device=x['image'].device),
                    'stop': torch.tensor([0.0], device=x['image'].device)
                }
                out = self.policy(z, txt_['dir'])
        else:
            z = self.extractor({'image': x['image'], 'text': txt_['obj']})
            out = self.policy(z, txt_['dir'])
        if self.stop_flagger is not None:
            stop = self.stop_flagger(self.policy.get_last_rep())
            out = torch.cat((out, stop), dim=1)

        out_dim = out.shape[-1]
        out = {k: out[...,i] for i, k in enumerate(self.output_names[:out_dim])}

        return out

    @property
    def modalities(self):
        return self.extractor.modalities
    
    @property
    def output_names(self):
        return self._output_names

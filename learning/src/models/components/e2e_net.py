from typing import List
import torch
from .base import BaseNet
from .extractors.base import BaseExtractor
from .policies.base import BasePolicy


class E2ENet(BaseNet):
    def __init__(
        self,
        extractor: BaseExtractor,
        policy: BasePolicy,
        output_names: List[str],
    ):
        super().__init__()

        self.extractor = extractor
        self.policy = policy
        self._output_names = output_names

    def forward(self, x):
        # if the policy is not lstm
        x = {'image': x['image'], 'text':x['text']}
        # print(x['image'].shape, x['image'].device, 'sanity_check_e2e')
        z = self.extractor(x)
        # print(torch.cuda.memory_summary(), 'sanity_test_e2e_cuda')
        out_action, out_stop = self.policy(z)

        out_dim = out_action.shape[-1]
        # assert out_dim == len(self.output_names), f"Model output of dim {out_dim} is not compatible with the target {self.output_names}"
        out_action = {k: out_action[...,i] for i, k in enumerate(self.output_names[:out_dim])}

        return out_action, out_stop

    @property
    def modalities(self):
        return self.extractor.modalities
    
    @property
    def output_names(self):
        return self._output_names

from typing import List

from src.models.components.base import BaseNet
from src.models.components.extractors.base import BaseExtractor
from src.models.components.policies.base import BasePolicy


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
        z = self.extractor(x)
        out = self.policy(z)

        out_dim = out.shape[-1]
        # assert out_dim == len(self.output_names), f"Model output of dim {out_dim} is not compatible with the target {self.output_names}"
        out = {k: out[...,i] for i, k in enumerate(self.output_names[:out_dim])}

        return out

    @property
    def modalities(self):
        return self.extractor.modalities
    
    @property
    def output_names(self):
        return self._output_names

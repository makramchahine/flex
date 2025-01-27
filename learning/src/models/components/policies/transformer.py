from typing import Optional
from omegaconf import DictConfig
from torch import nn

from src.models.components.policies.base import BasePolicy


class TransformerPolicy(BasePolicy):
    def __init__(
        self,
        model_type: Optional[str] = 'SimpleViT',
        cfg: Optional[DictConfig] = None,
        hidden_dim: Optional[int] = None,
        lstm_layers: int = 1
    ):
        super().__init__()
        
        if model_type == "SimpleViT": 
            from vit_pytorch import SimpleViT
            self.model = SimpleViT(
                image_size=tuple(cfg.image_size),
                patch_size=tuple(cfg.patch_size),
                num_classes=cfg.num_classes,
                dim=cfg.dim,
                depth=cfg.depth,
                heads=cfg.heads,
                mlp_dim=cfg.mlp_dim,
                channels=cfg.channels, # should be consistent with the feature extractor
                dim_head=cfg.dim_head,
            )
        elif model_type == "ViT":
            from vit_pytorch import ViT
            self.model = ViT(
                image_size=tuple(cfg.image_size),
                patch_size=tuple(cfg.patch_size),
                num_classes=cfg.num_classes,
                dim=cfg.dim,
                depth=cfg.depth,
                heads=cfg.heads,
                mlp_dim=cfg.mlp_dim,
                dropout=cfg.dropout,
                emb_dropout=cfg.emb_dropout,
                channels=cfg.channels, # should be consistent with the feature extractor
                dim_head=cfg.dim_head,
            )
        elif model_type == "DeepViT":
            from vit_pytorch.deepvit import DeepViT
            self.model = DeepViT(
                image_size=tuple(cfg.image_size),
                patch_size=tuple(cfg.patch_size),
                num_classes=cfg.num_classes,
                dim=cfg.dim,
                depth=cfg.depth,
                heads=cfg.heads,
                mlp_dim=cfg.mlp_dim,
                dropout=cfg.dropout,
                emb_dropout=cfg.emb_dropout,
                channels=cfg.channels, # should be consistent with the feature extractor
                dim_head=cfg.dim_head,
            )
        elif model_type == "CaiT":
            from vit_pytorch.cait import CaiT
            raise NotImplementedError(f"Need to handle custom channels other than 3")
        elif model_type == "T2TViT":
            from vit_pytorch.t2t import T2TViT
            t2t_layers = tuple([tuple(v) for v in cfg.t2t_layers])
            assert isinstance(cfg.image_size, int), "only allow square image now"
            self.model = T2TViT(
                dim=cfg.dim,
                image_size=cfg.image_size,
                depth=cfg.depth,
                heads=cfg.heads,
                mlp_dim=cfg.mlp_dim,
                num_classes=cfg.num_classes,
                channels=cfg.channels,
                t2t_layers=t2t_layers, # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
            )
        elif model_type == "CCT":
            from vit_pytorch.cct import CCT
            self.model = CCT(
                img_size=cfg.img_size,
                n_input_channels=cfg.n_input_channels,
                embedding_dim=cfg.embedding_dim,
                n_conv_layers=cfg.n_conv_layers,
                kernel_size=cfg.kernel_size,
                stride=cfg.stride,
                padding=cfg.padding,
                pooling_kernel_size=cfg.pooling_kernel_size,
                pooling_stride=cfg.pooling_stride,
                pooling_padding=cfg.pooling_padding,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                num_classes=cfg.num_classes,
                positional_embedding=cfg.positional_embedding, # ['sine', 'learnable', 'none']
            )
            # TODO: not working yet
        else:
            raise ValueError(f"Unrecognized model type {model_type}")

        self.model_type = model_type
        self.cfg = cfg
        self.lstm = None
        if hidden_dim is not None:
            self.lstm = nn.LSTM(
                input_size=cfg.num_classes,
                hidden_size=hidden_dim,
                num_layers=lstm_layers,
                batch_first=False          # Batch dim is actually sequence dim, one sequence at a time, can be changed later
            )
            self.hidden_state = None
            self.fc = nn.Linear(hidden_dim, cfg.num_classes)

    def forward(self, x):
        out = self.model(x)
        
        if self.lstm is not None:
            out, self.hidden_state = self.lstm(out, self.hidden_state)
            out = self.fc(out)
        
        return out

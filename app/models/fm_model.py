import torch
from torch import nn

from app.layers.fm import FactorizationMachine


class FMModule(nn.Module):
    """
    简单的 FM 模型
    """

    def __init__(
        self,
        sparse_shapes: dict[str, int],
        embedding_dim: int = 16,
    ):
        super().__init__()
        self.fm_layer = FactorizationMachine(sparse_shapes, embedding_dim)

    def forward(self, samples: dict[str, torch.Tensor]):
        fm_out = self.fm_layer.forward(samples)  # (batches,)
        return fm_out

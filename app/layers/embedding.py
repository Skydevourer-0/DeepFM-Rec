import torch
from torch import nn


class EmbeddingLayer(nn.Module):
    """
    嵌入层，用于将稀疏特征转换为嵌入向量
    """

    def __init__(
        self,
        feature_dims: dict[str, int],
        embedding_dim: int = 16,
        multi_feats: list[str] = [],
    ):
        super().__init__()
        self.embedding_layers = nn.ModuleDict()

        # 初始化嵌入层
        for feat, dim in feature_dims.items():
            if feat in multi_feats:
                # 对多值特征使用嵌入层 EmbeddingBag
                # EmbeddingBag 可以处理变长输入，适合多值特征
                self.embedding_layers[feat] = nn.EmbeddingBag(
                    dim, embedding_dim, sparse=True
                )
            else:
                # 对单值特征使用嵌入层 Embedding
                self.embedding_layers[feat] = nn.Embedding(dim, embedding_dim)

    def forward(self, samples: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        嵌入特征
        :param samples: 输入的特征数据，RecDataset.__getitem__ 返回的样本
        :return: 嵌入向量
        """
        embedded = {}
        for feat, layer in self.embedding_layers.items():
            if feat in samples:
                # 对多值特征使用 EmbeddingBag
                if isinstance(layer, nn.EmbeddingBag):
                    # 此时 samples[feat] 是 2D 张量，需要将其 flatten，并计算 offset
                    flats = samples[feat].flatten()
                    # 通过累加长度计算 offsets
                    lens = torch.tensor([0] + [len(v) for v in samples[feat]])
                    offsets = lens.cumsum(dim=0)[:-1]
                    # 使用 EmbeddingBag 处理变长输入
                    embedded[feat] = layer(flats, offsets)
                else:
                    embedded[feat] = layer(samples[feat])
        return embedded

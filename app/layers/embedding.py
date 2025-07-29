import torch
from torch import nn

from config import FEAT_NAMES


class EmbeddingLayer(nn.Module):
    """
    嵌入层，用于将稀疏特征转换为嵌入向量
    """

    def __init__(
        self,
        sparse_shapes: dict[str, int],
        embedding_dim: int = 16,
    ):
        super().__init__()
        self.embedding_layers = nn.ModuleDict()
        multi_feats = FEAT_NAMES["multi_sparse_feats"]
        dense_feats = FEAT_NAMES["dense_feats"]
        # 初始化嵌入层
        # 稀疏特征
        for feat, input_dim in sparse_shapes.items():
            if feat in multi_feats:
                # 对多值特征使用嵌入层 EmbeddingBag
                # EmbeddingBag 可以处理变长输入，适合多值特征
                self.embedding_layers[feat] = nn.EmbeddingBag(input_dim, embedding_dim)
            else:
                # 对单值特征使用嵌入层 Embedding
                self.embedding_layers[feat] = nn.Embedding(input_dim, embedding_dim)
        # 稠密特征
        for feat in dense_feats:
            self.embedding_layers[feat] = nn.Linear(1, embedding_dim, bias=False)

    def forward(self, samples: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        嵌入特征
        :param samples: 输入的特征数据，DataLoader 载入的样本集
        :return: 嵌入向量
        """
        embedded = {}
        for feat, layer in self.embedding_layers.items():
            if feat not in samples:
                continue
            val = samples[feat]
            # 对多值特征使用 EmbeddingBag
            if isinstance(layer, nn.EmbeddingBag):
                # 此时 val 是 (flats, offsets)
                flats, offsets = val
                # 使用 EmbeddingBag 处理变长输入
                embedded[feat] = layer(flats, offsets)
            else:
                embedded[feat] = layer(val)
        return embedded

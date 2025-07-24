import torch
from torch import nn


class EmbeddingLayer(nn.Module):
    """
    嵌入层，用于将稀疏特征转换为嵌入向量
    """

    def __init__(
        self,
        feature_dims: dict[str, int],
        multi_feats: list[str] = [],
        dense_feats: list[str] = [],
        embedding_dim: int = 16,
    ):
        super().__init__()
        self.embedding_layers = nn.ModuleDict()

        # 初始化嵌入层
        # 稀疏特征
        for feat, dim in feature_dims.items():
            if feat in multi_feats:
                # 对多值特征使用嵌入层 EmbeddingBag
                # EmbeddingBag 可以处理变长输入，适合多值特征
                self.embedding_layers[feat] = nn.EmbeddingBag(dim, embedding_dim)
            else:
                # 对单值特征使用嵌入层 Embedding
                self.embedding_layers[feat] = nn.Embedding(dim, embedding_dim)
        # 稠密特征
        for feat in dense_feats:
            self.embedding_layers[feat] = nn.Linear(1, embedding_dim, bias=False)

    def forward(self, samples: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        嵌入特征
        :param samples: 输入的特征数据，RecDataset.__getitem__ 返回的样本
        :return: 嵌入向量
        """
        embedded = {}
        for feat, layer in self.embedding_layers.items():
            if feat not in samples:
                continue
            val = samples[feat]
            # 对多值特征使用 EmbeddingBag
            if isinstance(layer, nn.EmbeddingBag):
                # 此时 val 是 2D 张量，其中存在非法值 -1
                # 需要过滤非法值并 flatten，计算 offset
                mask = val != -1  # (batches, max_len)
                flats = val[mask]
                # 通过累加长度计算 offsets
                valid_cnts = mask.sum(dim=1).tolist()  # (batches,)
                lens = torch.tensor([0] + valid_cnts, device=val.device)
                offsets = lens.cumsum(dim=0)[:-1]
                # 使用 EmbeddingBag 处理变长输入
                embedded[feat] = layer(flats, offsets)
            else:
                embedded[feat] = layer(val)
        return embedded

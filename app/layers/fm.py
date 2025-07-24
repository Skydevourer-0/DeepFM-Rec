import torch
from torch import nn

from app.layers.embedding import EmbeddingLayer


class FactorizationMachine(nn.Module):
    """
    因子分解机（FM），用于建模特征之间的二阶交互关系。
    公式：
        y = w_0 + sum(w_i * x_i) + sum(<v_i, v_j> * x_i * x_j)
    其中：
        - w_0：全局偏置
        - w_i：每个特征的一阶权重（可用 embedding_dim=1 的嵌入层实现）
        - x_i：输入特征的编码（通常为 LabelEncoder 编码后的索引）
        - v_i：每个特征的嵌入向量（embedding）
        - <v_i, v_j>：v_i 与 v_j 的内积，表示二阶特征交互
    实现说明：
        - 输入特征可直接使用编码后的索引，无需 one-hot。
        - 一阶项通过 embedding_dim=1 的嵌入层实现，等价于每个特征的权重。
        - 二阶项通过普通嵌入层获取特征向量，计算两两交互。
        - FM 能有效捕捉稀疏特征之间的隐式关系，适合推荐场景。
    """

    def __init__(
        self,
        feature_dims: dict[str, int],
        multi_feats: list[str] = [],
        dense_feats: list[str] = [],
        embedding_dim: int = 16,
    ):
        super().__init__()
        # 特征顺序
        self.feat_order = list(feature_dims.keys()) + dense_feats
        # 偏置值
        self.bias = nn.Parameter(torch.zeros(1))
        # 一阶项，embedding_dim 设为 1，即为线性映射
        self.linear_layers = EmbeddingLayer(
            feature_dims, multi_feats, dense_feats, embedding_dim=1
        )
        # 二阶项
        self.cross_layers = EmbeddingLayer(
            feature_dims, multi_feats, dense_feats, embedding_dim=embedding_dim
        )

    def forward(self, samples: dict[str, torch.Tensor]):
        # 一阶项
        linear_embeds = self.linear_layers.forward(samples)
        # (batches, 1, 1)
        linear_embeds = torch.stack(
            [linear_embeds[feat] for feat in self.feat_order], dim=1
        )
        # 一阶项求和
        first_order = linear_embeds.sum(dim=1).squeeze(-1)
        # 二阶项
        cross_embeds = self.cross_layers.forward(samples)
        # (batches, num_feats, embedding_dim)
        cross_embeds = torch.stack(
            [cross_embeds[feat] for feat in self.feat_order], dim=1
        )
        # 二阶项求和
        # 0.5 * (sum(vi) **2 - sum(vi **2))
        sum_square = cross_embeds.sum(dim=1) ** 2
        square_sum = (cross_embeds**2).sum(dim=1)
        second_order = 0.5 * (sum_square - square_sum).sum(dim=1)  # (batches,)

        return self.bias + first_order + second_order

import torch
from torch import nn

from app.layers.embedding import EmbeddingLayer
from config import FEAT_NAMES


class FactorizationMachine(nn.Module):
    """
    因子分解机 (FM)，用于建模特征之间的二阶交互关系。
    公式：
        y = w_0 + sum(w_i * x_i) + sum(<v_i, v_j> * x_i * x_j)
    其中：
        - w_0：全局偏置
        - w_i：每个特征的一阶权重 (可用 embedding_dim=1 的嵌入层实现)
        - x_i：输入特征的编码 (顺序的整数，可以作为 nn.Embedding 中的查表索引)
        - v_i：每个特征的嵌入向量 (embedding)
        - <v_i, v_j>：v_i 与 v_j 的内积，表示二阶特征交互
    实现说明：
        - 输入特征可直接使用编码后的索引，无需 one-hot。
        - 一阶项通过 embedding_dim=1 的嵌入层实现，等价于每个特征的权重。
        - 二阶项通过普通嵌入层获取特征向量，计算两两交互。
        - FM 能有效捕捉稀疏特征之间的隐式关系，适合推荐场景。
    """

    def __init__(
        self,
        sparse_shapes: dict[str, int],
        embedding_dim: int = 16,
    ):
        super().__init__()
        # 特征顺序
        self.feat_order = list(sparse_shapes.keys()) + FEAT_NAMES["dense_feats"]
        # 偏置值
        self.bias = nn.Parameter(torch.zeros(1))
        # 一阶项使用的嵌入向量，embedding_dim 设为 1，即为线性映射，所得结果为标量
        self.linear_layer = EmbeddingLayer(sparse_shapes, embedding_dim=1)
        # 二阶项使用的嵌入向量，embedding_dim 可调
        self.embed_layer = EmbeddingLayer(sparse_shapes, embedding_dim=embedding_dim)
        # 缓存堆叠嵌入向量
        self.cached_embeds = None

    def _stack_embeds(
        self, samples: dict[str, torch.Tensor], embed_layer: EmbeddingLayer
    ) -> torch.Tensor:
        """
        将嵌入向量堆叠成一个张量，便于后续计算
        :param samples: 输入样本，字典形式，包含稀疏特征和稠密特征
        :param embed_layer: 嵌入层实例，用于获取嵌入向量
        :return: 嵌入向量的堆叠结果, (batches, num_feats, embedding_dim)
        """
        # 获取嵌入向量, dict[str, (batches, embedding_dim)]
        embeds = embed_layer.forward(samples)
        # 堆叠嵌入向量, (batches, num_feats, embedding_dim)
        return torch.stack([embeds[feat] for feat in self.feat_order], dim=1)

    def get_cached_embeds(self) -> torch.Tensor:
        """
        获取缓存的堆叠嵌入向量
        :return: 缓存的堆叠嵌入向量
        """
        if self.cached_embeds is None:
            raise ValueError("不存在缓存的嵌入向量，请先调用 FM 层的 forward 方法。")
        return self.cached_embeds

    def forward(self, samples: dict[str, torch.Tensor]):
        # 一阶项, (batches, 1, 1)
        linear_embeds = self._stack_embeds(samples, self.linear_layer)
        # 一阶项求和
        first_order = linear_embeds.sum(dim=1).squeeze(-1)
        # 二阶项, (batches, num_feats, embedding_dim)
        cross_embeds = self._stack_embeds(samples, self.embed_layer)
        # 缓存堆叠嵌入向量以便复用
        self.cached_embeds = cross_embeds
        # 二阶项求和
        # 0.5 * (sum(vi) **2 - sum(vi **2))
        sum_square = cross_embeds.sum(dim=1) ** 2
        square_sum = (cross_embeds**2).sum(dim=1)
        second_order = 0.5 * (sum_square - square_sum).sum(dim=1)  # (batches,)

        return self.bias + first_order + second_order

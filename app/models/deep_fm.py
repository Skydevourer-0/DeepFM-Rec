import torch
from torch import nn
from app.layers.fm import FactorizationMachine
from app.layers.dnn import DNNLayer


class DeepFM(nn.Module):
    """
    深度因子分解机 (DeepFM)，结合了 FM 和 DNN 的特性.
    通过 FM 捕捉特征的二阶交互关系，通过 DNN 捕捉高阶特征交叉。
    """

    def __init__(
        self,
        sparse_shapes: dict[str, int],
        embedding_dim: int = 16,
        hidden_units: list[int] = [128, 64, 32],
    ) -> None:
        """
        :param sparse_shapes: 稀疏特征的形状，字典形式
        :param embedding_dim: 嵌入向量的维度，默认为 16
        :param hidden_units: DNN 的隐藏层单元数，列表形式，默认为 [128, 64, 32]
        """
        super().__init__()
        # 初始化 FM 层
        self.fm_layer = FactorizationMachine(sparse_shapes, embedding_dim)
        # 将嵌入向量堆叠后输入 DNN，因此输入维度为 特征数 * 嵌入向量维度
        input_dim = len(self.fm_layer.feat_order) * embedding_dim
        # 初始化 DNN 层，使用三层感知机
        self.dnn_layer = DNNLayer(input_dim, hidden_units)
        # 初始化输出层，输出维度为 1，表示最终的预测结果
        self.output_layer = nn.Linear(hidden_units[-1], 1)

    def forward(self, samples: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播函数，计算 DeepFM 的输出
        :param samples: 输入样本，字典形式，包含稀疏特征和稠密特征
        :return: 模型输出，形状为 (batches, 1)
        """
        # 计算 FM 输出
        fm_out = self.fm_layer(samples)
        # 获取堆叠嵌入向量, (batches, num_feats, embedding_dim)
        embed_out = self.fm_layer.get_cached_embeds()
        # 将嵌入向量展开为 (batches, num_feats * embedding_dim)
        # torch.stack 得到的张量必定连续，因此可以使用 view 安全展开
        embed_out = embed_out.view(embed_out.size(0), -1)
        # 通过 DNN 层进行非线性变换
        dnn_out = self.dnn_layer(embed_out)
        # 通过输出层得到最终预测结果, (batches, 1)
        output = self.output_layer(dnn_out)
        # 将 FM 输出和 DNN 输出相加，得到最终结果
        result = fm_out + output.squeeze(-1)  # (batches,)
        # 清空缓存，防止下次复用旧值
        self.fm_layer.cached_embeds = None

        return result

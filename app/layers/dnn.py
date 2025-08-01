import torch
import torch.nn as nn


class DNNLayer(nn.Module):
    """
    深度神经网络层，通过多层堆叠的全连接层实现特征的非线性变换，模拟高阶特征交叉。
    每层使用 ReLU 激活函数，并可选地添加 Dropout 以防止过拟合。
    """

    def __init__(self, input_dim: int, hidden_units: list[int], dropout: float = 0.5):
        """
        :param input_dim: 输入特征的维度
        :param hidden_units: 每层的隐藏单元数，列表形式
        :param dropout: Dropout 概率，默认为 0.5
        """
        super().__init__()
        # 初始化多层全连接网络
        layers = []
        for hidden_dim in hidden_units:
            # 线性映射，即全连接层，将输入维度映射到隐藏层维度
            layers.append(nn.Linear(input_dim, hidden_dim))
            # ReLU 激活函数，增加非线性，实现特征的非线性变换，模拟高阶特征交叉
            layers.append(nn.ReLU())
            # Dropout 层，防止过拟合
            layers.append(nn.Dropout(dropout))
            # 下一层的输入维度就是这一层的输出维度
            input_dim = hidden_dim

        # 将所有层堆叠成顺序容器，即 多层感知机
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.mlp(x)

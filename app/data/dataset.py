import pandas as pd
import torch
from torch.utils.data import Dataset


class RecDataset(Dataset):
    """
    将数据集转换为 PyTorch Dataset 格式
    """

    def __init__(
        self,
        df: pd.DataFrame,
        multi_sparse: dict[str, torch.Tensor],
        multi_feats: list[str] = [],
        dense_feats: list[str] = [],
    ):
        self.df = df
        self.multi_sparse = multi_sparse
        self.sparse_feats = df.columns.difference(
            dense_feats + multi_feats + ["label"]
        ).tolist()
        self.dense_feats = dense_feats
        self.labels = df["label"].astype("float32").values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = {}
        # 稀疏特征
        for feat in self.sparse_feats:
            sample[feat] = torch.tensor(self.df.iloc[idx][feat], dtype=torch.int64)
        # 多值稀疏特征
        for feat, vals in self.multi_sparse.items():
            sample[feat] = vals[idx]
        # 数值（稠密）特征
        # 稠密特征的 Embedding 层是 nn.Linear，需要输入 2D 张量
        # 因此特征值需要保存为 1D 张量，才能在 batch 的堆叠下成为 2D
        for feat in self.dense_feats:
            sample[feat] = torch.tensor([self.df.iloc[idx][feat]], dtype=torch.float32)
        # 标签
        sample["label"] = torch.tensor(self.df.iloc[idx]["label"], dtype=torch.float32)

        return sample

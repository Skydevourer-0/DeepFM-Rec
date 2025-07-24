import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class RecDataset(Dataset):
    """
    将数据集转换为 PyTorch Dataset 格式
    """

    def __init__(
        self,
        df: pd.DataFrame,
        multi_sparse: dict[str, np.ndarray],
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
            valid_vals = vals[idx][vals[idx] != -1]
            sample[feat] = torch.tensor(valid_vals, dtype=torch.int64)
        # 数值（稠密）特征
        for feat in self.dense_feats:
            sample[feat] = torch.tensor(self.df.iloc[idx][feat], dtype=torch.float32)
        # 标签
        sample["label"] = torch.tensor(self.df.iloc[idx]["label"], dtype=torch.float32)

        return sample

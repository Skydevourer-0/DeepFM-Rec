import torch
from torch.utils.data import Dataset

from app.utils.types import RaggedMemmap
from config import FEAT_NAMES


class RecDataset(Dataset):
    """
    将数据集转换为 PyTorch Dataset 格式
    """

    def __init__(
        self,
        # 通过 np.memmap 载入的特征编码，能够按需载入，避免内存压力过大
        encoded_feats: dict[str, RaggedMemmap],
        n_samples: int,
    ):
        self.encoded = encoded_feats
        self.length = n_samples
        # 稠密特征列
        self.dense_feats = FEAT_NAMES["dense_feats"]
        # 稀疏特征列
        self.sparse_feats = FEAT_NAMES["sparse_feats"]
        # 多值稀疏特征列
        self.multi_feats = FEAT_NAMES["multi_sparse_feats"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = {}
        # 单值稀疏特征
        for feat in self.sparse_feats:
            sample[feat] = torch.tensor(self.encoded[feat][idx], dtype=torch.int32)
        # 多值稀疏特征
        for feat in self.multi_feats:
            # 此时，encpded[feat] 为 (flats, offsets, indices)
            flats, offsets, indices = self.encoded[feat]
            _idx = indices[idx]
            cur_values = [] if _idx == -1 else flats[offsets[_idx] : offsets[_idx + 1]]
            sample[feat] = torch.tensor(cur_values, dtype=torch.int32)
        # 数值（稠密）特征
        # 稠密特征的 Embedding 层是 nn.Linear，需要输入 2D 张量
        # 因此特征值需要保存为 1D 张量，才能在 batch 的堆叠下成为 2D
        for feat in self.dense_feats:
            sample[feat] = torch.tensor([self.encoded[feat][idx]], dtype=torch.float32)
        # 标签
        sample["label"] = torch.tensor(self.encoded["label"][idx], dtype=torch.float32)

        return sample

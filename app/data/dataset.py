from functools import partial
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from config import FEAT_NAMES
from app.utils.types import RaggedMemmap


class RecDataset(Dataset):
    """
    将数据集转换为 PyTorch Dataset 格式
    """

    def __init__(
        self,
        # 通过 np.memmap 载入的特征编码，能够按需载入，避免内存压力过大
        encoded_feats: dict[str, RaggedMemmap],
        n_samples: Optional[int] = None,
        indices: Optional[torch.Tensor] = None,
    ):
        self.encoded = encoded_feats
        self.indices = self._load_indices(indices, n_samples)
        # 稠密特征列
        self.dense_feats = FEAT_NAMES["dense_feats"]
        # 稀疏特征列
        self.sparse_feats = FEAT_NAMES["sparse_feats"]
        # 多值稀疏特征列
        self.multi_feats = FEAT_NAMES["multi_sparse_feats"]

    def _load_indices(self, indices, n_samples):
        assert not (
            indices is None and n_samples is None
        ), "Either indices or n_samples must be provided."
        return indices if n_samples is None else torch.arange(n_samples)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 根据 indices 计算真实索引
        idx = self.indices[idx]
        encoded = self.encoded
        sample = {}
        # 单值稀疏特征
        for feat in self.sparse_feats:
            sample[feat] = torch.tensor(encoded[feat][idx], dtype=torch.int32)
        # 多值稀疏特征
        for feat in self.multi_feats:
            # 此时，encpded[feat] 为 (flats, offsets, indices)
            flats, offsets, indices = encoded[feat]
            _idx = indices[idx]
            cur_values = [] if _idx == -1 else flats[offsets[_idx] : offsets[_idx + 1]]
            sample[feat] = torch.tensor(cur_values, dtype=torch.int32)
        # 数值（稠密）特征
        # 稠密特征的 Embedding 层是 nn.Linear，需要输入 2D 张量
        # 因此特征值需要保存为 1D 张量，才能在 batch 的堆叠下成为 2D
        for feat in self.dense_feats:
            sample[feat] = torch.tensor([encoded[feat][idx]], dtype=torch.float32)
        # 标签
        sample["label"] = torch.tensor(encoded["label"][idx], dtype=torch.float32)

        return sample

    @staticmethod
    def _collate_fn(batch: list[dict[str, torch.Tensor]], multi_feats: list[str] = []):
        """
        DataLoader 调用的堆叠方法，对多值稀疏特征进行特殊处理
        """
        if not batch:
            return None
        collated = {}
        for feat in batch[0].keys():
            values = [sample[feat] for sample in batch]
            # 多值稀疏特征
            if feat in multi_feats:
                # 变长列表无法 stack, 存储为 values 和 offsets 两个张量
                # 拼接所有值，将变长列表展开为 1D 张量
                flats = torch.cat(values)
                # 累加长度计算偏移量
                lens = [0] + [len(x) for x in values]
                offsets = torch.tensor(lens, dtype=torch.int32).cumsum(dim=0)[:-1]
                # 输出堆叠结果
                collated[feat] = (flats, offsets)
            else:
                collated[feat] = torch.stack(values)
        return collated

    @staticmethod
    def to_device(batch, device):
        """将张量移动到设备中，便于 gpu 优化"""
        to_device = lambda x: x.to(device, non_blocking=True)
        labels = to_device(batch["label"])
        samples = {}
        for feat, val in batch.items():
            if feat == "label":
                continue
            if isinstance(val, torch.Tensor):
                samples[feat] = to_device(val)
            elif isinstance(val, (tuple, list)):
                samples[feat] = tuple(map(to_device, val))
        return labels, samples

    def split(
        self, train_size=0.7, valid_size=0.1, batch_size=1024, shuffle=True, seed=42
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        将数据集分割为训练集，验证集和测试集

        :param train_size: 训练集占比
        :param valid_size: 验证集占比
        :return: 训练集，验证集和测试集的特征数据及多值稀疏特征矩阵
        """
        # 确定各个输出集合的大小
        n_total = len(self)
        n_train = int(n_total * train_size)
        n_val = int(n_total * valid_size)
        # 构造数据集索引
        indices = torch.arange(n_total)
        if shuffle:
            rand_generator = torch.Generator().manual_seed(seed)
            indices = indices[torch.randperm(n_total, generator=rand_generator)]
        # 分割数据集索引
        train_indices = indices[:n_train]
        valid_indices = indices[n_train : n_train + n_val]
        test_indices = indices[n_train + n_val :]
        # 根据索引构造数据集
        train_dataset = RecDataset(self.encoded, indices=train_indices)
        valid_dataset = RecDataset(self.encoded, indices=valid_indices)
        test_dataset = RecDataset(self.encoded, indices=test_indices)
        # 构造加载器
        nparams = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "pin_memory": True,
            "collate_fn": partial(self._collate_fn, multi_feats=self.multi_feats),
        }
        train_loader = DataLoader(train_dataset, **nparams)
        valid_loader = DataLoader(valid_dataset, **nparams)
        test_loader = DataLoader(test_dataset, **nparams)

        return train_loader, valid_loader, test_loader

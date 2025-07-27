from functools import partial
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset


class RecDataset(Dataset):
    """
    将数据集转换为 PyTorch Dataset 格式
    """

    def __init__(
        self,
        # 多值稀疏特征编码存储为变长列表，其余特征编码存储为张量
        encoded_feats: dict[str, torch.Tensor | list[list[str]]],
        indices: Optional[torch.Tensor] = None,
        dense_feats: list[str] = [],
    ):
        self.encoded = encoded_feats
        # 特征编码字典的每个 value 的长度均为 样本数
        n_samples = self._infer_length()
        self.indices = indices if indices is not None else torch.arange(n_samples)
        # 构造稀疏特征列表
        self.sparse_feats = [
            k for k in encoded_feats.keys() if k not in (dense_feats + ["label"])
        ]
        # 稠密特征
        self.dense_feats = dense_feats

    def _infer_length(self):
        for feat_vals in self.encoded.values():
            if isinstance(feat_vals, torch.Tensor):
                return feat_vals.size(0)
        return 0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 根据 indices 计算真实索引
        idx = self.indices[idx]
        encoded = self.encoded
        sample = {}
        # 稀疏特征
        for feat in self.sparse_feats:
            sample[feat] = torch.as_tensor(encoded[feat][idx], dtype=torch.int64)
        # 数值（稠密）特征
        # 稠密特征的 Embedding 层是 nn.Linear，需要输入 2D 张量
        # 因此特征值需要保存为 1D 张量，才能在 batch 的堆叠下成为 2D
        for feat in self.dense_feats:
            sample[feat] = torch.as_tensor([encoded[feat][idx]], dtype=torch.float32)
        # 标签
        sample["label"] = torch.as_tensor(encoded["label"][idx], dtype=torch.float32)

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
                offsets = torch.tensor(lens, dtype=torch.int64).cumsum(dim=0)[:-1]
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
        self, train_size=0.7, val_size=0.1, batch_size=1024, shuffle=True, seed=42
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        将数据集分割为训练集，验证集和测试集

        :param train_size: 训练集占比
        :param val_size: 验证集占比
        :return: 训练集，验证集和测试集的特征数据及多值稀疏特征矩阵
        """
        # 确定各个输出集合的大小
        n_total = len(self)
        n_train = int(n_total * train_size)
        n_val = int(n_total * val_size)
        # 构造数据集索引
        indices = torch.arange(n_total)
        if shuffle:
            rand_generator = torch.Generator().manual_seed(seed)
            indices = indices[torch.randperm(n_total, generator=rand_generator)]
        # 分割数据集索引
        train_indices = indices[:n_train]
        val_indices = indices[n_train : n_train + n_val]
        test_indices = indices[n_train + n_val :]
        # 定义多值稀疏特征列和稠密特征列
        multi_feats = ["genres", "tag"]
        dense_feats = ["age"]
        # 根据索引构造数据集
        train_dataset = RecDataset(self.encoded, train_indices, dense_feats)
        val_dataset = RecDataset(self.encoded, val_indices, dense_feats)
        test_dataset = RecDataset(self.encoded, test_indices, dense_feats)
        # 构造加载器
        nparams = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "pin_memory": True,
            "collate_fn": partial(self._collate_fn, multi_feats=multi_feats),
        }
        train_loader = DataLoader(train_dataset, **nparams)
        val_loader = DataLoader(val_dataset, **nparams)
        test_loader = DataLoader(test_dataset, **nparams)

        return train_loader, val_loader, test_loader

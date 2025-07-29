from functools import partial

import torch
from torch.utils.data import DataLoader, Subset

from app.data.dataset import RecDataset
from app.utils.types import RaggedMemmap
from config import FEAT_NAMES


class RecDataManager:
    def __init__(self, encoded: dict[str, RaggedMemmap], n_samples: int):
        self.dataset = RecDataset(encoded, n_samples=n_samples)
        self.multi_feats = FEAT_NAMES["multi_sparse_feats"]
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

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
        """
        将张量移动到设备中，便于 gpu 优化
        """

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
        self,
        train_size=0.7,
        valid_size=0.1,
        batch_size=1024,
        num_workers=0,
        pin_memory=False,
        seed=42,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        将数据集分割为训练集，验证集和测试集

        :param train_size: 训练集占比
        :param valid_size: 验证集占比
        :return: 训练集，验证集和测试集的特征数据及多值稀疏特征矩阵
        """
        # 确定各个输出集合的大小
        n_total = len(self.dataset)
        n_train = int(n_total * train_size)
        n_val = int(n_total * valid_size)
        # 构造数据集索引
        indices = torch.arange(n_total)
        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n_total, generator=rng)
        # 分割数据集索引
        train_indices = indices[:n_train].tolist()
        valid_indices = indices[n_train : n_train + n_val].tolist()
        test_indices = indices[n_train + n_val :].tolist()
        # 根据索引构造数据集
        # Subset 自动支持索引切分，不需要在 RecDataset 中手动实现
        train_dataset = Subset(self.dataset, indices=train_indices)
        valid_dataset = Subset(self.dataset, indices=valid_indices)
        test_dataset = Subset(self.dataset, indices=test_indices)
        # 构造加载器
        nparams = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "collate_fn": partial(
                RecDataManager._collate_fn, multi_feats=self.multi_feats
            ),
        }
        train_loader = DataLoader(train_dataset, shuffle=True, **nparams)
        valid_loader = DataLoader(valid_dataset, shuffle=False, **nparams)
        test_loader = DataLoader(test_dataset, shuffle=False, **nparams)

        return train_loader, valid_loader, test_loader

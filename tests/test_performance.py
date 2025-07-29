from pathlib import Path

import pytest
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader

from app.data.manager import RecDataManager
from app.data.preprocessor import DataPreprocessor
from app.models.fm_model import FMModule
from app.train import Trainer


# 设置模块级别，数据加载和训练器构造只需执行一次
@pytest.fixture(scope="module")
def data_manager():
    data_path = Path("resources/data")
    encoded_path = data_path / "encoded"

    preprocessor = DataPreprocessor(data_path / "raw")
    encoded_feats, n_samples, sparse_shapes = preprocessor.load(encoded_path)
    manager = RecDataManager(encoded_feats, n_samples=n_samples)
    train_loader, valid_loader, _ = manager.split(
        num_workers=4, pin_memory=True, train_size=0.1
    )

    return train_loader, valid_loader, sparse_shapes


@pytest.fixture(scope="module")
def trainer(data_manager):
    # 使用 fixture 自动注入
    train_loader, valid_loader, sparse_shapes = data_manager
    multi_feats = ["genres", "tag"]
    dense_feats = ["age"]
    fm_model = FMModule(sparse_shapes, multi_feats, dense_feats)
    trainer = Trainer(fm_model, train_loader, valid_loader)

    return trainer


@pytest.fixture(scope="module")
def data_loader(data_manager):
    train_loader, _, _ = data_manager
    return train_loader


def test_performance_one_epoch(trainer: Trainer, data_loader: DataLoader):
    """
    测试训练器单个 epoch 性能瓶颈。
    只跑一个 epoch，内部用 autograd profiler 自动采样。

    :param trainer: Trainer 实例 (封装模型训练逻辑)
    :param data_loader: DataLoader 实例
    """

    with profile(
        activities=[
            ProfilerActivity.CPU,
            (
                ProfilerActivity.CUDA
                if torch.cuda.is_available()
                else ProfilerActivity.CPU
            ),
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        # 重点代码区域：训练一轮
        with record_function("train_one_epoch"):
            loss, mae = trainer._one_epoch(data_loader, training=True)
            print(f"Loss: {loss:.4f}, MAE: {mae:.4f}")

    # 你还可以导出成 chrome trace json，方便用 chrome://tracing 查看
    prof.export_chrome_trace("trace.json")

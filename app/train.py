import copy
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
from loguru import logger
from sklearn.metrics import mean_absolute_error
from torch.nn import Module, functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.data.dataset import RecDataset
from app.utils.utils import EarlyStopping, min_max_scale_tensor


class TrainMetrics:
    """训练过程中的指标收集和绘图工具"""

    def __init__(
        self,
        train_losses: Optional[list[float]] = None,
        train_maes: Optional[list[float]] = None,
        valid_losses: Optional[list[float]] = None,
        valid_maes: Optional[list[float]] = None,
    ):
        self.train_losses = train_losses or []
        self.train_maes = train_maes or []
        self.valid_losses = valid_losses or []
        self.valid_maes = valid_maes or []

    def draw(self, save_path: Optional[str | Path] = None):
        """绘制指标曲线"""
        epochs = range(1, len(self.train_losses) + 1)

        # 根据已有指标动态确定子图数量
        metrics_to_plot = {
            "Train Loss": self.train_losses,
            "Train MAE": self.train_maes,
            "Valid Loss": self.valid_losses,
            "Valid MAE": self.valid_maes,
        }
        # 过滤掉空指标
        metrics_to_plot = {k: v for k, v in metrics_to_plot.items() if v}

        n_plots = len(metrics_to_plot)
        if n_plots == 0:
            logger.warning("没有可绘制的指标数据")
            return

        plt.figure(figsize=(6 * n_plots, 5))

        for i, (name, values) in enumerate(metrics_to_plot.items(), 1):
            plt.subplot(1, n_plots, i)
            plt.plot(epochs, values, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel(name)
            plt.title(f"{name} Curve")
            plt.grid(True)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            logger.info(f"指标图已保存至: {save_path}")

        plt.show()


class Trainer:
    """训练模型的通用类"""

    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader],
        device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ) -> None:
        """
        :param model: PyTorch 模型，继承自 torch.nn.Module
        :param train_loader: 训练集 DataLoader，提供训练数据批次
        :param valid_loader: 验证集 DataLoader，提供验证数据批次（可选）
        :param device: 设备，默认为 CPU，可传入 GPU 设备

        :return: None，无返回值
        """
        # 将模型移到设备中
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.best_model_state = None
        # 早停器，避免过拟化
        self.early_stopping = EarlyStopping(on_continue=self._save_best_model)
        # 使用 Adam 优化器进行梯度下降
        self.optimizer = torch.optim.Adam(model.parameters())
        # 学习率调度器，当指标多轮未优化时令学习率下降
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, threshold=1e-5
        )

    def _save_best_model(self):
        self.best_model_state = copy.deepcopy(self.model.state_dict())

    def _one_epoch(self, dataloader: DataLoader, training: bool = True):
        """一轮内进行梯度下降或指标计算"""
        # 调用 训练/验证 方法
        if training:
            self.model.train()
        else:
            self.model.eval()

        epoch_loss = 0
        preds, targets = [], []

        for batch in tqdm(
            dataloader, desc="Training" if training else "Validate", ncols=100
        ):
            # 将张量移动到设备中，便于 gpu 优化
            labels, samples = RecDataset.to_device(batch, self.device)
            # 梯度置零
            if training:
                self.optimizer.zero_grad()
            # 根据参数设置是否计算梯度
            with torch.set_grad_enabled(training):
                # __call__() 自动调用 forward() 等方法
                logits = self.model(samples)
                # 模型输出结果需要 min-max 归一化，与数据预处理中标签的处理一致
                logits = min_max_scale_tensor(logits)
                # 回归模型，采用 MSE (均方误差)
                loss = functional.mse_loss(logits, labels)
                if training:
                    # 反向传播
                    loss.backward()
                    # 优化器进行梯度下降
                    self.optimizer.step()
            # 记录当前 batch 损失
            epoch_loss += loss.item()
            # 记录当前 batch 预测值和真实值
            # 使用 detach 断开 logits 与计算图的链接
            preds.append(logits.detach())
            targets.append(labels)

        # 计算平均损失
        avg_loss = epoch_loss / len(dataloader)
        # 计算 MAE(平均绝对误差)指标
        mae_score = 0.0
        preds = torch.cat(preds).flatten()
        targets = torch.cat(targets).flatten()
        mae_score = mean_absolute_error(targets.cpu().numpy(), preds.cpu().numpy())

        return avg_loss, mae_score

    def evaluate(self, dataloader: DataLoader, model_state=None):
        """评估模型指标"""
        # 载入模型参数
        if model_state is not None:
            self.model.load_state_dict(model_state)
        loss, mae = self._one_epoch(dataloader, training=False)
        return loss, mae

    def load_model(self, path: Path):
        model_state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(model_state)

    def save_model(self, path: Path, model_state=None):
        if model_state is None:
            model_state = self.model.state_dict()
        torch.save(model_state, path)

    def train(self, epochs: int):
        """训练模型"""
        train_losses, train_maes = [], []
        valid_losses, valid_maes = [], []

        # 开始迭代
        for epoch in range(epochs):
            start_time = time.perf_counter()
            loss, mae = self._one_epoch(self.train_loader, training=True)
            train_losses.append(loss)
            train_maes.append(mae)
            metric_str = f"Epoch {epoch+1}/{epochs} | Train Loss: {loss:.4f} | Train MAE: {mae:.4f}"

            if self.valid_loader:
                loss, mae = self._one_epoch(self.valid_loader, training=False)
                valid_losses.append(loss)
                valid_maes.append(mae)
                # 传入指标，更新学习率
                self.scheduler.step(mae)
                # 更新早停器
                if self.early_stopping.step(mae):
                    logger.info(f"避免过拟合，在第 {epoch} 轮早停.")
                    break

                metric_str += f" | Valid Loss: {loss:.4f} | Valid MAE: {mae:.4f}"
            elapsed = time.perf_counter() - start_time
            metric_str += f" | Cost: {elapsed:.2f} s"
            # 打印当前轮次的指标
            logger.info(metric_str)

        if self.best_model_state is None:
            self._save_best_model()
        metrics = TrainMetrics(train_losses, valid_maes, valid_losses, valid_maes)
        return self.best_model_state, metrics

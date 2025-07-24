import copy
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from sklearn.metrics import roc_auc_score
from torch.nn import Module, functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.utils import EarlyStopping


class TrainMetrics:
    """训练结果"""

    def __init__(self, train_losses, val_losses=[], val_aucs=[]):
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.val_aucs = val_aucs

    def draw(self, save_path: Optional[str | Path] = None):
        """绘制指标曲线"""
        train_losses = self.train_losses
        val_losses = self.val_losses
        val_aucs = self.val_aucs

        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(12, 5))

        # Loss 图
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label="Train Loss")
        if val_losses:
            plt.plot(epochs, val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Binary Cross Entropy Loss Curve")
        plt.legend()

        # AUC 图
        if val_aucs:
            plt.subplot(1, 2, 2)
            plt.plot(epochs, val_aucs, label="Val AUC", color="orange")
            plt.xlabel("Epoch")
            plt.ylabel("AUC")
            plt.title("AUC Score Curve")
            plt.legend()

        plt.tight_layout()

        # 保存图片（如果提供了路径）
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
        val_loader: Optional[DataLoader],
        device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ) -> None:
        """
        :param model: PyTorch 模型，继承自 torch.nn.Module
        :param train_loader: 训练集 DataLoader，提供训练数据批次
        :param val_loader: 验证集 DataLoader，提供验证数据批次（可选）
        :param device: 设备，默认为 CPU，可传入 GPU 设备

        :return: None，无返回值
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.best_model_state = None
        # 早停器，避免过拟化
        self.early_stopping = EarlyStopping(on_continue=self._save_best_model)
        # 使用 Adam 优化器进行梯度下降
        self.optimizer = torch.optim.Adam(model.parameters())
        # 学习率调度器，当指标多轮未优化时令学习率下降
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5, threshold=1e-5
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
            labels = batch["label"].float().to(self.device)
            samples = {f: t.to(self.device) for f, t in batch.items() if f != "label"}
            # 梯度置零
            if training:
                self.optimizer.zero_grad()
            # 根据参数设置是否计算梯度
            with torch.set_grad_enabled(training):
                # __call__() 自动调用 forward() 等方法
                logits = self.model(samples)
                # 计算 BCE(二分类交叉熵损失)
                loss = functional.binary_cross_entropy_with_logits(logits, labels)
                if training:
                    # 反向传播
                    loss.backward()
                    # 优化器进行梯度下降
                    self.optimizer.step()
            # 记录当前 batch 损失
            epoch_loss += loss.item()
            # 记录当前 batch 预测值和真实值
            # 使用 detach 断开 logits 与计算图的链接
            preds.append(logits.detach().cpu().numpy())
            targets.append(labels.cpu().numpy())

        # 计算平均损失
        avg_loss = epoch_loss / len(dataloader)
        auc_score = 0.0
        # 仅当验证时计算 AUC 分数
        if not training:
            # 拼接预测值和真实值，并 ravel 展开为 1D
            preds = np.concatenate(preds).ravel()
            targets = np.concatenate(targets).ravel()
            # 计算 AUC 分数
            auc_score = roc_auc_score(targets, preds)

        return avg_loss, auc_score

    def evaluate(self, dataloader: DataLoader, model_state=None):
        """评估模型指标"""
        # 载入模型参数
        if model_state is not None:
            self.model.load_state_dict(model_state)
        loss, auc = self._one_epoch(dataloader, training=False)
        return loss, auc

    def load_model(self, path: Path):
        model_state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(model_state)

    def save_model(self, path: Path, model_state=None):
        if model_state is None:
            model_state = self.model.state_dict()
        torch.save(model_state, path)

    def train(self, epochs: int):
        """训练模型"""
        # 将模型移到设备中
        self.model.to(self.device)
        train_losses, val_losses, val_aucs = [], [], []

        # 开始迭代
        for epoch in range(epochs):
            start_time = time.perf_counter()
            loss, _ = self._one_epoch(self.train_loader, training=True)
            train_losses.append(loss)
            metric_str = f"Epoch {epoch+1}/{epochs} | Train Loss: {loss:.4f}"

            if self.val_loader:
                loss, auc_score = self._one_epoch(self.val_loader, training=False)
                val_losses.append(loss)
                val_aucs.append(auc_score)
                # 传入指标，更新学习率
                self.scheduler.step(auc_score)
                # 更新早停器
                if self.early_stopping.step(auc_score):
                    logger.info(f"避免过拟合，在第 {epoch} 轮早停.")
                    break

                metric_str += f" | Val Loss: {loss:.4f} | Val AUC: {auc_score:.4f}"
            elapsed = time.perf_counter() - start_time
            metric_str += f" | Cost: {elapsed:.2f} s"
            # 打印当前轮次的指标
            logger.info(metric_str)

        if self.best_model_state is None:
            self._save_best_model()
        metrics = TrainMetrics(train_losses, val_losses, val_aucs)
        return self.best_model_state, metrics

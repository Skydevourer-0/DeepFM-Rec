from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch


def min_max_scale_tensor(x, min_val=None, max_val=None, eps=1e-8) -> torch.Tensor:
    """
    min-max 归一化，支持 pd.Series, np.ndarray, torch.Tensor
    """

    if isinstance(x, pd.Series):
        x = torch.tensor(x.values, dtype=torch.float32)
    elif isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    if min_val is None:
        min_val = x.min(dim=0, keepdim=True).values
    if max_val is None:
        max_val = x.max(dim=0, keepdim=True).values
    return (x - min_val) / (max_val - min_val + eps)


class EarlyStopping:
    """早停器，训练时判断验证集指标变化，避免过拟合"""

    def __init__(
        self,
        patience=5,
        mode="min",
        delta=1e-3,
        on_continue: Optional[Callable[[], None]] = None,
    ):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.on_continue = on_continue
        self.best_score = None
        self.counter = 0
        self.early_stop = False

        if mode == "min":
            self.compare = lambda current, best: current < best - delta
            self.best_score = float("inf")
        else:
            self.compare = lambda current, best: current > best + delta
            self.best_score = float("-inf")

    def step(self, curr_score):
        if self.compare(curr_score, self.best_score):
            self.best_score = curr_score
            self.counter = 0
            if self.on_continue is not None:
                self.on_continue()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

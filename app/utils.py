import pandas as pd
from torch.utils.data import DataLoader

from app.data.dataset import RecDataset


def data_split(
    df_model: pd.DataFrame, multi_sparse, train_size=0.7, val_size=0.1
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    将数据集分割为训练集，验证集和测试集
    :param df_model: 编码后的特征数据
    :param multi_sparse: 多值稀疏特征矩阵
    :param train_size: 训练集占比
    :param val_size: 验证集占比
    :return: 训练集，验证集和测试集的特征数据及多值稀疏特征矩阵
    """
    # 确定各个输出集合的大小
    n_total = len(df_model)
    n_train = int(n_total * train_size)
    n_val = int(n_total * val_size)
    # 分割数据集索引
    train_idx = df_model.index[:n_train]
    val_idx = df_model.index[n_train : n_train + n_val]
    test_idx = df_model.index[n_train + n_val :]
    # 分割数据集
    train_df = df_model.iloc[train_idx].reset_index(drop=True)
    val_df = df_model.iloc[val_idx].reset_index(drop=True)
    test_df = df_model.iloc[test_idx].reset_index(drop=True)
    # 分割多值稀疏特征矩阵
    train_ms = {col: multi_sparse[col][train_idx] for col in multi_sparse}
    val_ms = {col: multi_sparse[col][val_idx] for col in multi_sparse}
    test_ms = {col: multi_sparse[col][test_idx] for col in multi_sparse}
    # 定义稠密特征列表
    dense_feats = ["age"]
    # 构造数据集
    train_dataset = RecDataset(train_df, train_ms, dense_feats)
    val_dataset = RecDataset(val_df, val_ms, dense_feats)
    test_dataset = RecDataset(test_df, test_ms, dense_feats)
    # 构造加载器
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True)

    return train_loader, val_loader, test_loader

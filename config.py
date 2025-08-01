from pathlib import Path

# 配置项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

FEAT_NAMES = {
    # 稠密（数值）特征列
    "dense_feats": ["age"],
    # 单值稀疏特征列
    "sparse_feats": ["userId", "movieId", "gender", "occupation"],
    # 多值稀疏特征列
    "multi_sparse_feats": ["genres", "tag"],
}

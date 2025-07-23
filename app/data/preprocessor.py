import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class DataPreprocessor:
    def __init__(self, raw_path: Path):
        self.movies_path = raw_path / "movies.csv"
        self.ratings_path = raw_path / "ratings.csv"
        self.tags_path = raw_path / "tags.csv"
        # 用户信息, user_id: {'gender': ..., 'age': ..., 'occupation': ...}
        self.user_info = {}
        self.encoders = {}
        self.feature_dims = {}
        # 存储多值特征的稀疏编码
        self.multi_sparse = {}

    def _generate_user_info(self, user_id):
        """随机生成用户信息"""
        if user_id not in self.user_info:
            gender = np.random.randint(0, 2)
            age = np.random.randint(18, 61)
            occupation = np.random.randint(0, 10)  # 共 10 种职业
            self.user_info[user_id] = {
                "gender": gender,
                "age": age,
                "occupation": occupation,
            }
        return self.user_info[user_id]

    def _encode_list_label(self, le: LabelEncoder, col_labels: pd.Series):
        flat_labels = [label for labels in col_labels for label in labels]
        le.fit(flat_labels)
        # 记录最大列表长度，作为矩阵维度
        max_len = max(len(labels) for labels in col_labels)
        encoded = np.full((len(col_labels), max_len), -1, dtype=int)
        for i, labels in enumerate(col_labels):
            # 矩阵的前 len(labels) 列存储为编码值
            encoded[i, : len(labels)] = le.transform(labels)
        return encoded

    def fit_transform(self):
        # 1. 加载数据
        logger.info("读取原始数据...")
        movies = pd.read_csv(self.movies_path)
        ratings = pd.read_csv(self.ratings_path)
        tags = pd.read_csv(self.tags_path)

        # 2. 处理 movie genres, 存储为队列
        logger.info("处理电影类别...")
        movies["genres"] = movies["genres"].apply(
            lambda x: x.split("|") if isinstance(x, str) else ["Unknown"]
        )

        # 3. 合并 ratings 与 movies
        logger.info("合并评分表与电影表...")
        df = ratings.merge(movies[["movieId", "genres"]], on="movieId", how="left")
        df["genres"] = df["genres"].apply(
            lambda x: x if isinstance(x, list) else ["Unknown"]
        )

        # 4. 合并 tags
        logger.info("处理电影标签...")
        tags_grouped = (
            tags.groupby(["userId", "movieId"])["tag"].apply(list).reset_index()
        )
        logger.info("合并标签表...")
        df = df.merge(tags_grouped, on=["userId", "movieId"], how="left")
        df["tag"] = df["tag"].apply(lambda x: x if isinstance(x, list) else ["Unknown"])

        # 5. 特征扩展，构造 user 信息并合并
        logger.info("特征扩展, 构造用户信息, 合并用户表...")
        unique_users = df["userId"].unique()
        user_infos = [self._generate_user_info(uid) for uid in unique_users]
        user_df = pd.DataFrame(user_infos)
        user_df["userId"] = unique_users
        df = df.merge(user_df, on="userId", how="left")

        # 6. 生成标签（评分 >= 4 为正样本）
        logger.info("生成二分类标签...")
        df["label"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)

        # 7. 选择用于 embedding 的稀疏特征列，单独处理数值特征列
        sparse_cols = ["userId", "movieId", "gender", "occupation", "genres", "tag"]
        dense_cols = ["age"]
        df_model = df[sparse_cols + dense_cols + ["label"]].copy()
        # 打乱数据, 方便后续切分训练集和测试集
        df_model = df_model.sample(frac=1, random_state=42).reset_index(drop=True)

        # 8. 稀疏特征编码
        logger.info("稀疏特征编码...")
        for col in sparse_cols:
            le = LabelEncoder()
            if isinstance(df_model[col].iloc[0], list):
                # 多值特征编码
                self.multi_sparse[col] = self._encode_list_label(le, df_model[col])
            else:
                df_model[col] = le.fit_transform(df_model[col])
            self.encoders[col] = le
            self.feature_dims[col] = len(le.classes_)

        # 9. 数值特征 min-max 归一化
        logger.info("数值特征归一化...")
        for col in dense_cols:
            scaler = MinMaxScaler()
            df_model[col] = scaler.fit_transform(df_model[[col]])
            self.encoders[col + "_scaler"] = scaler

        return df_model

    def save(self, dir_path: Path, df_model: pd.DataFrame):
        dir_path.mkdir(parents=True, exist_ok=True)
        # 保存多值稀疏特征矩阵
        logger.info("保存多值稀疏特征矩阵...")
        for col, matrix in self.multi_sparse.items():
            np.save(dir_path / f"multi_sparse_{col}.npy", matrix)
        # 保存特征维度
        logger.info("保存特征维度信息...")
        with open(dir_path / "feature_dims.json", "w") as f:
            json.dump(self.feature_dims, f, indent=2)
        # 保存编码器（LabelEncoder 和 MinMaxScaler）
        logger.info("保存编码器...")
        with open(dir_path / "encoders.pkl", "wb") as f:
            pickle.dump(self.encoders, f)
        # 保存编码后特征数据
        logger.info("保存编码后的特征数据...")
        df_model.to_csv(dir_path / "df_model.csv", index=False)

    def load(self, dir_path: Path):
        # 判断关键文件是否存在
        required_files = [
            dir_path / "df_model.csv",
            dir_path / "feature_dims.json",
            dir_path / "encoders.pkl",
        ]
        missing = [f for f in required_files if not f.exists()]
        if missing:
            logger.info("部分编码文件缺失，重新进行特征处理和编码...")
            df_model = self.fit_transform()
            self.save(dir_path, df_model)
        else:
            # 读取多值稀疏特征矩阵
            logger.info("加载多值稀疏特征矩阵...")
            self.multi_sparse = {}
            for fn in dir_path.glob("multi_sparse_*.npy"):
                col = fn.stem.split("multi_sparse_")[1]
                self.multi_sparse[col] = np.load(fn)
            # 读取特征维度
            logger.info("加载特征维度信息...")
            with open(dir_path / "feature_dims.json", "r") as f:
                self.feature_dims = json.load(f)
            # 读取编码器
            logger.info("加载编码器...")
            with open(dir_path / "encoders.pkl", "rb") as f:
                self.encoders = pickle.load(f)
            # 读取编码后特征数据
            logger.info("加载编码后的特征数据...")
            df_model = pd.read_csv(dir_path / "df_model.csv")

        return df_model, self.multi_sparse, self.feature_dims, self.encoders


if __name__ == "__main__":
    data_path = Path("resources/data")
    preprocessor = DataPreprocessor(data_path / "raw")
    try:
        df_model = preprocessor.fit_transform()
        preprocessor.save(data_path / "encoded", df_model)
    except Exception as e:
        logger.exception(e)

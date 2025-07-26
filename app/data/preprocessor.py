import json
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


class DataPreprocessor:
    def __init__(self, raw_path: Path):
        self.movies_path = raw_path / "movies.csv"
        self.ratings_path = raw_path / "ratings.csv"
        self.tags_path = raw_path / "tags.csv"
        # 存储稀疏特征的编码类别数
        self.sparse_n_cls = {}

    def _generate_user_info(self, user_ids) -> pd.DataFrame:
        """批量生成用户信息"""
        shape = (len(user_ids),)
        genders = torch.randint(0, 2, shape)
        ages = torch.randint(18, 61, shape)
        occus = torch.randint(0, 21, shape)  # 共 20 种职业
        return pd.DataFrame(
            {
                "userId": user_ids,
                "gender": genders.numpy(),
                "age": ages.numpy(),
                "occupation": occus.numpy(),
            }
        )

    def _encode_sparse_feat(
        self, series: pd.Series
    ) -> tuple[torch.Tensor | list[list[str]], int]:
        """稀疏特征编码"""
        if isinstance(series.iloc[0], list):
            # 多值特征编码, 展开变长列表后再进行映射
            flats = [val for values in series for val in values]
            # 采用 pd.unique, 性能比 LabelEncoder 内部采用的 np.unique 好
            uniques = pd.Series(flats).dropna().unique()
            encoder = {v: i for i, v in enumerate(uniques)}
            # 张量不支持变长列表，记录原始的变长列表
            encoded_col = series.apply(lambda lst: [encoder.get(x) for x in lst])
            encoded_col = encoded_col.values.tolist()
        else:
            # 单值特征编码，直接进行映射
            uniques = series.dropna().unique()
            encoder = {v: i for i, v in enumerate(uniques)}
            # pd.Series.map 不支持直接映射 list
            encoded_col = torch.tensor(series.map(encoder).values, dtype=torch.int64)

        return encoded_col, len(uniques)

    def fit_transform(self) -> dict[str, torch.Tensor | list[list[str]]]:
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
        # 批量生成用户信息
        user_df = self._generate_user_info(unique_users)
        df = df.merge(user_df, on="userId", how="left")

        # 6. 生成标签（评分 >= 4 为正样本）
        logger.info("生成二分类标签...")
        df["label"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)

        # 7. 选择用于 embedding 的稀疏特征列，单独处理稠密（数值）特征列
        sparse_cols = ["userId", "movieId", "gender", "occupation", "genres", "tag"]
        dense_cols = ["age"]

        # 8. 构造字典存储编码结果
        encoded = dict[str, torch.Tensor | list[list[str]]]()

        # 9. 稀疏特征编码
        logger.info("稀疏特征编码...")
        for col in tqdm(sparse_cols, desc="稀疏特征编码", ncols=100):
            encoded_col, n_cls = self._encode_sparse_feat(df[col])
            encoded[col] = encoded_col
            self.sparse_n_cls[col] = n_cls

        # 10. 稠密特征编码，min-max 归一化
        logger.info("稠密特征归一化...")
        for col in dense_cols:
            # 通过 df[[col]] 传入 (n, 1) 的 二维数据，匹配接口
            encoded_list = MinMaxScaler().fit_transform(df[[col]]).flatten()
            encoded[col] = torch.tensor(encoded_list, dtype=torch.float32)

        # 11. 存储标签值
        encoded["label"] = torch.tensor(df["label"].values, dtype=torch.float32)

        return encoded

    def save(
        self, dir_path: Path, encoded_feats: dict[str, torch.Tensor | list[list[str]]]
    ) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        # 保存特征类别数
        logger.info("保存稀疏特征类别数...")
        with open(dir_path / "sparse_n_cls.json", "w") as f:
            json.dump(self.sparse_n_cls, f, indent=2)
        # 保存特征值编码，此时为张量列表，使用 pt 存储
        logger.info("保存特征值编码...")
        torch.save(encoded_feats, dir_path / "encoded_feats.pt")

    def load(
        self, dir_path: Path
    ) -> tuple[dict[str, torch.Tensor | list[list[str]]], dict[str, int]]:
        # 判断关键文件是否存在
        required_files = [
            dir_path / "encoded_feats.pt",
            dir_path / "sparse_n_cls.json",
        ]
        missing = [f for f in required_files if not f.exists()]
        if missing:
            logger.info("部分编码文件缺失，重新进行特征处理和编码...")
            encoded_feats = self.fit_transform()
            self.save(dir_path, encoded_feats)
        else:
            # 读取特征类别数
            logger.info("加载特征类别数...")
            with open(dir_path / "sparse_n_cls.json", "r") as f:
                self.sparse_n_cls = json.load(f)
            # 读取编码后特征数据
            logger.info("加载特征值编码...")
            encoded_feats = torch.load(dir_path / "encoded_feats.pt")

        return encoded_feats, self.sparse_n_cls


if __name__ == "__main__":
    data_path = Path("resources/data")
    preprocessor = DataPreprocessor(data_path / "raw")
    try:
        encoded_feats = preprocessor.fit_transform()
        preprocessor.save(data_path / "encoded", encoded_feats)
        # encoded_feats, sparse_n_cls = preprocessor.load(data_path / "encoded")
    except Exception as e:
        logger.exception(e)

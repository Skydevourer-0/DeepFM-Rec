import json
import pickle
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
        # 存储多值特征编码的不规则张量形式
        self.sparse_flats = {}
        self.sparse_offsets = {}

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

    def _encode_multi_sparse(self, series: pd.Series, feat: str):
        """多值稀疏特征编码，进行不规则张量处理"""
        if not isinstance(series.iloc[0], list):
            return
        # 多值特征编码, 转化为 ragged 形式，即 (flats, offsets)
        flats = [val for values in series for val in values]
        # 采用 pd.unique, 性能比 LabelEncoder 内部采用的 np.unique 好
        uniques = pd.Series(flats).dropna().unique()
        vocab = {v: i for i, v in enumerate(uniques)}
        # 对 flats 进行编码
        flats = [vocab[x] for x in flats]
        # 累加计算偏移量
        lens = torch.tensor([0] + [len(x) for x in series])
        offsets = lens.cumsum(dim=0)[:-1]
        # 存储 flats, offsets
        self.sparse_flats[feat] = flats
        self.sparse_offsets[feat] = offsets.tolist()
        # 存储 feat 类别数
        self.sparse_n_cls[feat] = len(uniques)
        # 表格中保存索引方便合并
        return torch.arange(len(offsets)).numpy()

    def _encode_feat(
        self, df: pd.DataFrame, feat: str, dense_feats: list[str] = []
    ) -> torch.Tensor | list[list[int]]:
        """特征编码，包括稠密特征和单值稀疏特征"""
        if feat in dense_feats:
            # 稠密特征编码, min-max 归一化
            # 需要将 series 转化为 2D 形式，匹配接口输入
            encoded_list = MinMaxScaler().fit_transform(df[[feat]]).flatten()
            encoded = torch.tensor(encoded_list, dtype=torch.float32)
            return encoded
        elif feat in self.sparse_flats:
            # 多值稀疏特征编码, 从 (flats, offsets) 中提取变长列表
            # 根据索引选择对应的 offset，从 flats 中提取列表
            flats = self.sparse_flats[feat]
            offsets = self.sparse_offsets[feat] + [len(flats)]
            encoded = [
                [] if idx == -1 else flats[offsets[idx] : offsets[idx + 1]]
                for idx in df[feat]
            ]
            return encoded
        else:
            # 单值稀疏特征编码，直接进行映射
            uniques = df[feat].dropna().unique()
            vocab = {v: i for i, v in enumerate(uniques)}
            encoded = torch.tensor(df[feat].map(vocab).values, dtype=torch.int64)
            # 记录特征类别数
            self.sparse_n_cls[feat] = len(uniques)
            return encoded

    def fit_transform(self) -> dict[str, torch.Tensor | list[list[int]]]:
        # 1. 加载数据
        logger.info("读取原始数据...")
        movies = pd.read_csv(self.movies_path, usecols=["movieId", "genres"])
        ratings = pd.read_csv(
            self.ratings_path, usecols=["userId", "movieId", "rating"]
        )
        tags = pd.read_csv(self.tags_path, usecols=["userId", "movieId", "tag"])

        # 2. 处理 movie genres, 存储为队列
        logger.info("处理电影类别...")
        movies["genres"] = movies["genres"].apply(
            lambda x: x.split("|") if isinstance(x, str) else []
        )
        # 3. 对 genres 编码，进行 ragged 处理
        movies["genres"] = self._encode_multi_sparse(movies["genres"], "genres")

        # 4. 将 rating 作为回归任务的标签
        logger.info("生成回归标签...")
        ratings.rename(columns={"rating": "label"}, inplace=True)

        # 5. 合并 ratings 与 movies
        logger.info("合并评分表与电影表...")
        df = ratings.merge(movies[["movieId", "genres"]], on="movieId", how="left")
        df["genres"] = df["genres"].fillna(-1).astype(int)

        # 6. 合并 tags
        logger.info("处理电影标签...")
        tags = tags.dropna(subset=["tag"])
        tags = tags.groupby(["userId", "movieId"])["tag"].agg(list).reset_index()
        # 对 tags 编码，进行 ragged 处理
        tags["tag"] = self._encode_multi_sparse(tags["tag"], "tag")
        logger.info("合并标签表...")
        df = df.merge(tags, on=["userId", "movieId"], how="left")
        df["tag"] = df["tag"].fillna(-1).astype(int)

        # 7. 特征扩展，构造 user 信息并合并
        logger.info("特征扩展, 构造用户信息, 合并用户表...")
        unique_users = df["userId"].unique()
        # 批量生成用户信息
        user_df = self._generate_user_info(unique_users)
        df = df.merge(user_df, on="userId", how="left")

        # 8. 选择需要归一化的稠密（数值）特征列，其中 label 是回归任务的标签
        dense_feats = ["age", "label"]

        # 9. 构造字典存储编码结果
        encoded = dict[str, torch.Tensor | list[list[int]]]()
        logger.info("特征编码...")
        for feat in tqdm(df.columns, desc="特征编码", ncols=100):
            encoded[feat] = self._encode_feat(df, feat, dense_feats)

        return encoded

    def save(
        self, dir_path: Path, encoded: dict[str, torch.Tensor | list[list[int]]]
    ) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        # 保存特征类别数
        logger.info("保存稀疏特征类别数...")
        with open(dir_path / "sparse_n_cls.json", "w") as f:
            json.dump(self.sparse_n_cls, f, indent=2)
        # 保存特征值编码
        # 拆分单值特征和多值特征
        single_feats = {k: v for k, v in encoded.items() if isinstance(v, torch.Tensor)}
        multi_feats = {k: v for k, v in encoded.items() if isinstance(v, list)}
        # 单值特征编码为 tensor, 存储为 pt
        logger.info("保存单值特征编码...")
        torch.save(single_feats, dir_path / "single_feats.pt")
        # 多值特征编码为 list[list[int]], 存储为 pkl.gz
        logger.info("保存多值特征编码...")
        with open(dir_path / "multi_feats.pkl", "wb") as f:
            pickle.dump(multi_feats, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(
        self, dir_path: Path
    ) -> tuple[dict[str, torch.Tensor | list[list[int]]], dict[str, int]]:
        # 判断关键文件是否存在
        required_files = [
            dir_path / "single_feats.pt",
            dir_path / "multi_feats.pkl",
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
            # 读取特征值编码
            logger.info("加载单值特征编码...")
            single_feats = torch.load(dir_path / "single_feats.pt")
            logger.info("加载多值特征编码...")
            with open(dir_path / "multi_feats.pkl", "rb") as f:
                multi_feats = pickle.load(f)
            # 整合单值特征和多值特征
            encoded_feats = {**single_feats, **multi_feats}

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

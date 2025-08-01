import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from app.utils.types import Ragged, RaggedMemmap
from app.utils.utils import min_max_scale_tensor
from config import FEAT_NAMES


class DataPreprocessor:
    def __init__(self, raw_path: Path):
        self.movies_path = raw_path / "movies.csv"
        self.ratings_path = raw_path / "ratings.csv"
        self.tags_path = raw_path / "tags.csv"
        # 存储元数据，包括样本数和特征类别数
        self.n_samples = 0
        self.sparse_shapes = {}
        # 存储多值特征编码的不规则张量形式
        self.sparse_flats = dict[str, np.ndarray]()
        self.sparse_offsets = dict[str, np.ndarray]()
        # 稠密特征列, 其中 label 是回归任务的标签
        self.dense_feats = FEAT_NAMES["dense_feats"] + ["label"]
        # 单值特征列
        self.sparse_feats = FEAT_NAMES["sparse_feats"]
        # 多值特征列
        self.multi_feats = FEAT_NAMES["multi_sparse_feats"]

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
        flats = np.array([vocab[x] for x in flats])
        # 累加计算偏移量
        lens = np.array([0] + [len(x) for x in series])
        # 保留最后一位，防止构造变长列表时 idx+1 越界
        offsets = lens.cumsum(axis=0)
        # 存储 flats, offsets
        self.sparse_flats[feat] = flats
        self.sparse_offsets[feat] = offsets
        # 存储 feat 类别数
        self.sparse_shapes[feat] = len(uniques)
        # 表格中保存索引方便合并
        return torch.arange(len(offsets) - 1).numpy()

    def _encode_feat(self, df: pd.DataFrame, feat: str) -> Ragged:
        """特征编码，包括稠密特征和单值稀疏特征"""
        if feat in self.dense_feats:
            # 稠密特征编码, min-max 归一化
            encoded = min_max_scale_tensor(df[feat])
            return encoded.cpu().numpy().astype(np.float32)
        elif feat in self.multi_feats:
            # 多值稀疏特征编码, 保存 (flats, offsets, indices)
            flats = self.sparse_flats[feat].astype(np.int32)
            offsets = self.sparse_offsets[feat].astype(np.int32)
            encoded = (flats, offsets, df[feat].to_numpy().astype(np.int32))
            return encoded
        else:
            # 单值稀疏特征编码，直接进行映射
            uniques = df[feat].dropna().unique()
            vocab = {v: i for i, v in enumerate(uniques)}
            encoded = df[feat].map(vocab).fillna(-1).to_numpy().astype(np.int32)
            # 记录特征类别数
            self.sparse_shapes[feat] = len(uniques)
            return encoded

    def fit_transform(self, save_path: Path) -> None:
        """将原始数据处理后，特征编码存储到指定目录"""

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

        # 8. 记录元数据，进行特征编码并存储到文件
        logger.info("记录元数据，进行特征编码并存储到文件...")
        save_path.mkdir(parents=True, exist_ok=True)
        self.n_samples = df.shape[0]
        for feat in df.columns:
            encoded = self._encode_feat(df, feat)
            self.save_encoded_feat(save_path, feat, encoded)
        self.save_metadata(save_path)

    def _encoded_completed(self, dir_path: Path) -> bool:
        if not dir_path.exists():
            return False
        # 所需文件列表
        required_files = []
        # 单值特征对应的文件
        required_files += [
            dir_path / f"{feat}.npy" for feat in self.dense_feats + self.sparse_feats
        ]
        # 多值特征需要 flats, offsets, indices 三个文件
        for feat in self.multi_feats:
            required_files += [
                dir_path / f"{feat}_flats.npy",
                dir_path / f"{feat}_offsets.npy",
                dir_path / f"{feat}_indices.npy",
            ]
        # 检查所有文件是否存在
        return all(path.exists() for path in required_files)

    def save_metadata(self, dir_path: Path):
        logger.info("保存元数据...")
        metadata = {
            "n_samples": self.n_samples,
            "sparse_shapes": self.sparse_shapes,
        }
        with open(dir_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def save_encoded_feat(self, dir_path: Path, feat: str, encoded: Ragged) -> None:
        # 保存特征值编码
        if isinstance(encoded, np.ndarray):
            # 单值特征编码为 np.ndarray, 存储单个文件
            logger.info(f"保存单值特征编码 {feat}...")
            np.save(dir_path / f"{feat}.npy", encoded)
        else:
            # 多值特征编码为 tuple, 存储 (flats, offsets, indices) 三个文件
            flats, offsets, indices = encoded
            np.save(dir_path / f"{feat}_flats.npy", flats)
            np.save(dir_path / f"{feat}_offsets.npy", offsets)
            np.save(dir_path / f"{feat}_indices.npy", indices)

    def load(
        self, dir_path: Path
    ) -> tuple[dict[str, RaggedMemmap], int, dict[str, int]]:
        """
        采用 memmap 的方式读取特征编码文件，实现按需读取，避免一次性读入内存过多
        """

        # 判断关键文件是否存在
        if not self._encoded_completed(dir_path):
            logger.info("部分编码文件缺失，重新进行特征处理和编码...")
            self.fit_transform(dir_path)

        # 读取特征类别数
        logger.info("加载元数据...")
        with open(dir_path / "metadata.json", "r") as f:
            metadata = json.load(f)
            self.n_samples = metadata["n_samples"]
            self.sparse_shapes = metadata["sparse_shapes"]
        # 读取特征值编码
        encoded_feats = {}
        logger.info("加载稠密特征编码...")
        for feat in self.dense_feats:
            encoded = np.load(dir_path / f"{feat}.npy", mmap_mode="r")
            encoded_feats[feat] = encoded
        logger.info("加载单值稀疏特征编码...")
        for feat in self.sparse_feats:
            encoded = np.load(dir_path / f"{feat}.npy", mmap_mode="r")
            encoded_feats[feat] = encoded
        logger.info("加载多值稀疏特征编码...")
        for feat in self.multi_feats:
            ragged_name = ["flats", "offsets", "indices"]
            encoded = tuple(
                np.load(dir_path / f"{feat}_{x}.npy", mmap_mode="r")
                for x in ragged_name
            )
            encoded_feats[feat] = encoded

        return encoded_feats, self.n_samples, self.sparse_shapes


if __name__ == "__main__":
    data_path = Path("resources/data")
    preprocessor = DataPreprocessor(data_path / "raw/ml-10m")
    try:
        encoded_path = data_path / "encoded"
        encoded_feats = preprocessor.fit_transform(encoded_path)
        # encoded_feats, n_samples, sparse_shapes = preprocessor.load(encoded_path)
    except Exception as e:
        logger.exception(e)

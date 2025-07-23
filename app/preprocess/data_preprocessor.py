import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class DataPreprocessor:
    def __init__(self, movies_path, ratings_path, tags_path):
        self.movies_path = movies_path
        self.ratings_path = ratings_path
        self.tags_path = tags_path
        # 用户信息, user_id: {'gender': ..., 'age': ..., 'occupation': ...}
        self.user_info = {}
        self.encoders = {}
        self.feature_dims = {}
        # 存储多值特征的稀疏编码
        self.multi_sparse = {}

    def _generate_user_info(self, user_id):
        """随机生成用户信息"""
        if user_id not in self.user_info:
            gender = np.random.choice([0, 1])
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

    def preprocess(self):
        # 1. 加载数据
        movies = pd.read_csv(self.movies_path)
        ratings = pd.read_csv(self.ratings_path)
        tags = pd.read_csv(self.tags_path)

        # 2. 处理 movie genres, 存储为队列
        movies["genres"] = movies["genres"].apply(
            lambda x: x.split("|") if isinstance(x, str) else ["Unknown"]
        )

        # 3. 合并 ratings 与 movies
        df = ratings.merge(movies[["movieId", "genres"]], on="movieId", how="left")
        df["genres"] = df["genres"].fillna("Unknown")

        # 4. 合并 tags
        tags_grouped = (
            tags.groupby(["userId", "movieId"])["tag"].apply("list").reset_index()
        )
        df = df.merge(tags_grouped, on=["userId", "movieId"], how="left")
        df["tag"] = df["tag"].fillna("Unknown")

        # 5. 特征扩展，构造 user 信息并合并
        unique_users = df["userId"].unique()
        user_infos = [self._generate_user_info(uid) for uid in unique_users]
        user_df = pd.DataFrame(user_infos)
        user_df["userId"] = unique_users
        df = df.merge(user_df, on="userId", how="left")

        # 6. 生成标签（评分 >= 4 为正样本）
        df["label"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)

        # 7. 选择用于 embedding 的稀疏特征列，单独处理数值特征列
        sparse_cols = ["userId", "movieId", "gender", "occupation", "genres", "tag"]
        dense_cols = ["age"]
        df_model = df[sparse_cols + dense_cols + ["label"]].copy()

        # 8. 稀疏特征编码
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
        for col in dense_cols:
            scaler = MinMaxScaler()
            df_model[col] = scaler.fit_transform(df_model[[col]])


        return df_model

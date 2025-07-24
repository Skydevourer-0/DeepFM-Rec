from pathlib import Path

from loguru import logger

from app.data.preprocessor import DataPreprocessor
from app.models.fm_model import FMModule
from app.train import Trainer
from app.utils import data_split


def main():
    data_path = Path("resources/data")
    encoded_path = data_path / "encoded"
    model_path = data_path / "models"

    try:
        preprocessor = DataPreprocessor(data_path / "raw")
        # 载入预处理数据
        df_model, multi_sparse, feature_dims, _ = preprocessor.load(encoded_path)
        # 分割数据集
        train_loader, val_loader, test_loader = data_split(df_model, multi_sparse)
    except Exception as e:
        logger.exception(f"数据预处理出错: {repr(e)}")
        return

    try:
        # 构造多值稀疏特征列和稠密特征列
        multi_feats = ["genres", "tag"]
        dense_feats = ["age"]
        # 构造模型实例
        fm_model = FMModule(feature_dims, multi_feats, dense_feats)
        # 构造训练器实例
        trainer = Trainer(fm_model, train_loader, val_loader)
        # 开始训练
        best_model, metrics = trainer.train(epochs=100)
        # 绘制指标
        metrics.draw(model_path / "metrics.jpg")
        # 保存训练结果
        trainer.save_model(model_path / "model_weights.pth", best_model)
        # 测试模型
        loss, auc = trainer.evaluate(test_loader, best_model)
        logger.info(f"测试集评估指标: BCE 损失: {loss:.2f}, AUC 分数: {auc:.2f}")

    except Exception as e:
        logger.exception(f"模型训练出错: {repr(e)}")


if __name__ == "__main__":
    main()

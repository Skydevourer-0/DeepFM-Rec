from pathlib import Path

from loguru import logger

from app.data.manager import RecDataManager
from app.data.preprocessor import DataPreprocessor
from app.models.deep_fm import DeepFM
from app.train import Trainer


def main():
    data_path = Path("resources/data")
    encoded_path = data_path / "encoded"
    model_path = data_path / "models"

    try:
        preprocessor = DataPreprocessor(data_path / "raw")
        # 载入预处理数据
        encoded_feats, n_samples, sparse_shapes = preprocessor.load(encoded_path)
        # 分割数据集
        manager = RecDataManager(encoded_feats, n_samples=n_samples)
        train_loader, valid_loader, test_loader = manager.split(
            num_workers=4, pin_memory=True
        )
    except Exception as e:
        logger.exception(f"数据预处理出错: {repr(e)}")
        return

    try:
        # 构造模型实例
        model = DeepFM(sparse_shapes)
        # 构造训练器实例
        trainer = Trainer(model, train_loader, valid_loader)
        # 开始训练
        best_model, metrics = trainer.train(epochs=50)
        # 绘制指标
        metrics.draw(model_path / "metrics.jpg")
        # 测试模型
        loss, mae = trainer.evaluate(test_loader, best_model)
        logger.info(f"测试集评估指标: MSE 损失: {loss:.4f}, MAE 损失: {mae:.4f}")

    except Exception as e:
        logger.exception(f"模型训练出错: {repr(e)}")


if __name__ == "__main__":
    main()

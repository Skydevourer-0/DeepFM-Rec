from pathlib import Path

from loguru import logger
from memory_profiler import profile

from app.data.dataset import RecDataset
from app.data.preprocessor import DataPreprocessor
from app.models.fm_model import FMModule
from app.train import Trainer


@profile
def main():
    data_path = Path("resources/data")
    encoded_path = data_path / "encoded"
    model_path = data_path / "models"

    try:
        preprocessor = DataPreprocessor(data_path / "raw")
        # 载入预处理数据
        encoded_feats, n_samples, sparse_shapes = preprocessor.load(encoded_path)
        # 分割数据集
        dataset = RecDataset(encoded_feats, n_samples=n_samples)
        train_loader, valid_loader, test_loader = dataset.split(train_size=0.1)
    except Exception as e:
        logger.exception(f"数据预处理出错: {repr(e)}")
        return

    try:
        # 构造模型实例
        fm_model = FMModule(sparse_shapes)
        # 构造训练器实例
        trainer = Trainer(fm_model, train_loader, valid_loader)
        # # 开始训练
        # best_model, metrics = trainer.train(epochs=100)
        # # 绘制指标
        # metrics.draw(model_path / "metrics.jpg")
        # # 保存训练结果
        # trainer.save_model(model_path / "model_weights.pth", best_model)
        # # 测试模型
        # loss, mse = trainer.evaluate(test_loader, best_model)
        # 训练一轮
        loss, mae = trainer._one_epoch(train_loader, training=True)
        logger.info(f"测试集评估指标: MSE 损失: {loss:.4f}, MAE 损失: {mae:.4f}")

    except Exception as e:
        logger.exception(f"模型训练出错: {repr(e)}")


if __name__ == "__main__":
    main()

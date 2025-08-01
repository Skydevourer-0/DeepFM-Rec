# DeepFM-Rec

基于 DeepFM 的电影推荐系统（DeepFM-Driven Movie Recommendation System）

## 目录结构

```python
DeepFM-Rec
├── app                         # 主应用包，包含项目核心代码
│   ├── __init__.py             # 包初始化文件
│   ├── data                    # 数据处理相关模块
│   │   ├── dataset.py          # 自定义 Dataset 类，封装数据读取与样本索引访问
│   │   ├── manager.py          # 数据管理类，处理数据加载、划分及批处理
│   │   └── preprocessor.py     # 数据预处理模块，包含特征编码、归一化等操作
│   ├── layers                  # 模型层定义
│   │   ├── dnn.py              # DNN 层实现，包含多层感知机结构
│   │   ├── embedding.py        # 嵌入层实现，支持稠密特征，单值和多值稀疏特征编码
│   │   └── fm.py               # FM 模型相关层及计算逻辑
│   ├── models                  # 模型定义
│   │   ├── deep_fm.py          # DeepFM 模型实现，融合 FM 和 DNN 结构
│   │   └── fm_model.py         # 基础 FM 模型实现
│   ├── scripts                 # 项目运行脚本及示例
│   │   └── baseline.py         # 基础训练与评估流水线脚本
│   ├── train.py                # 训练流程主入口，包含训练循环与验证逻辑
│   └── utils                   # 工具模块
│       ├── types.py            # 自定义类型定义与类型注解
│       └── utils.py            # 通用辅助函数集合
├── app.py                      # 项目主运行脚本或入口文件
├── config.py                   # 配置文件，包含超参数与路径设置
└── tests                       # 测试代码目录
    └── test_performance.py     # 性能测试相关单元测试，通过 torch.profiler 实现一个 batch 中的性能监测
```

## 项目亮点分析

### 0. 模型选择

- 深刻分析了传统协同过滤的不足（无监督、测试时效率低），有理有据地选用 DeepFM。
- DeepFM 结合因子分解机（FM）和深度神经网络（DNN）优势，既捕捉低阶特征交叉，又自动学习高阶非线性交叉，提升模型表达能力和泛化性能。相关结构实现见 `app/layers/fm.py` 和 `app/layers/dnn.py`。

### 1. 数据预处理创新

- 采用公开的 MovieLens ml-10m 数据集，针对用户信息缺失进行了特征扩展（性别、年龄、职业），详见 `app/data/preprocessor.py` 的 `_generate_user_info` 方法。
- 设计了多值稀疏特征的内存不规则存储结构（flats、offsets、indices）。使用 np.memmap 加载特征编码，极大节省内存，且高效访问。相关编码逻辑见 `app/data/preprocessor.py`。
- 自定义 Dataset 和 DataLoader，配合自定义 collate_fn 实现多值变长特征批处理，嵌入层使用 EmbeddingBag 无缝支持，详见 `app/data/dataset.py` 和 `app/layers/embedding.py`。

### 2. 训练流程成熟

- 同时监控 MSE 和 MAE 指标，采用验证集 MAE 实现早停，防止过拟合。训练主流程见 `app/train.py`。
- 学习率调度结合指标反馈，动态调整训练步长，提高收敛效率，相关实现见 `app/train.py`。
- 使用 Loguru 记录训练日志，结合 matplotlib 实时绘制指标曲线，方便训练过程的可视化和分析，详见 `app/train.py`。

### 3. 模型实现细节

- FM 模型设计严谨，一阶线性项 embedding_dim=1 保证线性映射，二阶交叉项采用经典的 sum-square 技巧计算，详见 `app/layers/fm.py`。
- DeepFM 模型复用 FM 的嵌入层，结合多层 DNN 自动学习高阶非线性交叉，模型模块化清晰，易扩展，详见 `app/models/deep_fm.py`。
- 嵌入向量复用机制避免重复计算，提升训练和推理效率，相关逻辑见 `app/layers/embedding.py`。

### 4. 性能与调优

- 利用 PyTorch Profiler 精细分析训练过程 CPU 和 GPU 利用瓶颈，详见 `tests/test_performance.py`。
- 设置 DataLoader 的 num_workers 和 pin_memory，实现多线程载入加速训练，详见 `app/data/manager.py`。
- 控制训练耗时合理，兼顾训练速度与模型性能，适合实验性质项目。

### 5. 项目实验性质明确

- 虽未实现 TopK 推荐的完整业务流程，但完成了模型训练、验证和测试评估，取得了稳定的 MSE 和 MAE 表现，相关评估流程见 `app/train.py`。
- 项目目标明确，兼顾系统设计和工程实现，适合作为推荐系统学习和实验示范。

## 许可证

该项目遵循 MIT 许可证。

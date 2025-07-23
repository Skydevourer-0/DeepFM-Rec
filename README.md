# DeepFM-Rec
基于 DeepFM 的电影推荐系统（DeepFM-Driven Movie Recommendation System）

## 项目结构

```
flask-app
├── app
│   ├── __init__.py        # Flask 应用的初始化文件
│   ├── routes.py          # 定义应用的路由
│   └── models.py          # 数据模型定义
├── requirements.txt        # 项目所需的 Python 包
├── config.py               # 应用配置设置
└── README.md               # 项目的文档和说明
```

## 安装依赖

在项目根目录下运行以下命令以安装所需的依赖：

```
pip install -r requirements.txt
```

## 启动应用

在项目根目录下，使用以下命令启动 Flask 应用：

```
flask run
```

确保在运行之前设置了 `FLASK_APP` 环境变量，例如：

```
export FLASK_APP=app
```

## 贡献

欢迎任何形式的贡献！请提交问题或拉取请求。

## 许可证

该项目遵循 MIT 许可证。
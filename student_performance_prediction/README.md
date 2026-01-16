# 基于机器学习的学生成绩预测系统

这个项目实现了一个基于机器学习的学生成绩预测系统，通过分析学生的历史数据（如出勤率、作业完成情况、课堂参与度等）来预测他们在未来考试中的表现。

## 项目概述

本系统旨在帮助教育工作者提前识别可能需要额外支持的学生，并为个性化教学提供数据支持。系统不仅可以预测最终成绩，还能识别影响学生学习成绩的关键因素，为教师提供有价值的教学反馈。

## 功能特点

- 数据收集与预处理：清洗、标准化学生数据并进行特征工程
- 特征工程：创建新特征、特征选择和降维
- 模型训练与评估：支持多种机器学习算法，自动选择最佳模型
- 成绩预测：基于学生历史数据预测未来成绩
- 特征重要性分析：识别影响学生成绩的关键因素
- Web应用界面：提供友好的用户界面，方便教师使用系统功能

## 技术栈

- **Python 3.8+**：主要编程语言
- **Pandas & NumPy**：数据处理和分析
- **Scikit-learn**：机器学习模型构建和评估
- **Matplotlib & Seaborn**：数据可视化
- **Flask**：Web应用后端
- **SQLite/MongoDB**：数据存储
- **HTML/CSS/JavaScript**：前端界面

## 项目结构

```
student_performance_prediction/
│
├── data/
│   ├── raw/                  # 原始数据
│   │   └── student_data.csv  # 示例学生数据
│   └── processed/            # 处理后的数据
│
├── model/                    # 训练好的模型
│
├── src/
│   ├── data_processing.py    # 数据预处理模块
│   ├── feature_engineering.py # 特征工程模块
│   ├── model_training.py     # 模型训练模块
│   └── visualization.py      # 数据可视化模块
│
├── static/
│   ├── css/                  # 样式表
│   └── js/                   # 前端脚本
│
├── templates/                # HTML模板
│
├── app.py                    # Flask应用主文件
├── train_model.py            # 模型训练脚本
├── requirements.txt          # 项目依赖
└── README.md                 # 项目说明
```

## 安装与使用

### 环境准备

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

系统默认包含一个示例数据集，位于 `data/raw/student_data.csv`。您也可以使用自己的数据集，只需确保数据格式与示例数据集兼容。

### 数据处理与特征工程

```bash
# 运行数据预处理
python src/data_processing.py

# 运行特征工程
python src/feature_engineering.py
```

### 模型训练

```bash
# 训练模型
python train_model.py
```

### 启动Web应用

```bash
# 启动Flask应用
python app.py
```

启动后，打开浏览器访问 http://localhost:5000 即可使用系统。

## 系统使用流程

1. **数据上传**：上传学生历史数据
2. **数据预处理**：系统自动清洗和标准化数据
3. **特征工程**：系统创建新特征并选择最重要的特征
4. **模型训练**：系统训练多个模型并选择最佳模型
5. **成绩预测**：输入学生信息，获取预测成绩
6. **结果分析**：查看预测结果和影响因素分析

## 贡献指南

欢迎对本项目进行贡献！您可以通过以下方式参与：

1. 提交Bug报告或功能请求
2. 提交代码改进或新功能
3. 完善文档

## 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

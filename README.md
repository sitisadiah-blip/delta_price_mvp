# 🎮 Delta Price Prediction: 基于机器学习的游戏账号价值评估系统

> **毕业设计项目** | **三角洲行动 (Delta Force) 账号交易价格预测**

## 📖 项目简介 (Introduction)

本项目是针对《三角洲行动》游戏账号交易市场开发的机器学习价格预测系统。针对虚拟资产定价难、非线性特征强、数据长尾分布等痛点，构建了从数据清洗、特征工程到多模型对比评估的端到端（End-to-End）流水线。

系统集成了 **Ridge回归**、**SVM**、**随机森林** 及 **LightGBM** 四种算法，旨在探索不同数学原理模型在虚拟资产估值中的表现，最终实现高精度的价格预测。

## ✨ 核心特性 (Key Features)

* **⚡️ 自动化全流程**：一键完成数据加载、清洗、特征提取、建模与评估。
* **🧠 深度特征工程**：
  * **时间序列处理**：将非结构化交易时间转化为"生命周期衰减（Days Elapse）"与"周末效应"特征。
  * **长尾分布矫正**：集成 Log1p 变换与 1%-99% 缩尾处理（Winsorization），解决价格偏态问题。
* **⚔️ 多模型竞技场**：内置四大流派算法对比（线性基线 vs 核方法 vs Bagging vs Boosting）。
* **🚀 硬件加速优化**：针对 Apple Silicon (M-series) 芯片进行了并行计算与中文字体渲染优化。
* **📊 可视化报告**：自动生成预测拟合散点图与详细的性能指标报表（R2, RMSE, MAE）。

## 📂 目录结构 (Directory Structure)

```text
delta_price_mvp/
├── data/                   # 数据存放目录
│   └── train.xlsx          # 原始交易数据 (需自行放入)
├── outputs/                # 输出结果目录
│   ├── figures/            # 生成的对比图表
│   │   ├── model_arena_v5.png
│   │   ├── price_distribution.png
│   │   ├── scatter_gbdt.png
│   │   ├── scatter_ridge.png
│   │   └── top_corr_features.png
│   └── metrics/            # 性能指标 CSV/JSON
│       ├── final_results.csv
│       ├── model_results.csv
│       ├── model_results.json
│       └── data_description.csv
├── src/                    # 源代码目录
│   ├── run_mvp_v5_m4pro.py # 主启动脚本（M4 Pro 优化版）
│   └── run_mvp.py          # 基础版本
├── environment.yml         # Conda 环境配置
├── requirements.txt        # 依赖库列表
└── README.md               # 项目说明文档
```

## 🛠️ 安装与配置 (Installation)

### 方案 1：使用 Conda (推荐)

```bash
# 创建虚拟环境
conda env create -f environment.yml

# 激活环境
conda activate price_mvp
```

### 方案 2：使用 Pip

```bash
# 安装依赖库
pip install -r requirements.txt
```

### 环境依赖版本

| 库 | 版本 | 用途 |
| --- | --- | --- |
| pandas | >=1.3.0 | 数据处理与 Excel 读写 |
| numpy | >=1.21.0 | 数值计算 |
| matplotlib | >=3.4.0 | 可视化 |
| scikit-learn | >=1.0.0 | 机器学习模型 |
| lightgbm | >=3.3.0 | 梯度提升树 |
| openpyxl | >=3.0.0 | Excel 文件支持 |

### 数据准备

请确保将原始数据文件 `train.xlsx` 放入 `data/` 目录下。

* **数据格式要求**：需包含 `成交时间`、`价格` 以及 `战场等级`、`总资产`、`皮肤数量` 等特征列。
* **支持格式**：`.xlsx` 或 `.csv` 文件（脚本自动检测）

## 🚀 快速开始 (Usage)

在项目根目录下运行以下命令启动预测流水线：

```bash
python src/run_mvp_v5_m4pro.py
```

### 运行逻辑：

1. **自动检测**：脚本会自动扫描 `data/` 目录下的 `.xlsx` 或 `.csv` 文件。
2. **预处理**：执行缺失值填充、时间特征提取及对数变换。
3. **模型训练**：依次训练 Ridge, SVM, Random Forest, LightGBM。
4. **结果输出**：终端打印性能战报，并保存图表至 `outputs/`。

**预期输出示例：**
```
📖 正在读取数据: train.xlsx ...
✓ 数据已加载，样本数: 12483，特征数: 18
⚙️ 正在处理时间特征: 成交时间
📊 数据描述已保存至 outputs/metrics/data_description.csv

🤖 开始模型竞技场...
├─ Ridge 回归 ... ✓ R² = 0.757, RMSE = 0.476
├─ SVM (RBF) ... ✓ R² = 0.823, RMSE = 0.406
├─ 随机森林 ... ✓ R² = 0.856, RMSE = 0.367
└─ LightGBM ... ✓ R² = 0.866, RMSE = 0.354 ⭐ BEST

🎉 所有结果已保存至 outputs/
```

## 🔬 实验结果 (Results)

基于测试集（约 1.2 万样本）的实证表现如下（M4 Pro 环境）：

| 模型 (Model) | 算法流派 | R² Score | RMSE (Log) | 训练耗时 | 结论 |
| --- | --- | --- | --- | --- | --- |
| **LightGBM** | Boosting (SOTA) | **0.866** | **0.354** | ~6.0s | **推荐方案，精度最高** |
| Random Forest | Bagging | 0.856 | 0.367 | ~8.2s | 表现稳健，略逊于 LGBM |
| SVM (RBF) | Kernel Method | 0.823 | 0.406 | ~1.2s* | 优于线性，但计算复杂度高 |
| Ridge | Linear | 0.757 | 0.476 | <0.1s | 证明特征间存在非线性关系 |

> *注：SVM 训练采用了下采样策略以平衡计算效率。*

## 📝 方法论细节 (Methodology)

### 1. 数据清洗 (Data Cleaning)
- 剔除无意义特征（ID 列 `商品号`）
- 使用**中位数**填充数值型缺失值，确保鲁棒性

### 2. 目标变换 (Target Transformation)
- 采用 `np.log1p()` 对 `价格` 进行对数变换
- 将偏态分布转化为近似正态分布，提升模型收敛速度
- 应用 1%-99% Winsorization 处理异常值

### 3. 特征工程 (Feature Engineering)
| 特征 | 构造方法 | 作用 |
| --- | --- | --- |
| `is_weekend` | `datetime.dayofweek ∈ [5,6]` | 捕捉周末流量高峰 |
| `days_elapse` | `(日期 - 最早日期).days` | 反映账号生命周期趋势 |
| 其他特征 | 中位数填充 + 保留原始值 | 保留市场竞争信息 |

### 4. 标准化 (Normalization)
- 对所有特征进行 **StandardScaler** (Z-Score) 标准化
- 消除量纲差异，确保 SVM 和 Ridge 模型的有效性

### 5. 模型对比 (Model Comparison)
四大经典算法的纵向对比，从线性到非线性的递进学习：
- **Ridge**：参数化线性模型，拟合基线
- **SVM**：核方法，引入非线性映射
- **Random Forest**：Bagging 集成，减少方差
- **LightGBM**：梯度提升，全局最优搜索

## 🖥️ 系统要求 (Requirements)

- **Python**: 3.9+
- **操作系统**: macOS (M-series 优化), Windows, Linux
- **内存**: >= 4GB（推荐 8GB+）
- **磁盘**: >= 500MB

## 📊 输出文件说明 (Output Files)

| 文件 | 用途 |
| --- | --- |
| `model_arena_v5.png` | 四模型 R² 与 RMSE 对比柱状图 |
| `price_distribution.png` | 原始价格与对数变换后的分布对比 |
| `scatter_gbdt.png` | LightGBM 预测值 vs 实际值散点图 |
| `scatter_ridge.png` | Ridge 预测值 vs 实际值散点图 |
| `top_corr_features.png` | 特征相关性热力图 TOP 15 |
| `final_results.csv` | 完整性能指标汇总表 |
| `model_results.json` | 模型训练日志（JSON 格式） |
| `data_description.csv` | 原始数据统计描述 |

## 🔧 常见问题 (FAQ)

**Q：脚本报错 `FileNotFoundError: 未找到数据文件`**

A：确保 `data/` 目录下有 `.xlsx` 或 `.csv` 文件。文件名可任意，脚本会自动扫描。

**Q：macOS 上中文显示乱码**

A：脚本已内置字体自动检测。如仍有问题，可手动在系统中安装 PingFang SC 或 Arial Unicode MS 字体。

**Q：模型训练速度很慢**

A：这是正常的。LightGBM 和 SVM 在大数据集上需要更多计算时间。可尝试：
- 减少样本量进行快速测试
- 在更高配置的机器上运行
- 调整脚本中的 `n_samples` 参数

**Q：输出的 R² 分数低于预期**

A：虚拟资产定价本身复杂度高，非线性特征多。建议：
- 检查原始数据质量与特征相关性
- 添加更多领域知识特征（例如：账号持仓期、完成任务数等）
- 调整超参数或尝试集成模型

## 📚 参考资源 (References)

- [LightGBM 官方文档](https://lightgbm.readthedocs.io/)
- [Scikit-learn 用户指南](https://scikit-learn.org/stable/user_guide.html)
- [Pandas 数据处理指南](https://pandas.pydata.org/docs/)

## ⚖️ 许可证 (License)

本项目采用 **MIT License** 开源许可证。任何人可自由使用、修改、分发，但需保留原作者声明。

---

**Author**: Schenberg  
**Institution**: 毕业设计项目  
**Last Update**: 2026-02-06  
**Repository**: https://github.com/sitisadiah-blip/delta_price_mvp

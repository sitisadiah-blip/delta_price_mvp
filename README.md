# 三角洲账号价格预测 MVP（可复现工程）

## 你会得到什么
- 一键跑通：读入数据 → EDA 图表 → Ridge 基线 → LightGBM 树模型（失败则自动回退）→ 输出指标/图表到 outputs/
- 默认已适配你当前数据列名：目标=价格、成交时间=成交时间、并默认丢弃“商品号”

## 目录结构
- data/train.xlsx：数据（已放好示例）
- src/run_mvp.py：主脚本
- outputs/figures：图
- outputs/metrics：指标表/日志

## 方式A（推荐）：conda 一键建环境
```bash
conda env create -f environment.yml
conda activate price_mvp
python src/run_mvp.py
```

## 方式B：pip 安装
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/run_mvp.py
```

## 常用开关（救火用）
- 目标取 log：`python src/run_mvp.py --log_target`
- 目标上下1%截断：`python src/run_mvp.py --winsorize`
- 跑 5 折 CV（会多输出 outputs/metrics/cv_r2.json）：`python src/run_mvp.py --cv`

## 你应该看到的产物
- outputs/figures/price_distribution.png
- outputs/figures/top_corr_features.png
- outputs/figures/feature_importance_top10.png（若 LightGBM 可用）
- outputs/metrics/model_results.csv / model_results.json / data_description.csv

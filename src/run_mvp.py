import os
import sys
import argparse
import json
import warnings
import platform
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
# 注意：LightGBM 和 HistGBDT 会在需要时动态导入

# ================= 配置与初始化 =================
# 忽略一些无关紧要的警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

DIRS = ["data", "outputs/figures", "outputs/metrics"]
for d in DIRS:
    os.makedirs(d, exist_ok=True)

# 修复中文乱码的关键设置
def setup_plotting_style():
    """根据操作系统自动设置 Matplotlib 字体"""
    system_name = platform.system()
    if system_name == "Darwin":  # macOS
        # 尝试 macOS 常见中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC']
    elif system_name == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    else:
        # Linux/Docker 环境
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
    
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示方块的问题

# ================= 核心逻辑函数 =================

def load_data_auto(data_dir="data"):
    """智能加载数据：自动寻找 xlsx 或 csv，并处理依赖缺失"""
    # 优先查找 xlsx (因为通常包含原始格式)
    xlsx_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    file_path = None
    
    # 尝试加载 xlsx
    if xlsx_files:
        try:
            import openpyxl
            file_path = xlsx_files[0]
            print(f"📖 发现 Excel 文件，正在读取: {os.path.basename(file_path)}")
            return pd.read_excel(file_path)
        except ImportError:
            print("⚠️ 发现 .xlsx 文件但缺少 'openpyxl' 库。建议运行 pip install openpyxl")
            print("🔄 尝试寻找 CSV 文件作为替代...")
    
    # 尝试加载 csv
    if csv_files:
        # 优先用看起来像主数据的（排除 data_description 等）
        candidates = [f for f in csv_files if "description" not in f and "result" not in f]
        if candidates:
            file_path = candidates[0]
        else:
            file_path = csv_files[0]
        print(f"📖 正在读取 CSV 文件: {os.path.basename(file_path)}")
        return pd.read_csv(file_path)
    
    raise FileNotFoundError("❌ 在 data/ 目录下未找到 train.xlsx 或 .csv 数据文件，请检查上传。")

def build_features_v2(df, target_col, date_col, drop_cols):
    """特征工程 V2：增强时间特征"""
    df = df.copy()
    
    # 1. 提取目标变量
    if target_col not in df.columns:
        raise ValueError(f"数据中找不到目标列 '{target_col}'")
    y = df[target_col].astype(float)
    X = df.drop(columns=[target_col])

    # 2. 处理不需要的列 (如 ID)
    for c in drop_cols:
        if c in X.columns:
            X = X.drop(columns=[c])

    # 3. 增强型时间处理
    if date_col in X.columns:
        print(f"⚙️ 正在处理时间特征: {date_col}")
        dt = pd.to_datetime(X[date_col])
        
        # A. 长期趋势：距离最早交易日的天数 (替代原来的 ordinal)
        X['days_since_start'] = (dt - dt.min()).dt.days
        
        # B. 周期性特征：是否周末 (流量通常更大)
        X['is_weekend'] = dt.dt.dayofweek.isin([5, 6]).astype(int)
        
        # 移除原始时间字符串
        X = X.drop(columns=[date_col])
    
    # 4. 简单的缺失值填充 (数值型填中位数，类别型填0)
    # 实际项目中应更精细，这里为了 MVP 快速跑通
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    
    return X, y

def evaluate(name, y_true, y_pred):
    """统一评估函数"""
    return {
        "model": name,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true))
    }

def plot_pred_scatter(y_true, y_pred, title, path):
    """绘制预测值 vs 真实值散点图"""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
    
    # 画对角线
    limit_min = min(y_true.min(), y_pred.min())
    limit_max = max(y_true.max(), y_pred.max())
    plt.plot([limit_min, limit_max], [limit_min, limit_max], 'r--', lw=2, label="完美预测线")
    
    plt.title(title)
    plt.xlabel("真实值 (Transformed)")
    plt.ylabel("预测值 (Transformed)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# ================= 主程序 =================

def main():
    setup_plotting_style()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--winsorize", action="store_true", help="是否对目标变量做1%-99%缩尾处理")
    parser.add_argument("--log_target", action="store_true", help="是否对目标变量取 log1p")
    parser.add_argument("--cv", action="store_true", help="是否跑交叉验证 (速度较慢)")
    args = parser.parse_args()

    # 1. 加载数据
    try:
        raw_df = load_data_auto()
    except Exception as e:
        print(f"{e}")
        return

    # 2. 特征工程
    # 根据你的数据列名配置
    TARGET_COL = "价格"
    DATE_COL = "成交时间"
    DROP_COLS = ["商品号"] # ID类无用特征
    
    X, y = build_features_v2(raw_df, TARGET_COL, DATE_COL, DROP_COLS)
    
    print(f"📊 数据准备完毕: 样本数 {X.shape[0]}, 特征数 {X.shape[1]}")

    # 3. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. 目标变量变换 (缩尾 + Log)
    y_train_work = y_train.copy()
    y_test_work = y_test.copy()
    transform_note = []

    if args.winsorize:
        lower = y_train.quantile(0.01)
        upper = y_train.quantile(0.99)
        y_train_work = y_train_work.clip(lower, upper)
        y_test_work = y_test_work.clip(lower, upper)
        transform_note.append("winsorize(1%,99%)")
    
    if args.log_target:
        y_train_work = np.log1p(y_train_work)
        y_test_work = np.log1p(y_test_work)
        transform_note.append("log1p")
    
    transform_desc = "+".join(transform_note) if transform_note else "none"
    print(f"🔄 目标变量变换: {transform_desc}")

    results = []

    # === 模型 1: Ridge (线性基线) ===
    print("\n🚀 正在训练 Ridge 回归...")
    ridge = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge())
    ])
    ridge.fit(X_train, y_train_work)
    pred_ridge = ridge.predict(X_test)
    
    res_ridge = evaluate("Ridge", y_test_work, pred_ridge)
    res_ridge['target_transform'] = transform_desc
    results.append(res_ridge)
    
    plot_pred_scatter(y_test_work, pred_ridge, 
                      f"Ridge 预测对比 (R2={res_ridge['R2']:.3f})", 
                      "outputs/figures/scatter_ridge.png")

    # === 模型 2: GBDT (优先 LightGBM，失败降级 HistGBDT) ===
    print("\n🚀 正在训练 GBDT 树模型...")
    gbdt_model = None
    model_name = "Unknown"

    try:
        import lightgbm as lgb
        print("✅ 检测到 LightGBM，正在尝试训练...")
        # 显式设置 n_jobs=1 可以缓解某些 macOS OpenMP 冲突，或者让它自动
        gbdt_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
        gbdt_model.fit(X_train, y_train_work)
        model_name = "LightGBM"
    
    except Exception as e:
        print(f"⚠️ LightGBM 启动失败 (通常是因为 macOS 缺少 libomp)。")
        print(f"   错误详情: {str(e)[:100]}...")
        print("🔄 正在降级使用 Scikit-learn 的 HistGradientBoostingRegressor (效果相近)...")
        
        from sklearn.ensemble import HistGradientBoostingRegressor
        gbdt_model = HistGradientBoostingRegressor(random_state=42)
        gbdt_model.fit(X_train, y_train_work)
        model_name = "HistGBDT(sklearn)"

    # 统一预测评估
    pred_gbdt = gbdt_model.predict(X_test)
    res_gbdt = evaluate(model_name, y_test_work, pred_gbdt)
    res_gbdt['target_transform'] = transform_desc
    results.append(res_gbdt)

    plot_pred_scatter(y_test_work, pred_gbdt, 
                      f"{model_name} 预测对比 (R2={res_gbdt['R2']:.3f})", 
                      "outputs/figures/scatter_gbdt.png")
    
    # === 输出总结 ===
    res_df = pd.DataFrame(results)
    print("\n🏆 最终成绩单:")
    print(res_df[['model', 'MAE', 'RMSE', 'R2']])
    
    res_df.to_csv("outputs/metrics/final_results.csv", index=False)
    print(f"\n✨ 运行结束！结果已保存至 outputs/metrics/final_results.csv")
    print(f"🖼️  可视化图表已保存至 outputs/figures/ 目录")

if __name__ == "__main__":
    main()

import os
import sys
import time
import argparse
import warnings
import platform
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn 核心组件
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 四大金刚：线性、SVM、随机森林、梯度提升
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# ================= 配置与初始化 =================
warnings.filterwarnings('ignore')

# 确保输出目录存在
DIRS = ["outputs/figures", "outputs/metrics"]
for d in DIRS:
    os.makedirs(d, exist_ok=True)

def setup_plotting_style():
    """根据系统自动设置中文字体"""
    system_name = platform.system()
    if system_name == "Darwin":  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
    elif system_name == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False

# ================= 数据处理核心 =================

def load_data_auto(data_dir="data"):
    """自动寻找并读取 Excel 或 CSV"""
    files = glob.glob(os.path.join(data_dir, "*.xlsx")) + glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError("❌ data/ 目录下未找到数据文件，请确认 train.xlsx 存在！")
    
    f = files[0]
    print(f"📖 正在读取数据: {os.path.basename(f)} ...")
    if f.endswith('.xlsx'):
        return pd.read_excel(f)
    return pd.read_csv(f)

def preprocess_data(df):
    """特征工程：清洗、时间特征提取、对数变换"""
    df = df.copy()
    
    # 1. 目标列处理 (Log + Winsorize)
    if '价格' not in df.columns:
        raise ValueError("❌ 数据中缺少 '价格' 列")
    
    y = df['价格'].astype(float)
    # 缩尾处理：去掉最贵和最便宜的 1% 异常值
    lower, upper = y.quantile(0.01), y.quantile(0.99)
    y = y.clip(lower, upper)
    # Log 变换：让价格分布更符合正态分布
    y = np.log1p(y)
    
    # 2. 特征列处理
    X = df.drop(columns=['价格'])
    
    # 丢弃无用列
    if '商品号' in X.columns:
        X = X.drop(columns=['商品号'])
        
    # 时间处理
    if '成交时间' in X.columns:
        dt = pd.to_datetime(X['成交时间'])
        X['is_weekend'] = dt.dt.dayofweek.isin([5, 6]).astype(int)
        # 转化为距离最早一天过去了多少天
        X['days_elapse'] = (dt - dt.min()).dt.days
        X = X.drop(columns=['成交时间'])
        
    # 填充缺失值
    X = X.fillna(X.median(numeric_only=True))
    
    return X, y

# ================= 模型竞技场 =================

def run_model_arena(X_train, X_test, y_train, y_test):
    """运行四大模型并对比"""
    setup_plotting_style()
    
    # 定义模型清单
    models = []
    
    # 1. Ridge (线性基线)
    models.append(("Ridge (基线)", Pipeline([('scaler', StandardScaler()), ('reg', Ridge())])))
    
    # 2. SVM (限制样本量，防止卡死)
    # 注意：SVM 在 M4 Pro 上也需要计算很久，所以我们只用 3000 条数据做演示
    svm_pipeline = Pipeline([('scaler', StandardScaler()), ('svr', SVR(C=10, kernel='rbf'))])
    models.append(("SVM (下采样)", svm_pipeline))
    
    # 3. Random Forest (M4 Pro 火力全开版)
    # n_jobs=-1 调用所有核心，n_estimators=500 增加精度
    rf = RandomForestRegressor(n_estimators=500, max_depth=None, n_jobs=-1, random_state=42)
    models.append(("Random Forest", rf))
    
    # 4. LightGBM (最终 Boss)
    try:
        import lightgbm as lgb
        lgbm = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbose=-1, random_state=42)
        models.append(("LightGBM", lgbm))
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        hgb = HistGradientBoostingRegressor(random_state=42)
        models.append(("HistGBDT", hgb))

    # 结果容器
    results = []
    plt.figure(figsize=(14, 10))
    
    print("\n⚔️  模型竞技场开启 (M4 Pro 加速中) ⚔️")
    print("="*50)

    for i, (name, model) in enumerate(models):
        print(f"🏃 正在训练: {name} ...")
        t0 = time.time()
        
        # 特殊处理 SVM：数据量太大跑不动，强制下采样
        if "SVM" in name and len(X_train) > 3000:
            X_train_run, y_train_run = X_train[:3000], y_train[:3000]
        else:
            X_train_run, y_train_run = X_train, y_train
            
        # 训练
        model.fit(X_train_run, y_train_run)
        
        # 预测
        pred = model.predict(X_test)
        time_cost = time.time() - t0
        
        # 评估
        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        
        # 记录
        results.append({"模型": name, "R2": r2, "RMSE": rmse, "耗时(s)": round(time_cost, 2)})
        
        # 画图 (2x2)
        plt.subplot(2, 2, i+1)
        plt.scatter(y_test, pred, alpha=0.2, s=5, c='steelblue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title(f"{name}\nR2: {r2:.3f} | RMSE: {rmse:.3f}")
        plt.xlabel("真实价格 (Log)")
        plt.ylabel("预测价格 (Log)")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/figures/model_arena_v5.png", dpi=150)
    print(f"\n📊 对比图已保存: outputs/figures/model_arena_v5.png")
    
    return pd.DataFrame(results)

# ================= 主程序 =================

if __name__ == "__main__":
    try:
        # 1. 读数据
        df_raw = load_data_auto()
        
        # 2. 预处理
        X, y = preprocess_data(df_raw)
        print(f"✅ 数据预处理完成: {X.shape} 行数据")
        
        # 3. 切分
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 4. 开战
        res_df = run_model_arena(X_train, X_test, y_train, y_test)
        
        # 5. 输出战报
        print("\n🏆 最终战报:")
        print(res_df.sort_values("R2", ascending=False).to_string(index=False))
        res_df.to_csv("outputs/metrics/final_arena_results.csv", index=False)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ 程序出错: {e}")
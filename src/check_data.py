import pandas as pd
import sys

try:
    sample_file = "data/zx/zx1026.xlsx"
    df = pd.read_excel(sample_file)

    print(f"📊 文件: {sample_file}")
    print(f"📏 形状: {df.shape}")
    print(f"\n📋 列名:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    print(f"\n📈 前 3 行数据:")
    print(df.head(3).to_string())
    print(f"\n📊 数据类型:")
    print(df.dtypes)
    print(f"\n✓ 缺失值统计:")
    print(df.isnull().sum())
    print(f"\n📊 数值列统计:")
    print(df.describe())
except Exception as e:
    print(f"❌ 错误: {e}")
    sys.exit(1)

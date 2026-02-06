import pandas as pd
import re

# 读取样本文件
sample_file = "data/zx/zx1026.xlsx"
df = pd.read_excel(sample_file)

print("=" * 80)
print("📊 分析 bigTitle 列的嵌套结构")
print("=" * 80)

# 获取 bigTitle 列
big_title = df['bigTitle'].dropna()
print(f"\n✓ bigTitle 非空行数: {len(big_title)}")

print("\n📋 前 5 个样本（完整内容）:")
for i, title in enumerate(big_title.head(5), 1):
    print(f"\n{i}. {title}")

# 分析结构
print("\n" + "=" * 80)
print("🔍 结构分析")
print("=" * 80)

# 取一个样本进行详细分析
sample = big_title.iloc[0]
print(f"\n样本: {sample}")

# 按逗号分割
parts = sample.split('，')
print(f"\n按「，」分割后的部分数 ({len(parts)}):")
for i, part in enumerate(parts, 1):
    print(f"  {i}. {part}")

# 提取键值对
print(f"\n🔑 键值对提取:")
kv_dict = {}
for part in parts:
    if ':' in part:
        key, value = part.split(':', 1)
        kv_dict[key] = value
        print(f"  '{key}' → '{value}'")

# 统计所有可能的 key
print("\n" + "=" * 80)
print("📊 全数据集中所有 key 统计")
print("=" * 80)

all_keys = set()
for title in big_title:
    parts = str(title).split('，')
    for part in parts:
        if ':' in part:
            key = part.split(':', 1)[0]
            all_keys.add(key)

print(f"\n发现的所有 key ({len(all_keys)}):")
for i, key in enumerate(sorted(all_keys), 1):
    print(f"  {i}. {key}")

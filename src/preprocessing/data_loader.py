"""
多数据集加载器
支持加载 train.xlsx 和 data/zx/ 目录下的多个数据文件
"""

import os
import glob
import pandas as pd
from typing import List, Tuple
import warnings

warnings.filterwarnings('ignore')


class MultiDatasetLoader:
    """多数据集加载与合并"""
    
    def __init__(self, base_dir: str = "."):
        """
        初始化加载器
        
        Args:
            base_dir: 项目根目录
        """
        self.base_dir = base_dir
        self.original_data = None
        self.zx_data = None
        self.combined_data = None
    
    def load_original_data(self, filepath: str = "data/train.xlsx") -> pd.DataFrame:
        """
        加载原始数据
        
        Args:
            filepath: 原始数据文件路径
            
        Returns:
            pd.DataFrame: 原始数据
        """
        full_path = os.path.join(self.base_dir, filepath)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"原始数据文件不存在: {full_path}")
        
        print(f"📖 加载原始数据: {filepath}")
        self.original_data = pd.read_excel(full_path)
        print(f"   ✓ 形状: {self.original_data.shape}")
        return self.original_data
    
    def load_zx_datasets(self, zx_dir: str = "data/zx") -> pd.DataFrame:
        """
        加载并合并 zx 目录下的所有数据文件
        
        Args:
            zx_dir: zx 数据目录
            
        Returns:
            pd.DataFrame: 合并后的 zx 数据
        """
        full_path = os.path.join(self.base_dir, zx_dir)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"zx 数据目录不存在: {full_path}")
        
        # 查找所有 xlsx 文件
        xlsx_files = glob.glob(os.path.join(full_path, "*.xlsx"))
        print(f"📂 发现 {len(xlsx_files)} 个 zx 数据文件")
        
        dfs = []
        for file_path in sorted(xlsx_files):
            filename = os.path.basename(file_path)
            try:
                df = pd.read_excel(file_path)
                dfs.append(df)
                print(f"   ✓ 加载 {filename}: {df.shape[0]} 行")
            except Exception as e:
                print(f"   ✗ 加载失败 {filename}: {e}")
        
        # 合并所有数据
        if dfs:
            self.zx_data = pd.concat(dfs, ignore_index=True)
            print(f"\n✓ zx 数据合并完成: {self.zx_data.shape}")
        else:
            raise ValueError("未成功加载任何 zx 数据文件")
        
        return self.zx_data
    
    def get_common_columns(self) -> List[str]:
        """获取两个数据集的公共列"""
        if self.original_data is None or self.zx_data is None:
            raise ValueError("请先加载原始数据和 zx 数据")
        
        common = list(set(self.original_data.columns) & set(self.zx_data.columns))
        print(f"\n📊 公共列数: {len(common)}")
        if common:
            print(f"   {common}")
        return common
    
    def combine_datasets(self, on_columns: List[str] = None) -> pd.DataFrame:
        """
        合并两个数据集
        
        Args:
            on_columns: 合并时使用的列（如果为 None，使用 concat）
            
        Returns:
            pd.DataFrame: 合并后的数据
        """
        if self.original_data is None or self.zx_data is None:
            raise ValueError("请先加载原始数据和 zx 数据")
        
        print("\n🔗 开始合并数据集...")
        
        # 简单的行级别合并（append）
        # 使用共同列和填充
        common_cols = self.get_common_columns()
        
        # 标记数据来源
        self.original_data['data_source'] = 'original'
        self.zx_data['data_source'] = 'zx'
        
        # 对齐列
        all_cols = sorted(set(list(self.original_data.columns) + list(self.zx_data.columns)))
        
        # 填充缺失列
        for col in all_cols:
            if col not in self.original_data.columns:
                self.original_data[col] = None
            if col not in self.zx_data.columns:
                self.zx_data[col] = None
        
        # 合并
        self.combined_data = pd.concat(
            [self.original_data[all_cols], self.zx_data[all_cols]],
            ignore_index=True,
            sort=False
        )
        
        print(f"✓ 合并完成: {self.combined_data.shape}")
        print(f"  原始数据: {len(self.original_data)} 行")
        print(f"  zx 数据: {len(self.zx_data)} 行")
        print(f"  合并结果: {len(self.combined_data)} 行, {len(self.combined_data.columns)} 列")
        
        return self.combined_data
    
    def get_combined_data(self) -> pd.DataFrame:
        """获取合并后的数据"""
        if self.combined_data is None:
            raise ValueError("请先调用 combine_datasets()")
        return self.combined_data


if __name__ == '__main__':
    # 测试脚本
    loader = MultiDatasetLoader(base_dir='.')
    
    print("=" * 80)
    print("🚀 多数据集加载与合并测试")
    print("=" * 80)
    
    # 加载数据
    loader.load_original_data()
    loader.load_zx_datasets()
    
    # 合并数据
    combined = loader.combine_datasets()
    
    print("\n📊 合并后的数据预览:")
    print(combined.head())
    
    print("\n✓ 数据来源分布:")
    print(combined['data_source'].value_counts())

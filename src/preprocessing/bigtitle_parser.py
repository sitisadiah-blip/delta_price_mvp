"""
BigTitle 特征解析器
从 bigTitle 列中提取结构化特征
"""

import re
import pandas as pd
from typing import Dict, Any, Optional


class BigTitleParser:
    """解析 bigTitle 字段，提取结构化特征"""
    
    # 特征提取规则
    FEATURES = [
        '总资产', '纯币资产', '等级', '训练中心', 
        '安全箱', '区服', '段位', '登录方式', '启动器'
    ]
    
    @staticmethod
    def parse_total_assets(value: str) -> Optional[float]:
        """提取总资产数字（M/K 转换）"""
        if pd.isna(value) or not isinstance(value, str):
            return None
        match = re.search(r'(\d+\.?\d*)([MK]?)', value)
        if match:
            num = float(match.group(1))
            unit = match.group(2)
            if unit == 'M':
                return num * 1_000_000
            elif unit == 'K':
                return num * 1_000
            else:
                return num
        return None
    
    @staticmethod
    def parse_level(value: str) -> Optional[int]:
        """提取等级数字"""
        if pd.isna(value) or not isinstance(value, str):
            return None
        match = re.search(r'(\d+)', str(value))
        return int(match.group(1)) if match else None
    
    @staticmethod
    def parse_safe_box(value: str) -> Dict[str, Any]:
        """解析安全箱类型与容量"""
        if pd.isna(value) or not isinstance(value, str):
            return {'box_type': None, 'capacity': None}
        
        box_type = None
        capacity = None
        
        if '顶级' in value:
            box_type = '顶级'
        elif '高级' in value:
            box_type = '高级'
        elif '普通' in value:
            box_type = '普通'
        
        match = re.search(r'(\d+)\*(\d+)', value)
        if match:
            capacity = int(match.group(1)) * int(match.group(2))
        
        return {'box_type': box_type, 'capacity': capacity}
    
    @classmethod
    def extract_features(cls, bigtitle_series: pd.Series) -> pd.DataFrame:
        """
        从 bigTitle 列提取所有特征
        
        Args:
            bigtitle_series: pandas Series，包含 bigTitle 数据
            
        Returns:
            pd.DataFrame: 提取后的特征数据框
        """
        features_dict = {
            'total_assets': [],
            'pure_coin_assets': [],
            'level': [],
            'train_center_level': [],
            'safe_box_type': [],
            'safe_box_capacity': [],
            'region': [],
            'rank': [],
            'login_method': [],
            'launcher': []
        }
        
        for bigtitle in bigtitle_series:
            if pd.isna(bigtitle):
                # 缺失值处理
                for key in features_dict.keys():
                    if 'box' in key:
                        features_dict[key].append(None)
                    else:
                        features_dict[key].append(None)
                continue
            
            # 按「，」分割
            parts = str(bigtitle).split('，')
            kv_dict = {}
            
            for part in parts:
                if ':' in part:
                    key, value = part.split(':', 1)
                    kv_dict[key.strip()] = value.strip()
            
            # 提取各特征
            features_dict['total_assets'].append(
                cls.parse_total_assets(kv_dict.get('总资产'))
            )
            features_dict['pure_coin_assets'].append(
                cls.parse_total_assets(kv_dict.get('纯币资产'))
            )
            features_dict['level'].append(
                cls.parse_level(kv_dict.get('等级'))
            )
            features_dict['train_center_level'].append(
                cls.parse_level(kv_dict.get('训练中心'))
            )
            
            # 安全箱
            safe_box = cls.parse_safe_box(kv_dict.get('安全箱', ''))
            features_dict['safe_box_type'].append(safe_box['box_type'])
            features_dict['safe_box_capacity'].append(safe_box['capacity'])
            
            # 分类特征
            features_dict['region'].append(kv_dict.get('区服'))
            features_dict['rank'].append(kv_dict.get('段位'))
            features_dict['login_method'].append(kv_dict.get('登录方式'))
            features_dict['launcher'].append(kv_dict.get('启动器'))
        
        return pd.DataFrame(features_dict)


if __name__ == '__main__':
    # 测试脚本
    import pandas as pd
    
    df = pd.read_excel('data/zx/zx1026.xlsx')
    parser = BigTitleParser()
    
    print("📊 原始 bigTitle 样本:")
    print(df['bigTitle'].head(3))
    
    print("\n" + "=" * 80)
    print("🔍 提取后的特征:")
    
    features_df = parser.extract_features(df['bigTitle'])
    print(features_df.head())
    print("\n📊 数据类型:")
    print(features_df.dtypes)
    print("\n✓ 缺失值:")
    print(features_df.isnull().sum())

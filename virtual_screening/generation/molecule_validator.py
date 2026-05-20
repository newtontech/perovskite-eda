#!/usr/bin/env python3
"""
分子验证器
验证生成分子的化学有效性和合成可行性
"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from typing import List, Dict

def validate_molecule(smiles: str) -> Dict:
    """
    验证单个分子
    
    Args:
        smiles: 分子 SMILES 字符串
    
    Returns:
        验证结果字典
    """
    mol = Chem.MolFromSmiles(smiles)
    
    if not mol:
        return {
            'is_valid': False,
            'qed_score': 0.0,
            'sa_score': 10.0,  # 最高难度
            'reason': 'Invalid SMILES'
        }
    
    # 计算类药性（QED）
    qed_score = QED.qed(mol)
    
    # 计算合成可行性评分（简化版）
    # 实际应使用 SA Score (Synthesis Accessibility Score)
    # 这里用分子量作为简化代理
    mw = Descriptors.MolWt(mol)
    sa_score = min(mw / 100.0, 10.0)  # 简化评分
    
    return {
        'is_valid': True,
        'qed_score': qed_score,
        'sa_score': sa_score,
        'reason': 'Valid'
    }

def validate_molecule_library(
    input_path: str = 'virtual_sams.csv',
    output_path: str = 'validation_results.csv'
) -> pd.DataFrame:
    """
    验证整个分子库
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
    
    Returns:
        包含验证结果的 DataFrame
    """
    print(f"🔍 验证分子库: {input_path}")
    
    # 读取分子库
    df = pd.read_csv(input_path)
    
    # 验证每个分子
    results = []
    for idx, row in df.iterrows():
        validation = validate_molecule(row['smiles'])
        
        result = {
            'mol_id': row['mol_id'],
            'smiles': row['smiles'],
            **validation
        }
        results.append(result)
    
    # 创建结果 DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存结果
    results_df.to_csv(output_path, index=False)
    
    # 统计
    valid_count = results_df['is_valid'].sum()
    total_count = len(results_df)
    
    print(f"✅ 验证完成:")
    print(f"  - 总分子数: {total_count}")
    print(f"  - 有效分子: {valid_count} ({valid_count/total_count*100:.1f}%)")
    print(f"  - 无效分子: {total_count - valid_count}")
    print(f"  - 平均 QED: {results_df['qed_score'].mean():.3f}")
    print(f"  - 平均 SA Score: {results_df['sa_score'].mean():.2f}")
    print(f"💾 结果保存到: {output_path}")
    
    return results_df

if __name__ == "__main__":
    validate_molecule_library()

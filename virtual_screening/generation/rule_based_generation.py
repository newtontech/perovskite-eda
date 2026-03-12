#!/usr/bin/env python3
"""
基于规则的虚拟 SAMs 分子生成器
"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from typing import List, Dict
import itertools

# SAMs 骨架库（简化 SMILES）
SCAFFOLDS = {
    'carbazole': 'c1ccc2c(c1)[nH]c3ccccc23',  # 咔唑
    'triphenylamine': 'c1ccc(c(c1)N(c2ccccc2)c3ccccc3)',  # 三苯胺
    'phenoxazine': 'c1ccc2c(c1)OC3=CC=CC=C3N2',  # 吩噁嗪
    'phenothiazine': 'c1ccc2c(c1)SC3=CC=CC=C3N2',  # 吩噻嗪
}

# 取代基库
SUBSTITUENTS = {
    'electron_donating': ['OMe', 'Me', 'NMe2', 'NH2', 'OH'],
    'electron_withdrawing': ['F', 'Cl', 'Br', 'CF3', 'CN', 'NO2'],
    'neutral': ['H'],
}

# 锚定基团（用于界面结合）
ANCHORING_GROUPS = {
    'phosphonic_acid': 'P(=O)(O)O',  # 磷酸基
    'carboxylic_acid': 'C(=O)O',  # 羧基
    'silane': '[Si](Cl)(Cl)Cl',  # 硅烷
    'thiol': 'S',  # 硫醇
}

def generate_virtual_sams(
    n_variations: int = 1000,
    output_path: str = 'virtual_sams.csv'
) -> pd.DataFrame:
    """
    生成虚拟 SAMs 分子库
    
    Args:
        n_variations: 生成的分子数量
        output_path: 输出文件路径
    
    Returns:
        包含虚拟分子的 DataFrame
    """
    molecules = []
    mol_id = 0
    
    print(f"🧪 生成虚拟 SAMs 分子库（目标: {n_variations} 个）...")
    
    # 遍历所有骨架
    for scaffold_name, scaffold_smiles in SCAFFOLDS.items():
        scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
        if not scaffold_mol:
            continue
        
        # 组合不同取代基
        all_substituents = (
            SUBSTITUENTS['electron_donating'] + 
            SUBSTITUENTS['electron_withdrawing']
        )
        
        # 随机选择取代基组合（简化版本）
        for i in range(n_variations // len(SCAFFOLDS)):
            # 选择取代基
            sub1 = all_substituents[i % len(all_substituents)]
            sub2 = all_substituents[(i + 1) % len(all_substituents)]
            
            # 选择锚定基团
            anchor_name = list(ANCHORING_GROUPS.keys())[i % len(ANCHORING_GROUPS)]
            anchor_smiles = ANCHORING_GROUPS[anchor_name]
            
            # 生成分子（简化：直接组合 SMILES）
            # 实际应用中需要更复杂的化学合成逻辑
            modified_smiles = f"{scaffold_smiles}{sub1}{sub2}{anchor_smiles}"
            
            # 验证分子
            mol = Chem.MolFromSmiles(modified_smiles)
            if mol:
                # 计算分子描述符
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                
                molecules.append({
                    'mol_id': mol_id,
                    'smiles': modified_smiles,
                    'scaffold': scaffold_name,
                    'substituent_1': sub1,
                    'substituent_2': sub2,
                    'anchoring_group': anchor_name,
                    'molecular_weight': mw,
                    'logp': logp,
                    'is_valid': True,
                })
                mol_id += 1
    
    # 创建 DataFrame
    df = pd.DataFrame(molecules)
    
    # 保存到文件
    df.to_csv(output_path, index=False)
    print(f"✅ 生成了 {len(df)} 个虚拟分子")
    print(f"💾 保存到: {output_path}")
    
    return df

if __name__ == "__main__":
    df = generate_virtual_sams(n_variations=1000)
    print(f"\n📊 数据统计:")
    print(df.head())

#!/usr/bin/env python3
"""
虚拟 SAMs 筛选主流程
"""
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from generation.rule_based_generation import generate_virtual_sams
from generation.molecule_validator import validate_molecule_library
from prediction.high_throughput_prediction import batch_predict
from screening.multi_objective_screening import multi_objective_screening
from analysis.molecular_interpretability import analyze_top_molecules

def main():
    """
    执行完整的虚拟筛选流程
    """
    print("🎨 SAMs 高通量虚拟筛选")
    print("=" * 70)
    print()
    
    # Phase 1: 生成虚拟分子库
    print("📋 Phase 1: 生成虚拟分子库")
    print("-" * 70)
    generate_virtual_sams(
        n_variations=1000,
        output_path='virtual_screening/molecular_library/virtual_sams.csv'
    )
    print()
    
    # Phase 2: 验证分子
    print("🔍 Phase 2: 验证分子有效性")
    print("-" * 70)
    validate_molecule_library(
        input_path='virtual_screening/molecular_library/virtual_sams.csv',
        output_path='virtual_screening/molecular_library/validation_results.csv'
    )
    print()
    
    # Phase 3: 高通量预测
    print("🔮 Phase 3: 高通量性能预测")
    print("-" * 70)
    batch_predict(
        molecules_path='virtual_screening/molecular_library/virtual_sams.csv',
        model_path='models/prediction_model.pkl',
        output_path='virtual_screening/prediction/prediction_results.csv'
    )
    print()
    
    # Phase 4: 多目标筛选
    print("🎯 Phase 4: 多目标筛选")
    print("-" * 70)
    multi_objective_screening(
        prediction_path='virtual_screening/prediction/prediction_results.csv',
        output_path='virtual_screening/screening/screened_molecules.csv',
        min_pce=20.0,
        max_sa_score=5.0,
        top_n=100
    )
    print()
    
    # Phase 5: 可解释性分析
    print("🔬 Phase 5: 分子可解释性分析")
    print("-" * 70)
    analyze_top_molecules(
        screened_path='virtual_screening/screening/screened_molecules.csv',
        output_dir='reports/virtual_screening/figures'
    )
    print()
    
    # 总结
    print("=" * 70)
    print("✅ 虚拟筛选流程完成！")
    print()
    print("📊 生成的文件:")
    print("  1. virtual_screening/molecular_library/virtual_sams.csv")
    print("  2. virtual_screening/molecular_library/validation_results.csv")
    print("  3. virtual_screening/prediction/prediction_results.csv")
    print("  4. virtual_screening/screening/screened_molecules.csv")
    print("  5. reports/virtual_screening/screening_report.md")
    print("  6. reports/virtual_screening/figures/*.png")
    print()
    print("🎉 可以查看筛选报告了解详细结果")

if __name__ == "__main__":
    main()

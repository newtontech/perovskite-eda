# PSC ML Exploration Tasks — 2026-04-30

## 并行探索任务列表

5个 kimi CLI 实例并行运行，每个执行1轮随机管线探索。

| # | Session ID | Task ID | Output | Seed | Status |
|---|-----------|---------|--------|------|--------|
| 1 | 899f083a-7e53-42a1-b36d-a5a7cef799ce (续接) | b2n9u0t1f | run_0001.json | 续接 | running |
| 2 | new | bo8g0439w | run_0002.json | 42 | running |
| 3 | new | bcxrjkj9f | run_0003.json | 123 | running |
| 4 | new | b992926n3 | run_0004.json | 456 | running |
| 5 | new | b14v1g331 | run_0005.json | 789 | running |

## 模型配置

- Engine: kimi CLI (kimi-k2.6)
- Flags: --yolo --work-dir /share/yhm/test/AutoML_EDA/hybrid_agent_exploration
- 所有Python命令开头: import warnings; warnings.filterwarnings('ignore')

## 完成后操作

1. 检查 results/exploration_logs/ 下 run_0001.json ~ run_0005.json
2. 运行: bash run_report_kimi.sh 生成8张图 + 科研报告

## 修复记录

- features/fingerprints.py: 修复 Atom Pair (rdMolDescriptors) 和 Topological Torsion API 兼容性
- src/pipeline.py: 修复 PCE 列字符串类型问题 (pd.to_numeric)
- src/pipeline.py: 补全所有评估方法 (Optuna/Grid/SHAP)
- 补全缺失模块: features/fingerprints.py, features/rdkit_descriptors.py, models/model_registry.py

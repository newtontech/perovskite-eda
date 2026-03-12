# Perovskite Database 数据字典

## 数据来源

- **来源**: [Perovskite Database](https://github.com/Jesperkemist/perovskitedatabase_data)
- **文件**: Perovskite_database_content_all_data.csv
- **版本**: Latest
- **更新日期**: 2026-03-12

## 数据集概述

钙钛矿太阳能电池性能数据库，包含文献中的实验数据和计算数据。

### 数据维度
- **总记录数**: ~50,000+
- **特征数**: 65+
- **时间范围**: 2012-2024

---

## 主要字段说明

### 1. 器件信息 (Device Information)

| 字段名 | 数据类型 | 说明 | 取值范围 |
|--------|----------|------|----------|
| `solar_cell_structure` | string | 太阳能电池结构 | n-i-p, p-i-n, etc. |
| `cell_stack_sequence` | string | 电池堆叠序列 | 文本描述 |
| `etl_stack_sequence` | string | 电子传输层序列 | 文本描述 |
| `htl_stack_sequence` | string | 空穴传输层序列 | 文本描述 |

### 2. 钙钛矿材料 (Perovskite Material)

| 字段名 | 数据类型 | 说明 | 取值范围 |
|--------|----------|------|----------|
| `perovskite_composition` | string | 钙钛矿组成 | 化学式 |
| `perovskite_band_gap` | float | 带隙 (eV) | 1.2 - 2.5 |
| `perovskite_thickness` | float | 厚度 (nm) | 100 - 1000 |
| `perovskite_deposition_method` | string | 沉积方法 | Solution, Vapor, etc. |

### 3. 性能指标 (Performance Metrics)

| 字段名 | 数据类型 | 说明 | 取值范围 |
|--------|----------|------|----------|
| `jv_reverse_scan_pce` | float | 功率转换效率 (%) | 0 - 30+ |
| `jv_reverse_scan_v_oc` | float | 开路电压 (V) | 0.8 - 1.3 |
| `jv_reverse_scan_j_sc` | float | 短路电流密度 (mA/cm²) | 15 - 30 |
| `jv_reverse_scan_ff` | float | 填充因子 | 0.5 - 0.9 |

### 4. 稳定性 (Stability)

| 字段名 | 数据类型 | 说明 | 取值范围 |
|--------|----------|------|----------|
| `stability_time_total_exposure` | float | 总暴露时间 (h) | 0 - 10000+ |
| `stability_pce_end_of_experiment` | float | 实验结束时 PCE (%) | 0 - 25 |
| `stability_test_conditions` | string | 测试条件 | 文本描述 |

### 5. 化学性质 (Chemical Properties)

| 字段名 | 数据类型 | 说明 | 取值范围 |
|--------|----------|------|----------|
| `cas_number` | string | CAS 号 | 文本 |
| `pubchem_id` | string | PubChem ID | 数字 |
| `smiles` | string | SMILES 表示式 | 文本 |
| `molecular_formula` | string | 分子式 | 文本 |
| `molecular_weight` | float | 分子量 | 0 - 1000+ |

### 6. 文献信息 (Publication Info)

| 字段名 | 数据类型 | 说明 | 取值范围 |
|--------|----------|------|----------|
| `title` | string | 论文标题 | 文本 |
| `authors` | string | 作者列表 | 文本 |
| `journal` | string | 期刊名称 | 文本 |
| `publication_date` | date | 发表日期 | 2012-2024 |
| `doi` | string | DOI | 文本 |

---

## 数据质量

### 缺失值情况
- **完整字段**: 器件结构、性能指标（主要字段）
- **部分缺失**: 稳定性数据（~60% 缺失）
- **高度缺失**: 化学性质（~40% 缺失）

### 数据清洗建议
1. 移除 PCE > 30% 的异常值（可能是错误数据）
2. 填充缺失的稳定性数据（使用插值或删除）
3. 标准化化学式格式
4. 移除重复记录

---

## 使用示例

```python
import pandas as pd

# 读取数据
df = pd.read_csv('data/raw/perovskite_database_all.csv')

# 查看基本统计
print(df.describe())

# 筛选高效器件
high_eff = df[df['jv_reverse_scan_pce'] > 20]

# 按年份统计
df['year'] = pd.to_datetime(df['publication_date']).dt.year
yearly_stats = df.groupby('year')['jv_reverse_scan_pce'].mean()
```

---

## 更新历史

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-03-12 | 1.0 | 初始版本 |

---

**维护者**: OpenClaw AI Assistant
**最后更新**: 2026-03-12

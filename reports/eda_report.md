# 钙钛矿太阳能电池数据库 - EDA 分析报告

**生成时间**: 2026-03-12 02:23:24
**数据来源**: Jesperkemist/perovskitedatabase_data
**样本数量**: 41,447

---

## 📊 性能数据概览

### PCE (光电转换效率)
| 统计量 | 值 |
|--------|-----|
| 有效样本数 | 41,447 |
| 平均值 | 12.05% |
| 最大值 | 36.20% |
| 最小值 | 0.00% |
| 中位数 | 12.75% |
| 标准差 | 5.21% |

### Voc (开路电压)
| 统计量 | 值 |
|--------|-----|
| 有效样本数 | 40,105 |
| 平均值 | 0.963 V |
| 最大值 | 4.050 V |

### Jsc (短路电流)
| 统计量 | 值 |
|--------|-----|
| 有效样本数 | 40,136 |
| 平均值 | 17.95 mA/cm² |
| 最大值 | 144.10 mA/cm² |

### FF (填充因子)
| 统计量 | 值 |
|--------|-----|
| 有效样本数 | 40,075 |
| 平均值 | 0.65% |
| 最大值 | 5.00% |

---

## 🧪 材料分布

### 钙钛矿类型 (Top 10)
```
perovskite_type
MA-Pb       28396
MAFA-Pb      7618
Cs-Pb        1883
FACs-Pb      1021
FA-Pb         730
MA-Sn         340
FA-Sn         313
MAFA-Sn       238
Cs-Other      149
MACs-Pb       146
```

### 电池架构 (Top 10)
```
Cell_architecture
nip                  28926
pin                  12479
Back contacted          25
Front contacted          7
Schottky                 5
Unknown                  4
Pn-Heterojunction        1
```

### 维度类型
```
dimension_type
3D       40169
2D         973
0D         182
2D/3D      123
```

---

## 📈 可视化图表

- `figures/performance/performance_distributions.png` - 性能分布图
- `figures/temporal/temporal_trends.png` - 时间趋势图
- `figures/materials/material_analysis.png` - 材料分析图
- `figures/performance/performance_by_material.png` - 按材料性能分析
- `figures/correlations/correlation_matrix.png` - 相关性矩阵

---

## 🔍 关键发现

1. **最高效率**: 36.20% (光电转换效率)
2. **主流材料**: MA-Pb
3. **主流架构**: nip

---

**分析工具**: OpenClaw AI Assistant

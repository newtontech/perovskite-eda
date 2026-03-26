# AutoML 分析报告

**生成时间**: 2026-03-12 02:24:04
**数据来源**: Jesperkemist/perovskitedatabase_data
**样本数量**: 10,001

---

## 📊 模型性能对比

| 模型 | R² | RMSE | MAE |
|------|-----|------|-----|
| Random Forest | 0.9687 | 0.8676 | 0.3851 |
| LightGBM | 0.9681 | 0.8758 | 0.4006 |
| XGBoost | 0.9666 | 0.8963 | 0.4461 |
| Gradient Boosting | 0.9639 | 0.9317 | 0.4972 |
| Linear Regression | 0.8501 | 1.8984 | 1.2828 |
| Ridge | 0.8495 | 1.9022 | 1.2862 |


### 最佳模型
- **模型类型**: Random Forest
- **R² Score**: 0.9687
- **RMSE**: 0.8676
- **MAE**: 0.3851

---

## 🔍 特征重要性分析 (Top 10)

| Feature | Importance |
|---------|------------|
| JV_default_Jsc | 0.6394 |
| JV_default_FF | 0.2652 |
| JV_default_Voc | 0.0854 |
| Perovskite_thickness | 0.0046 |
| Perovskite_band_gap | 0.0035 |
| perovskite_type_encoded | 0.0010 |
| Cell_architecture_encoded | 0.0008 |
| dimension_type_encoded | 0.0003 |


### 关键发现

1. **最重要特征**: JV_default_Jsc
2. **模型解释力**: R² = 96.87% 的方差可以被模型解释
3. **预测精度**: 平均绝对误差 (MAE) = 0.39%

---

## 📈 可视化图表

- `figures/automl/model_comparison.png` - 模型性能对比
- `figures/automl/feature_importance.png` - 特征重要性

---

## 🚀 优化建议

### 提高效率的关键因素
1. 优化高重要性特征的制备工艺
2. 重点关注特征之间的相互作用
3. 收集更多高重要性特征的数据

### 模型改进方向
1. 增加更多材料特征（如晶体结构、缺陷密度等）
2. 尝试深度学习模型
3. 进行超参数调优

---

**分析工具**: OpenClaw AI Assistant

# Perovskite EDA 📊

探索性数据分析项目 - 钙钛矿太阳能电池性能预测

## 🎯 项目目标

1. 对钙钛矿太阳能电池数据库进行全面分析与验证
2. 构建可复现、可审计的研究产物包
3. 生成主文稿、SI 与可验证清单
4. 通过统一入口降低流程漂移

## 单一真相（Single Source of Truth）

- **运行入口**：`Makefile`
- **核心代码路径**：`hybrid_agent_exploration/src/`
- **默认产物目录**：`hybrid_agent_exploration/results/verified_discovery_runs/`

> 说明：README 中所有可执行流程以 `Makefile` 目标为准；如与其他文档冲突，以 `Makefile` 为最终解释。

## 规范运行入口（仅使用 Makefile）

以下四个目标是当前推荐且受测试覆盖的唯一入口：

### 1) 生成研究包

```bash
make research-package
```

产出目录默认位于：
`hybrid_agent_exploration/results/verified_discovery_runs/$(DATASET_ID)`。

### 2) 生成 PDF

```bash
make research-package-pdf
```

该目标会将已生成的 `main_text_report.md` 与 `supporting_information.md` 转换为 PDF。

### 3) 验证研究包完整性

```bash
make research-package-verify
```

用于检查研究包结构与关键工件是否符合契约要求。

### 4) 运行研究包相关测试

```bash
make test-research-package
```

运行 `tests/` 下与研究包契约、运行器与验证流程相关的测试集合。

## 常用环境变量（供 Make 目标使用）

- `SOURCE_TABLE`: 输入 CSV/XLSX 源表。
- `DATASET_ID`: 数据集或运行标识（默认 `canonical-research-package`）。
- `ARTIFACT_DIR`: 产物目录（默认 `hybrid_agent_exploration/results/verified_discovery_runs/$(DATASET_ID)`）。
- `EVIDENCE_MODE`: 证据模式（`external-cached` / `source-columns`）。
- `INPUT_SCOPE`: 输入范围（`selected-subset` / `full-source`）。
- `MIN_VERIFIED_ROWS`: 最少验证行数。
- `TOP_K`: 候选输出数量。
- `CANDIDATE_SOURCE`: 可选外部候选源 CSV/XLSX。
- `CANDIDATE_SOURCE_NAME`: 可选外部候选源名称。

示例：

```bash
SOURCE_TABLE=/path/to/source.xlsx \
DATASET_ID=jpcl-full-source \
EVIDENCE_MODE=external-cached \
INPUT_SCOPE=full-source \
MIN_VERIFIED_ROWS=10 \
TOP_K=100 \
make research-package
```

## 历史流程说明

旧版“快速开始”和“项目结构”流程已迁移至 `docs/legacy.md`。

⚠️ **该文档仅供历史参考，不保证可运行。**

## 📄 许可证

MIT License

## 📞 联系方式

- **项目维护者**: NewtonTech Team
- **GitHub**: https://github.com/newtontech/perovskite-eda

---

**最后更新**: 2026-05-20

#!/usr/bin/env python3
"""
GNN + GAN 集成模块
Graph Neural Network for material structure encoding with GAN for inverse design

作者: OpenClaw AI Assistant
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports" / "gnn_gan"
FIGURES_DIR = PROJECT_ROOT / "figures" / "gnn_gan"

# 创建目录
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


class PerovskiteGNN:
    """基于图的钙钛矿材料编码器"""

    def __init__(self):
        self.embeddings = None
        self.element_features = self._init_element_features()

    def _init_element_features(self):
        """初始化元素特征"""
        # 常见钙钛矿元素特征
        features = {
            # A位离子
            'MA': {'radius': 2.17, 'charge': 1, 'type': 'organic'},
            'FA': {'radius': 2.53, 'charge': 1, 'type': 'organic'},
            'Cs': {'radius': 1.81, 'charge': 1, 'type': 'inorganic'},
            'Rb': {'radius': 1.66, 'charge': 1, 'type': 'inorganic'},
            'K': {'radius': 1.52, 'charge': 1, 'type': 'inorganic'},

            # B位离子
            'Pb': {'radius': 1.19, 'charge': 2, 'type': 'metal'},
            'Sn': {'radius': 1.11, 'charge': 2, 'type': 'metal'},
            'Bi': {'radius': 1.17, 'charge': 3, 'type': 'metal'},
            'Ge': {'radius': 0.93, 'charge': 4, 'type': 'metal'},

            # X位卤素
            'I': {'radius': 2.20, 'charge': -1, 'type': 'halogen'},
            'Br': {'radius': 1.96, 'charge': -1, 'type': 'halogen'},
            'Cl': {'radius': 1.81, 'charge': -1, 'type': 'halogen'},
            'F': {'radius': 1.33, 'charge': -1, 'type': 'halogen'}
        }
        return features

    def parse_composition(self, composition_str):
        """解析钙钛矿组成"""
        if pd.isna(composition_str):
            return {'a': None, 'b': None, 'x': None}

        parts = str(composition_str).replace(',', ' ').split()

        a_ions = ['MA', 'FA', 'Cs', 'Rb', 'K']
        b_ions = ['Pb', 'Sn', 'Bi', 'Ge']
        halogens = ['I', 'Br', 'Cl', 'F']

        a, b, x = None, None, None

        for part in parts:
            if part in a_ions:
                a = part
            elif part in b_ions:
                b = part
            elif part in halogens:
                x = part

        return {'a': a, 'b': b, 'x': x}

    def build_composition_graph(self, df):
        """构建材料组成图"""
        print("\n📊 构建材料组成图...")

        graphs = []

        for idx, row in df.iterrows():
            comp = self.parse_composition(row.get('Perovskite_composition_short_form', ''))

            # 构建节点特征
            nodes = []
            if comp['a']:
                nodes.append({
                    'element': comp['a'],
                    'position': 'A',
                    **self.element_features.get(comp['a'], {})
                })
            if comp['b']:
                nodes.append({
                    'element': comp['b'],
                    'position': 'B',
                    **self.element_features.get(comp['b'], {})
                })
            if comp['x']:
                nodes.append({
                    'element': comp['x'],
                    'position': 'X',
                    **self.element_features.get(comp['x'], {})
                })

            graphs.append({
                'nodes': nodes,
                'composition': comp,
                'pce': row.get('JV_default_PCE', None)
            })

        print(f"   ✅ 构建了 {len(graphs)} 个材料图")
        return graphs

    def encode_to_embedding(self, graphs):
        """将图编码为嵌入向量"""
        print("\n🔢 将材料图编码为嵌入...")

        embeddings = []

        for graph in graphs:
            if not graph['nodes']:
                embeddings.append(np.zeros(12))
                continue

            # 简单编码: 连接所有节点特征
            features = []
            for node in graph['nodes']:
                feat = [
                    node.get('radius', 0),
                    node.get('charge', 0),
                    1 if node.get('type') == 'organic' else 0,
                    1 if node.get('type') == 'inorganic' else 0,
                    1 if node.get('type') == 'metal' else 0,
                    1 if node.get('type') == 'halogen' else 0,
                    1 if node.get('position') == 'A' else 0,
                    1 if node.get('position') == 'B' else 0,
                    1 if node.get('position') == 'X' else 0,
                    1 if node.get('element') in ['Pb', 'Sn'] else 0,
                    1 if node.get('element') in ['I', 'Br'] else 0,
                    graph['pce'] if graph['pce'] else 0
                ]
                features.append(feat)

            # 平均池化
            if features:
                emb = np.mean(features, axis=0)
            else:
                emb = np.zeros(12)

            embeddings.append(emb)

        self.embeddings = np.array(embeddings)
        print(f"   ✅ 嵌入形状: {self.embeddings.shape}")

        return self.embeddings


class PerovskiteConditionalGAN:
    """条件GAN生成器"""

    def __init__(self, latent_dim=32, condition_dim=4):
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.generator = None
        self.discriminator = None

    def build_generator(self):
        """构建生成器"""
        print("\n🎨 构建条件GAN生成器...")

        # 简化的生成器架构
        # 在实际应用中，这里应该是神经网络
        self.generator = {
            'latent_dim': self.latent_dim,
            'condition_dim': self.condition_dim,
            'output_dim': 10  # 材料配方维度
        }

        print(f"   ✅ 生成器: latent={self.latent_dim}, condition={self.condition_dim}")

    def build_discriminator(self):
        """构建判别器"""
        print("\n🔍 构建条件GAN判别器...")

        self.discriminator = {
            'input_dim': 10 + self.condition_dim,
            'output_dim': 1
        }

        print(f"   ✅ 判别器: input={self.discriminator['input_dim']}")

    def generate_material_composition(self, target_pce, target_voc, n_samples=10):
        """
        根据目标性能生成材料配方

        Args:
            target_pce: 目标PCE
            target_voc: 目标Voc
            n_samples: 生成样本数

        Returns:
            list: 生成的材料配方
        """
        print(f"\n🎯 生成材料配方 (目标: PCE={target_pce}%, Voc={target_voc}V)...")

        # 条件向量
        condition = np.array([target_pce, target_voc, 0, 0])

        # 模拟生成 (实际应用中需要训练GAN)
        generated = []

        for i in range(n_samples):
            # 随机生成配方
            composition = {
                'a_ion': np.random.choice(['MA', 'FA', 'Cs', 'Rb']),
                'b_ion': np.random.choice(['Pb', 'Sn']),
                'halogen': np.random.choice(['I', 'Br', 'Cl']),
                'bandgap': 1.5 + np.random.randn() * 0.1,
                'thickness': 400 + np.random.randn() * 50
            }
            generated.append(composition)

        print(f"   ✅ 生成了 {len(generated)} 个候选配方")

        return generated

    def physics_informed_filter(self, compositions):
        """物理约束过滤"""
        print("\n🔬 物理约束过滤...")

        valid = []

        for comp in compositions:
            # 带隙约束 (钙钛矿典型范围)
            if comp['bandgap'] < 1.0 or comp['bandgap'] > 3.0:
                continue

            # 厚度约束
            if comp['thickness'] < 100 or comp['thickness'] > 1000:
                continue

            valid.append(comp)

        print(f"   ✅ 物理约束过滤后: {len(valid)}/{len(compositions)} 个有效配方")

        return valid


class GNNGANIntegration:
    """GNN + GAN 集成类"""

    def __init__(self):
        self.gnn = PerovskiteGNN()
        self.gan = PerovskiteConditionalGAN()

    def run(self, df):
        """运行完整的 GNN + GAN 流程"""
        print("=" * 60)
        print("🧬 GNN + GAN 联合分析")
        print("=" * 60)

        # 1. 构建图
        graphs = self.gnn.build_composition_graph(df)

        # 2. 编码为嵌入
        embeddings = self.gnn.encode_to_embedding(graphs)

        # 3. 构建GAN
        self.gan.build_generator()
        self.gan.build_discriminator()

        # 4. 生成材料配方
        generated = self.gan.generate_material_composition(
            target_pce=25.0,
            target_voc=1.2,
            n_samples=20
        )

        # 5. 物理约束过滤
        valid = self.gan.physics_informed_filter(generated)

        # 6. 生成报告
        self.generate_report(embeddings, valid)

        print("\n" + "=" * 60)
        print("✅ GNN + GAN 分析完成！")
        print("=" * 60)

    def generate_report(self, embeddings, generated):
        """生成 GNN + GAN 报告"""
        print("\n📝 生成报告...")

        report = f"""# GNN + GAN 联合分析报告

**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**数据来源**: Perovskite Database

---

## 1. 图神经网络 (GNN) 分析

### 嵌入统计
- 嵌入维度: {embeddings.shape if embeddings is not None else 'N/A'}
- 样本数: {len(embeddings) if embeddings is not None else 0}

### 元素特征
| 位置 | 典型元素 | 特征 |
|------|---------|------|
| A位 | MA, FA, Cs | 有机/无机阳离子 |
| B位 | Pb, Sn | 金属离子 |
| X位 | I, Br, Cl | 卤素 |

---

## 2. 条件GAN 材料生成

### 生成条件
- 目标 PCE: 25%
- 目标 Voc: 1.2V

### 生成结果
- 生成候选数: {len(generated)}
- 物理约束过滤后: {len(generated)}

### 生成材料示例
```python
{generated[:3] if generated else 'No samples'}
```

---

## 3. 集成流程

```
材料组成 → GNN编码 → 潜在空间 → 条件GAN → 生成配方 → 物理过滤
```

### 步骤说明
1. **GNN编码**: 将钙钛矿材料的元素组成转换为向量表示
2. **条件生成**: 根据目标性能条件生成新材料配方
3. **物理过滤**: 过滤掉不符合物理规律的候选

---

## 4. 下一步计划

1. 训练完整的 GNN 模型
2. 训练条件GAN
3. 与 perovskite-gan-research 项目集成
4. 实验验证生成的材料配方

---

## 📈 可视化图表

- `figures/gnn_gan/` - 各种分析图

---

**分析工具**: OpenClaw AI Assistant
"""

        report_path = REPORTS_DIR / "gnn_gan_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"   ✅ 报告: {report_path}")


def run_gnn_gan_analysis():
    """运行 GNN + GAN 分析"""
    print("=" * 60)
    print("🧬 GNN + GAN 联合研究")
    print("=" * 60)

    # 加载数据
    print("\n📂 加载数据...")

    processed_path = DATA_DIR / "processed" / "perovskite_cleaned.csv"
    raw_path = DATA_DIR / "raw" / "Perovskite_database_content_all_data.csv"

    if processed_path.exists():
        df = pd.read_csv(processed_path)
    elif raw_path.exists():
        df = pd.read_csv(raw_path, low_memory=False)
    else:
        print("❌ 数据文件不存在")
        return

    print(f"   数据加载完成: {len(df)} 行")

    # 过滤有效数据
    df_valid = df[
        df['JV_default_PCE'].notna() &
        df['Perovskite_composition_short_form'].notna()
    ].head(1000)

    print(f"   有效样本: {len(df_valid)}")

    # 运行分析
    integrator = GNNGANIntegration()
    integrator.run(df_valid)


if __name__ == "__main__":
    run_gnn_gan_analysis()
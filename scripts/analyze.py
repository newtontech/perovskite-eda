#!/usr/bin/env python3
"""
钙钛矿太阳能电池数据库 - 探索性数据分析 (EDA)
Exploratory Data Analysis for Perovskite Solar Cells Database

生成时间: 2026-03-11
作者: OpenClaw AI Assistant
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import Counter
import re
from wordcloud import WordCloud
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

# 创建目录
for dir_path in [DATA_DIR / "raw", DATA_DIR / "processed", 
                 FIGURES_DIR / "temporal", FIGURES_DIR / "materials",
                 FIGURES_DIR / "performance", REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

class PerovskiteEDA:
    """钙钛矿数据库 EDA 分析类"""
    
    def __init__(self, data_path):
        """初始化"""
        self.data_path = Path(data_path)
        self.df = None
        self.report_lines = []
        
    def load_data(self):
        """加载数据"""
        print(f"📂 加载数据: {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        
        # 读取 Excel 文件
        self.df = pd.read_excel(self.data_path)
        print(f"✅ 数据加载成功: {len(self.df)} 行, {len(self.df.columns)} 列")
        
        # 添加到报告
        self.add_to_report("## 📊 数据概览")
        self.add_to_report(f"- **总记录数**: {len(self.df):,}")
        self.add_to_report(f"- **字段数**: {len(self.df.columns)}")
        self.add_to_report(f"- **字段列表**: {', '.join(self.df.columns.tolist())}")
        self.add_to_report("")
        
    def basic_statistics(self):
        """基础统计分析"""
        print("\n🔍 基础统计分析...")
        
        self.add_to_report("## 📈 基础统计")
        
        # 数据类型统计
        dtype_counts = self.df.dtypes.value_counts()
        self.add_to_report("### 数据类型分布")
        self.add_to_report("```")
        for dtype, count in dtype_counts.items():
            self.add_to_report(f"{dtype}: {count} 列")
        self.add_to_report("```\n")
        
        # 缺失值统计
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        
        self.add_to_report("### 缺失值统计")
        self.add_to_report("| 字段 | 缺失数量 | 缺失率 |")
        self.add_to_report("|------|---------|--------|")
        for col in self.df.columns:
            if missing[col] > 0:
                self.add_to_report(f"| {col} | {missing[col]:,} | {missing_pct[col]}% |")
        self.add_to_report("")
        
    def temporal_analysis(self):
        """时间趋势分析"""
        print("\n📅 时间趋势分析...")
        
        self.add_to_report("## 🕐 时间趋势分析")
        
        # 尝试识别年份字段
        year_cols = [col for col in self.df.columns if 'year' in col.lower() or 'date' in col.lower()]
        
        if not year_cols:
            print("⚠️ 未找到年份字段")
            self.add_to_report("⚠️ 未找到年份字段\n")
            return
        
        year_col = year_cols[0]
        print(f"   使用字段: {year_col}")
        
        # 提取年份
        self.df['year'] = pd.to_datetime(self.df[year_col], errors='coerce').dt.year
        year_counts = self.df['year'].dropna().astype(int).value_counts().sort_index()
        
        # 绘制趋势图
        fig, ax = plt.subplots(figsize=(12, 6))
        year_counts.plot(kind='bar', ax=ax, color='steelblue', alpha=0.8)
        ax.set_title('钙钛矿太阳能电池研究文献发表趋势', fontsize=16, fontweight='bold')
        ax.set_xlabel('年份', fontsize=12)
        ax.set_ylabel('文献数量', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "temporal" / "publication_trend.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ 图表已保存: figures/temporal/publication_trend.png")
        
        # 添加到报告
        self.add_to_report("### 发表趋势")
        self.add_to_report("```")
        for year, count in year_counts.tail(10).items():
            self.add_to_report(f"{year}: {count:,} 篇")
        self.add_to_report("```\n")
        
        # 增长率分析
        if len(year_counts) > 1:
            growth_rates = year_counts.pct_change().dropna() * 100
            avg_growth = growth_rates.mean()
            
            self.add_to_report(f"**平均年增长率**: {avg_growth:.2f}%")
            self.add_to_report("")
        
    def keyword_analysis(self):
        """关键词分析"""
        print("\n🔑 关键词分析...")
        
        self.add_to_report("## 🔍 关键词分析")
        
        # 查找标题或摘要字段
        text_cols = [col for col in self.df.columns if any(x in col.lower() for x in ['title', 'abstract', 'keyword'])]
        
        if not text_cols:
            print("⚠️ 未找到文本字段")
            self.add_to_report("⚠️ 未找到文本字段\n")
            return
        
        # 合并所有文本
        text_data = self.df[text_cols].fillna('').astype(str).agg(' '.join, axis=1)
        all_text = ' '.join(text_data.tolist())
        
        # 关键词列表
        keywords = [
            'perovskite', 'solar cell', 'photovoltaic', 'efficiency', 'stability',
            'lead-free', 'tin', 'germanium', 'double perovskite', '2D perovskite',
            '3D perovskite', 'mixed cation', 'mixed halide', 'interface',
            'hole transport', 'electron transport', 'encapsulation',
            'flexible', 'tandem', 'quantum dot', 'nanocrystal'
        ]
        
        # 统计关键词频率
        keyword_counts = {}
        for keyword in keywords:
            count = all_text.lower().count(keyword)
            if count > 0:
                keyword_counts[keyword] = count
        
        # 排序
        keyword_counts = dict(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True))
        
        # 绘制条形图
        if keyword_counts:
            fig, ax = plt.subplots(figsize=(12, 8))
            pd.Series(keyword_counts).head(15).plot(kind='barh', ax=ax, color='coral', alpha=0.8)
            ax.set_title('钙钛矿研究热门关键词 (Top 15)', fontsize=16, fontweight='bold')
            ax.set_xlabel('出现次数', fontsize=12)
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "keyword_frequency.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ 图表已保存: figures/keyword_frequency.png")
        
        # 添加到报告
        self.add_to_report("### 热门关键词 (Top 10)")
        self.add_to_report("```")
        for i, (keyword, count) in enumerate(list(keyword_counts.items())[:10], 1):
            self.add_to_report(f"{i}. {keyword}: {count:,} 次")
        self.add_to_report("```\n")
        
    def generate_report(self):
        """生成报告"""
        print("\n📝 生成报告...")
        
        report_path = REPORTS_DIR / "eda_report.md"
        
        # 报告头部
        header = """# 钙钛矿太阳能电池数据库 - 探索性数据分析报告

**生成时间**: 2026-03-11  
**数据来源**: `20250623_crossref.xlsx`  
**分析工具**: OpenClaw AI Assistant

---

"""
        
        # 写入文件
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write('\n'.join(self.report_lines))
        
        print(f"   ✅ 报告已生成: {report_path}")
        
    def add_to_report(self, line):
        """添加行到报告"""
        self.report_lines.append(line)
        
    def run(self):
        """运行完整分析"""
        print("=" * 60)
        print("🔬 钙钛矿太阳能电池数据库 - EDA 分析")
        print("=" * 60)
        
        try:
            self.load_data()
            self.basic_statistics()
            self.temporal_analysis()
            self.keyword_analysis()
            self.generate_report()
            
            print("\n" + "=" * 60)
            print("✅ EDA 分析完成！")
            print("=" * 60)
            print(f"\n📊 查看报告: {REPORTS_DIR / 'eda_report.md'}")
            print(f"📈 查看图表: {FIGURES_DIR}/")
            
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # 数据路径
    data_path = DATA_DIR / "raw" / "crossref.xlsx"
    
    # 如果本地数据不存在，提示下载
    if not data_path.exists():
        print("⚠️ 数据文件不存在！")
        print(f"   预期路径: {data_path}")
        print("\n请先运行下载脚本:")
        print("   python scripts/download_data.py")
        exit(1)
    
    # 运行分析
    eda = PerovskiteEDA(data_path)
    eda.run()

#!/usr/bin/env python3
"""
下载钙钛矿数据库
Download Perovskite Database
"""

import requests
from pathlib import Path
from tqdm import tqdm
import sys

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 数据源
DATA_URL = "https://raw.githubusercontent.com/newtontech/perovskite_literature_rag/main/20250623_crossref.xlsx"
OUTPUT_PATH = DATA_DIR / "crossref.xlsx"

def download_with_progress(url, output_path):
    """带进度条的下载"""
    print(f"📥 下载文件...")
    print(f"   URL: {url}")
    print(f"   保存到: {output_path}")
    
    response = requests.get(url, stream=True, timeout=60)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="下载进度") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"✅ 下载完成！文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

def main():
    """主函数"""
    print("=" * 60)
    print("📥 钙钛矿数据库下载工具")
    print("=" * 60)
    
    try:
        download_with_progress(DATA_URL, OUTPUT_PATH)
        print("\n✅ 所有文件下载成功！")
        print(f"\n运行分析: python scripts/analyze.py")
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

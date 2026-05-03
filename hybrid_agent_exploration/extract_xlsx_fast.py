#!/usr/bin/env python3
"""Fast extract needed columns from huge xlsx internal XML using per-column mmap scan."""
import warnings
warnings.filterwarnings('ignore')
import mmap
import re
from pathlib import Path
import pandas as pd

XLSX_XML = "/tmp/xlsx_extract/xl/worksheets/sheet1.xml"
OUT_PKL = Path("data_cache.pkl")
OUT_CSV = Path("data_cache.csv")

COLS = {
    "R": "jv_reverse_scan_pce_without_modulator",
    "V": "jv_reverse_scan_pce",
    "BA": "smiles",
}

def extract_col(data, col_prefix):
    """Extract all cells for a given column prefix (e.g., b'R' or b'BA').
    Returns dict: row_number -> value
    """
    results = {}
    prefix = b'<c r="' + col_prefix
    offset = 0
    while True:
        idx = data.find(prefix, offset)
        if idx == -1:
            break
        # Parse row number after prefix
        j = idx + len(prefix)
        row_num = b""
        while j < len(data) and data[j:j+1].isdigit():
            row_num += data[j:j+1]
            j += 1
        # Find value: look for </c> first to bound search
        c_close = data.find(b'</c>', j)
        if c_close == -1:
            break
        val = b""
        # Try <v>...</v>
        v_open = data.find(b'<v>', j)
        if v_open != -1 and v_open < c_close:
            v_close = data.find(b'</v>', v_open)
            if v_close != -1 and v_close < c_close:
                val = data[v_open+3:v_close]
        else:
            # Try <t>...</t>
            t_open = data.find(b'<t>', j)
            if t_open != -1 and t_open < c_close:
                t_close = data.find(b'</t>', t_open)
                if t_close != -1 and t_close < c_close:
                    val = data[t_open+3:t_close]
        if row_num:
            results[row_num.decode('utf-8', errors='ignore')] = val.decode('utf-8', errors='ignore')
        offset = c_close + 4
    return results

def extract():
    with open(XLSX_XML, 'rb') as f:
        data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        col_data = {}
        for prefix, name in [(b"R", "R"), (b"V", "V"), (b"BA", "BA")]:
            print(f"Extracting {name}...")
            col_data[name] = extract_col(data, prefix)
        data.close()

    # Merge by row number
    all_rows = set(col_data["R"].keys()) & set(col_data["V"].keys()) & set(col_data["BA"].keys())
    data_list = []
    for row_num in sorted(all_rows, key=lambda x: int(x)):
        data_list.append({
            "jv_reverse_scan_pce_without_modulator": col_data["R"][row_num],
            "jv_reverse_scan_pce": col_data["V"][row_num],
            "smiles": col_data["BA"][row_num],
        })

    df = pd.DataFrame(data_list)
    for col in ["jv_reverse_scan_pce_without_modulator", "jv_reverse_scan_pce"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["delta_pce"] = df["jv_reverse_scan_pce"] - df["jv_reverse_scan_pce_without_modulator"]
    df = df.dropna(subset=["smiles", "delta_pce"])
    df.to_pickle(OUT_PKL)
    df.to_csv(OUT_CSV, index=False)
    print(f"Extracted {len(df)} rows, saved to {OUT_PKL} and {OUT_CSV}")

if __name__ == "__main__":
    extract()

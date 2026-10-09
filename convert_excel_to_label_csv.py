"""
将 Excel 数据清单转换为 3D 预处理器需要的 label CSV 格式

输入 Excel: tr_path | label
输出 CSV:   filename | fault_type

用法:
    python convert_excel_to_label_csv.py \
        --input "D:\data\08014_sc_generatorspeedfromconvertererror.xlsx" \
        --output "D:\data\08014_labels.csv"
"""
import pandas as pd
import argparse


def convert(input_path, output_path, path_col='tr_path', label_col='label'):
    df = pd.read_excel(input_path)
    if path_col not in df.columns or label_col not in df.columns:
        print(f"错误: 未找到列 '{path_col}' 或 '{label_col}'，当前列: {df.columns.tolist()}")
        return
    df_out = pd.DataFrame({'filename': df[path_col], 'fault_type': df[label_col]})
    df_out = df_out.dropna()
    df_out.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"转换完成: {len(df_out)} 条 → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--path_col", default="tr_path")
    parser.add_argument("--label_col", default="label")
    args = parser.parse_args()
    convert(args.input, args.output, args.path_col, args.label_col)

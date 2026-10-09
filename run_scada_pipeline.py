# -*- coding: utf-8 -*-
"""
SCADA数据处理管道运行脚本

使用方法:
    python run_scada_pipeline.py --config configs/preprocess_scada.yaml

功能:
    1. 读取配置文件中的输入文件(source_excel)
    2. 批量处理SCADA数据并划分数据集
    3. 输出 PyTorch 张量文件(.pt):
       - train.pt
       - val.pt
       - test.pt
       - full_dataset.pt
"""

import argparse
import traceback
import torch
from data_preprocessing.scada_pipeline import SCADA_Pipeline


def progress_callback(current, total, message):
    """进度回调函数"""
    if current % 10 == 0 or current == total:
        percent = (current / total) * 100
        print(f"  [{current}/{total}] {percent:.1f}% - {message}")


def verify_pt_file(file_path, file_name):
    """验证 .pt 文件内容"""
    try:
        data = torch.load(file_path, map_location='cpu')
        print(f"   - {file_name}: {file_path}")

        if isinstance(data, dict):
            if 'samples' in data:
                print(f"     samples shape: {tuple(data['samples'].shape)}")
            if 'labels' in data:
                print(f"     labels shape : {tuple(data['labels'].shape)}")
            if 'X' in data:
                print(f"     X shape      : {tuple(data['X'].shape)}")
            if 'y' in data:
                print(f"     y shape      : {tuple(data['y'].shape)}")
        else:
            print(f"     内容类型: {type(data)}")
    except Exception as e:
        print(f"   - ⚠️ {file_name} 验证失败: {e}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='SCADA数据处理管道')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/preprocess_scada.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--source',
        type=str,
        default=None,
        help='输入文件路径(可选,覆盖配置文件中的设置)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SCADA数据处理管道")
    print("=" * 60)
    print(f"配置文件: {args.config}\n")

    # 创建管道实例
    try:
        pipeline = SCADA_Pipeline(config_path=args.config)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    # 显示配置信息
    print("📋 配置信息:")
    print(f"  - 时间窗口: {pipeline.config['time_window']}")
    print(f"  - 默认标签点: {pipeline.default_tags}")
    print(f"  - 标签点数量: {len(pipeline.default_tags)}")

    data_split = pipeline.config.get('data_split', {})
    print(f"  - 测试集比例: {data_split.get('test_size', 0.1)}")
    print(f"  - 验证集比例: {data_split.get('val_size', 0.111)}")
    print(f"  - 分层采样: {data_split.get('stratify', True)}")

    # 批量处理
    print("\n" + "=" * 60)
    print("开始批量处理...")
    print("=" * 60 + "\n")

    try:
        result = pipeline.process_batch(
            source_excel=args.source,
            progress_callback=progress_callback
        )
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        traceback.print_exc()
        return

    # 显示处理结果
    print("\n" + "=" * 60)
    print("处理结果统计")
    print("=" * 60)

    if result['status'] == 'success':
        print("✅ 状态: 成功")
        print(f"📊 总样本数: {result['total']}")
        print(f"   - 成功: {result['success']}")
        print(f"   - 失败: {result['failed']}")

        if 'full_size' in result:
            print(f"\n📦 完整数据集:")
            print(f"   - 全量样本: {result['full_size']} 样本")

        print(f"\n📦 数据集划分:")
        print(f"   - 训练集: {result['train_size']} 样本")
        print(f"   - 验证集: {result['val_size']} 样本")
        print(f"   - 测试集: {result['test_size']} 样本")

        print(f"\n💾 输出文件:")
        if 'full_file' in result:
            print(f"   - 完整数据集: {result['full_file']}")
        print(f"   - 训练集: {result['train_file']}")
        print(f"   - 验证集: {result['val_file']}")
        print(f"   - 测试集: {result['test_file']}")

        print(f"\n🔍 文件验证:")
        if 'full_file' in result:
            verify_pt_file(result['full_file'], '完整数据集')
        verify_pt_file(result['train_file'], '训练集')
        verify_pt_file(result['val_file'], '验证集')
        verify_pt_file(result['test_file'], '测试集')

    else:
        print("❌ 状态: 失败")
        print(f"📊 总样本数: {result.get('total', 0)}")
        print(f"   - 失败: {result.get('failed', 0)}")
        if 'message' in result:
            print(f"   - 原因: {result['message']}")

    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

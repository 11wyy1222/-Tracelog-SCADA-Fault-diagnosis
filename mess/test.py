import torch

# 加载两个文件
file1 = torch.load('external_data/prediction_features.pt', weights_only=False)
file2 = torch.load('external_data/sample_data_without_labels.pt', weights_only=False)

print("=" * 80)
print("文件1: prediction_features.pt")
print("=" * 80)
print(f"数据类型: {type(file1)}")
if isinstance(file1, dict):
    print(f"字典的键: {list(file1.keys())}")
    for key, value in file1.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: type={type(value)}, value={value if not hasattr(value, '__len__') or len(str(value)) < 100 else f'{str(value)[:100]}...'}")
else:
    print(f"数据内容: {file1}")

print("\n" + "=" * 80)
print("文件2: sample_data_without_labels.pt")
print("=" * 80)
print(f"数据类型: {type(file2)}")
if isinstance(file2, dict):
    print(f"字典的键: {list(file2.keys())}")
    for key, value in file2.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: type={type(value)}, value={value if not hasattr(value, '__len__') or len(str(value)) < 100 else f'{str(value)[:100]}...'}")
else:
    print(f"数据内容: {file2}")

print("\n" + "=" * 80)
print("格式对比")
print("=" * 80)

# 检查是否都是字典
if isinstance(file1, dict) and isinstance(file2, dict):
    print("✓ 两个文件都是字典格式")
    
    # 检查键是否相同
    keys1 = set(file1.keys())
    keys2 = set(file2.keys())
    
    print(f"\n文件1的键: {sorted(keys1)}")
    print(f"文件2的键: {sorted(keys2)}")
    
    common_keys = keys1 & keys2
    only_in_file1 = keys1 - keys2
    only_in_file2 = keys2 - keys1
    
    if common_keys:
        print(f"\n共同的键: {sorted(common_keys)}")
    if only_in_file1:
        print(f"仅在文件1中: {sorted(only_in_file1)}")
    if only_in_file2:
        print(f"仅在文件2中: {sorted(only_in_file2)}")
    
    # 检查samples字段
    if 'samples' in file1 and 'samples' in file2:
        print(f"\n'samples' 字段对比:")
        print(f"  文件1: shape={file1['samples'].shape}, dtype={file1['samples'].dtype}")
        print(f"  文件2: shape={file2['samples'].shape}, dtype={file2['samples'].dtype}")
        print(f"  形状是否相同: {file1['samples'].shape[1:] == file2['samples'].shape[1:]}")
    
    # 检查labels字段
    has_labels_1 = 'labels' in file1
    has_labels_2 = 'labels' in file2
    
    print(f"\n标签字段:")
    print(f"  文件1有labels: {has_labels_1}")
    print(f"  文件2有labels: {has_labels_2}")
    
    if has_labels_1:
        labels1 = file1['labels']
        if hasattr(labels1, 'shape'):
            print(f"  文件1 labels: shape={labels1.shape}, dtype={labels1.dtype}")
            print(f"  文件1 标签范围: [{labels1.min()}, {labels1.max()}]")
        else:
            print(f"  文件1 labels: {labels1}")
    
    if has_labels_2:
        labels2 = file2['labels']
        if hasattr(labels2, 'shape'):
            print(f"  文件2 labels: shape={labels2.shape}, dtype={labels2.dtype}")
            print(f"  文件2 labels范围: [{labels2.min()}, {labels2.max()}]")
        else:
            print(f"  文件2 labels: {labels2}")
    
else:
    print("✗ 两个文件的数据类型不同")
    print(f"  文件1类型: {type(file1)}")
    print(f"  文件2类型: {type(file2)}")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)

if isinstance(file1, dict) and isinstance(file2, dict):
    if 'samples' in file1 and 'samples' in file2:
        same_structure = file1['samples'].shape[1:] == file2['samples'].shape[1:]
        if same_structure:
            print("✓ 两个文件的数据格式基本相同")
            print("  - 都包含 'samples' 字段")
            print(f"  - samples 的特征维度相同: {file1['samples'].shape[1:]}")
            
            if has_labels_1 and not has_labels_2:
                print("  - 主要区别: 文件1有标签，文件2无标签")
            elif not has_labels_1 and has_labels_2:
                print("  - 主要区别: 文件2有标签，文件1无标签")
            elif has_labels_1 and has_labels_2:
                print("  - 两个文件都有标签")
            else:
                print("  - 两个文件都没有标签")
        else:
            print("✗ 两个文件的数据维度不同")
            print(f"  文件1 samples维度: {file1['samples'].shape}")
            print(f"  文件2 samples维度: {file2['samples'].shape}")
    else:
        print("✗ 至少有一个文件缺少 'samples' 字段")
else:
    print("✗ 数据格式不兼容")
"""
故障诊断映射模块
从Excel表格读取标签-故障原因-处理方案的映射关系
"""
import pandas as pd
import os
from typing import Dict, Optional


class FaultDiagnosisMapper:
    """故障诊断映射器"""
    
    def __init__(self, mapping_file: str):
        """
        初始化故障诊断映射器
        
        Args:
            mapping_file: Excel映射文件路径
        """
        self.mapping_file = mapping_file
        self.mapping_dict = {}
        self._load_mapping()
    
    def _load_mapping(self):
        """从Excel文件加载映射关系"""
        if not os.path.exists(self.mapping_file):
            print(f"警告: 映射文件不存在: {self.mapping_file}")
            print("将无法提供故障诊断信息")
            return
        
        try:
            # 读取Excel文件
            df = pd.read_excel(self.mapping_file)
            
            # 检查必需的列
            required_columns = ['标签', '故障原因', '处理方案']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"警告: Excel文件缺少必需的列: {missing_columns}")
                print(f"当前列名: {df.columns.tolist()}")
                return
            
            # 构建映射字典
            for _, row in df.iterrows():
                label = int(row['标签'])
                self.mapping_dict[label] = {
                    '故障原因': str(row['故障原因']).strip(),
                    '处理方案': str(row['处理方案']).strip()
                }
            
            print(f"✓ 已加载故障诊断映射: {len(self.mapping_dict)} 个类别")
            
            # 显示映射信息
            for label, info in sorted(self.mapping_dict.items()):
                print(f"  类别 {label}: {info['故障原因']}")
                
        except Exception as e:
            print(f"警告: 加载映射文件时出错: {str(e)}")
            print("将无法提供故障诊断信息")
    
    def get_diagnosis(self, label: int) -> Optional[Dict[str, str]]:
        """
        根据标签获取故障诊断信息
        
        Args:
            label: 预测的标签
            
        Returns:
            包含故障原因和处理方案的字典，如果标签不存在则返回None
        """
        return self.mapping_dict.get(label, None)
    
    def format_diagnosis(self, label: int, confidence: float = None) -> str:
        """
        格式化故障诊断信息为可读字符串
        
        Args:
            label: 预测的标签
            confidence: 预测置信度（可选）
            
        Returns:
            格式化的诊断信息
        """
        diagnosis = self.get_diagnosis(label)
        
        if diagnosis is None:
            return f"类别 {label}: 未找到对应的故障诊断信息"
        
        # 构建输出字符串
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"预测类别: {label}")
        if confidence is not None:
            lines.append(f"置信度: {confidence:.4f} ({confidence*100:.2f}%)")
        lines.append(f"{'=' * 60}")
        lines.append(f"故障原因: {diagnosis['故障原因']}")
        lines.append(f"\n处理方案:")
        
        # 处理换行符，使处理方案格式更清晰
        solutions = diagnosis['处理方案'].replace('\\n', '\n')
        for line in solutions.split('\n'):
            if line.strip():
                lines.append(f"  {line.strip()}")
        
        lines.append(f"{'=' * 60}")
        
        return '\n'.join(lines)
    
    def has_mapping(self) -> bool:
        """检查是否有有效的映射数据"""
        return len(self.mapping_dict) > 0
    
    def get_all_labels(self):
        """获取所有已定义的标签"""
        return sorted(self.mapping_dict.keys())
    
    def print_all_mappings(self):
        """打印所有映射关系"""
        if not self.has_mapping():
            print("没有可用的故障诊断映射")
            return
        
        print("\n" + "=" * 80)
        print("故障诊断映射表")
        print("=" * 80)
        
        for label in sorted(self.mapping_dict.keys()):
            info = self.mapping_dict[label]
            print(f"\n类别 {label}:")
            print(f"  故障原因: {info['故障原因']}")
            print(f"  处理方案: {info['处理方案'][:50]}..." if len(info['处理方案']) > 50 else f"  处理方案: {info['处理方案']}")
        
        print("=" * 80)


def create_default_mapping_excel(output_path: str = 'class_mapping.xlsx'):
    """
    创建一个默认的映射Excel文件模板
    
    Args:
        output_path: 输出文件路径
    """
    # 示例数据
    data = {
        '标签': [0, 1, 2, 3],
        '故障原因': [
            '滑环编码器转速跳变',
            '滑环编码器转速规律波动',
            '超速继电器转速异常',
            '转速未超限'
        ],
        '处理方案': [
            '1、检查梅花联轴器有无松动或损坏\n2、检查滑环编码器线束固定有无松动\n3、检查机舱吊机线束有无松动',
            '1、检查梅花联轴器有无松动或损坏\n2、检查滑环编码器屏蔽线有无异常\n3、检查CMS在线监测系统数据采集',
            '1、检查滑环编码器有无损坏\n2、检查超速继电器1有无掉电或损坏',
            '1、检查超速对应的安全链回路接线有无松动\n2、检查超速对应的安全链继电器和底座有无松动'
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"✓ 已创建默认映射文件: {output_path}")


if __name__ == '__main__':
    # 测试代码
    import sys
    
    if len(sys.argv) > 1:
        mapping_file = sys.argv[1]
    else:
        mapping_file = 'class_mapping.xlsx'
    
    # 如果文件不存在，创建默认模板
    if not os.path.exists(mapping_file):
        print(f"映射文件不存在，创建默认模板...")
        create_default_mapping_excel(mapping_file)
    
    # 加载并测试
    mapper = FaultDiagnosisMapper(mapping_file)
    
    # 打印所有映射
    mapper.print_all_mappings()
    
    # 测试单个查询
    print("\n测试诊断查询:")
    for label in range(4):
        print("\n" + mapper.format_diagnosis(label, confidence=0.85))
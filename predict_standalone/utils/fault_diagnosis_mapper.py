"""
故障诊断映射模块
从Excel表格读取标签-故障原因-处理方案的映射关系
"""
import pandas as pd
import os
from typing import Dict, Optional


class FaultDiagnosisMapper:
    """故障诊断映射器"""
    
    def __init__(self, mapping_file: str, fault_name: str = None):
        """
        初始化故障诊断映射器
        
        Args:
            mapping_file: Excel映射文件路径
            fault_name: 故障名称，用于从统一映射表中筛选对应故障的行。
                        如果为 None，则加载整个表（兼容旧的单故障文件）。
        """
        self.mapping_file = mapping_file
        self.fault_name = fault_name
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
            
            # 如果指定了故障名称，按 'fault_name' 列筛选
            if self.fault_name and 'fault_name' in df.columns:
                df = df[df['fault_name'] == self.fault_name]
                if df.empty:
                    print(f"警告: 映射表中未找到故障 '{self.fault_name}' 的记录")
                    return
            
            cols = df.columns.tolist()
            
            # 确定 key 列（数字，用于匹配预测结果）
            key_col = None
            for candidate in ('id', '序号', '标签'):
                if candidate in cols:
                    key_col = candidate
                    break
            
            if key_col is None:
                print(f"警告: 无法确定数字ID列，当前列: {cols}")
                return
            
            # 确定故障原因和处理方案列（兼容中英文列名）
            reason_col = next((c for c in ('fault_reason', '故障原因') if c in cols), None)
            solution_col = next((c for c in ('solution', '处理方案') if c in cols), None)
            
            if solution_col is None:
                print(f"警告: 缺少处理方案列，当前列: {cols}")
                return
            
            # 构建映射字典
            for _, row in df.iterrows():
                try:
                    label = int(row[key_col])
                except (ValueError, TypeError):
                    continue
                
                entry = {
                    '处理方案': str(row.get(solution_col, '')).strip(),
                    '故障原因': str(row.get(reason_col, '')).strip() if reason_col else '',
                }
                self.mapping_dict[label] = entry
            
            print(f"[OK] 已加载故障诊断映射: {len(self.mapping_dict)} 个类别")
            
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

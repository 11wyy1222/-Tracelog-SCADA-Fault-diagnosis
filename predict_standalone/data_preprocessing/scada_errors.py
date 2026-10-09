# -*- coding: utf-8 -*-
"""
SCADA错误处理模块

提供统一的错误码定义、错误对象和日志记录功能。
"""

import os
from datetime import datetime
from typing import Dict, Any, List


# ============================================================================
# 错误码定义
# ============================================================================

class SCADA_ERROR_CODES:
    """SCADA错误码常量"""
    FORMAT_UNSUPPORTED = 'SCADA_FORMAT_UNSUPPORTED'
    FILE_NOT_FOUND = 'SCADA_FILE_NOT_FOUND'
    TIMESTAMP_OUT_OF_RANGE = 'SCADA_TIMESTAMP_OUT_OF_RANGE'
    INSUFFICIENT_DATA = 'SCADA_INSUFFICIENT_DATA'
    CORRUPTED_FILE = 'SCADA_CORRUPTED_FILE'
    MISSING_TAG = 'SCADA_MISSING_TAG'
    ENCODING_ERROR = 'SCADA_ENCODING_ERROR'
    EXTRACTION_ERROR = 'SCADA_EXTRACTION_ERROR'


# ============================================================================
# 错误对象
# ============================================================================

class SCADA_Error:
    """SCADA错误处理类"""

    def __init__(
        self,
        error_code: str,
        message: str,
        file_path: str = None,
        details: Dict[str, Any] = None
    ):
        """
        初始化SCADA错误对象

        参数:
            error_code: 错误码
            message: 错误消息
            file_path: 相关文件路径
            details: 额外的错误详情
        """
        self.error_code = error_code
        self.message = message
        self.file_path = file_path
        self.details = details or {}
        self.timestamp = datetime.now()

    def log(self, log_file: str = None):
        """
        将错误记录到日志文件

        参数:
            log_file: 日志文件路径，None时使用默认路径
        """
        if log_file is None:
            log_file = os.path.join(os.getcwd(), 'SCADA数据文件读取异常记录.txt')

        error_line = (
            f"[{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"{self.file_path or 'N/A'}: "
            f"{self.error_code} - {self.message}\n"
        )

        if self.details:
            error_line += f"  详情: {self.details}\n"

        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(error_line)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'file_path': self.file_path,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }

    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.error_code}: {self.message}"

    def __repr__(self) -> str:
        """调试表示"""
        return f"SCADA_Error(code={self.error_code}, message={self.message})"


# ============================================================================
# 日志记录函数
# ============================================================================

def log_to_error_file(list_msg: List[str], log_file: str = None):
    """
    记录错误消息到日志文件

    参数:
        list_msg: 消息列表
        log_file: 日志文件路径，None时使用默认路径
    """
    if log_file is None:
        log_file = os.path.join(os.getcwd(), "SCADA数据文件读取异常记录.txt")

    os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)

    with open(log_file, "a", encoding="utf-8") as txtfile:
        txtfile.write("\n")
        txtfile.write("时间：" + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
        for item_msg in list_msg:
            txtfile.write(str(item_msg))
            txtfile.write("\n")


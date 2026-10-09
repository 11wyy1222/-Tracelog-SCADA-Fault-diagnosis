from enum import Enum

# 1.错误码常量定义
SUCCESS = 0
ERROR_UNKNOWN = 1000
ERROR_INVALID_PARAM = 1001
ERROR_PERMISSION_DENIED = 1002
ERROR_NOT_FOUND = 1003
ERROR_DATABASE_ERROR = 1004

# 2.使用枚举类定义错误码
class ErrorCode(Enum):
    # 通用错误码 (0-999)
    SUCCESS = (0, "成功")
    ERROR_UNKNOWN = (1000, "未知错误")
    ERROR_INVALID_PARAM = (1001, "参数无效")
    ERROR_PERMISSION_DENIED = (1002, "权限拒绝")
    ERROR_NOT_FOUND = (1003, "资源未找到")
    ERROR_DATABASE_ERROR = (1004, "数据库错误")
    FILE_NOT_FOUND = (1005, "文件未找到")
    DATA_EMPTY = (1006, "数据为空")
    PROCESS_ERROR = (1007, "处理过程出错")
    
    # 数据输入错误码 (1100-1199)
    ERR_DAT_101 = (1101, "无法解析CSV编码或文件格式错误")
    ERR_DAT_102 = (1102, "时间窗口截取后无有效数据")
    ERR_DAT_103 = (1103, "内部处理崩溃")
    ERR_DAT_104 = (1104, "数据为空")
    ERR_DAT_105 = (1105, "缺失timestamp关键列")
    
    # 特征提取错误码 (1200-1299)
    ERR_DAT_201 = (1201, "通道缺失")
    ERR_DAT_202 = (1202, "通道数据全空或全为NaN")
    ERR_DAT_203 = (1203, "通道特征提取崩溃")
    ERR_DAT_204 = (1204, "多通道特征因列不足跳过")
    ERR_DAT_205 = (1205, "多通道特征提取失败")
    
    # 批量处理错误码 (1300-1399)
    ERR_DAT_301 = (1301, "没有样本被成功处理")
    ERR_LBL_001 = (1302, "标签未在映射表中定义")
    
    # 特征计算错误码 (1400-1499)
    ERR_FEAT_001 = (1401, "特征提取被中止")
    ERR_FEAT_002 = (1402, "计算特征时发生未捕获异常")
    
    # 输入文件错误码 (1500-1599)
    ERR_INP_001 = (1501, "读取Excel失败")
    
    # 业务规则警告码 (1600-1699)
    WARN_BIZ_001 = (1601, "故障模式分析非关键性出错")
    
    def __init__(self, code, message):
        self.code = code
        self.message = message
    
    @classmethod
    def from_string(cls, error_str):
        """从字符串错误码获取枚举值"""
        for member in cls:
            if member.name == error_str:
                return member
        return cls.ERROR_UNKNOWN


class GlobalVars:
    # 数据库配置
    DATABASE_URL = "sqlite:///app.db"
    DB_TIMEOUT = 30
    
    # API配置
    API_BASE_URL = "https://api.example.com"
    API_TIMEOUT = 10
    API_RETRY_COUNT = 3
    
    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FILE = "app.log"
    LOG_MAX_SIZE = 1024 * 1024 * 10  # 10MB
    
    # 应用配置
    APP_NAME = "MyApplication"
    VERSION = "1.0.0"
    DEBUG_MODE = False
    
    # 用户配置
    DEFAULT_USER_ROLE = "user"
    SESSION_TIMEOUT = 3600  # 1小时
    
    # 系统配置
    MAX_UPLOAD_SIZE = 1024 * 1024 * 5  # 5MB
    SUPPORTED_FORMATS = ['jpg', 'png', 'pdf', 'docx']
    
    # 缓存配置
    CACHE_ENABLED = True
    CACHE_TTL = 300  # 5分钟

# 创建全局变量实例
globals_instance = GlobalVars()




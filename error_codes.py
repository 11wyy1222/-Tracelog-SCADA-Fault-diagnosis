"""
统一错误代码定义

所有 api 模块共用的错误代码和日志函数。
各模块按编号段划分:
    1001-1999  preprocess_3D（数据预处理）
    2101-2999  train（模型训练）
    3101-3999  predict（模型预测）
    4101-4199  prepare_data（数据准备）
"""

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorCode(Enum):
    # ── preprocess_3D: 1001-1099 ──
    CONFIG_LOAD_ERROR     = 1001
    CONFIG_OVERRIDE_ERROR = 1002
    PREPROCESS_INIT_ERROR = 1003
    PROCESS_ERROR         = 1004
    SPLIT_NORMALIZE_ERROR = 1005
    SAVE_DATASET_ERROR    = 1006
    PREPROCESS_UNKNOWN    = 1099

    # ── data_preprocessing (2D): 1100-1699 ──
    # 通用 (1100-1109)
    SUCCESS               = 1100
    FILE_NOT_FOUND        = 1101
    DATA_EMPTY            = 1102

    # 数据输入 (1110-1129)
    CSV_PARSE_ERROR       = 1110   # 无法解析CSV编码或文件格式错误
    TIME_WINDOW_EMPTY     = 1111   # 时间窗口截取后无有效数据
    DATA_PROCESS_CRASH    = 1112   # 内部处理崩溃
    DATA_IS_EMPTY         = 1113   # 数据为空
    TIMESTAMP_MISSING     = 1114   # 缺失timestamp关键列

    # 特征提取 (1130-1149)
    CHANNEL_MISSING       = 1130   # 通道缺失
    CHANNEL_ALL_NAN       = 1131   # 通道数据全空或全为NaN
    CHANNEL_EXTRACT_CRASH = 1132   # 通道特征提取崩溃
    MULTI_CH_INSUFFICIENT = 1133   # 多通道特征因列不足跳过
    MULTI_CH_EXTRACT_FAIL = 1134   # 多通道特征提取失败
    FEAT_ABORTED          = 1135   # 特征提取被中止
    FEAT_UNCAUGHT_ERROR   = 1136   # 计算特征时发生未捕获异常

    # 批量处理 (1150-1169)
    NO_VALID_SAMPLES      = 1150   # 没有样本被成功处理
    LABEL_NOT_IN_MAP      = 1151   # 标签未在映射表中定义

    # 输入文件 (1170-1189)
    EXCEL_READ_ERROR      = 1170   # 读取Excel失败

    # 业务规则警告 (1190-1199)
    BIZ_ANALYSIS_WARN     = 1190   # 故障模式分析非关键性出错

    # ── train: 2101-2999 ──
    TRAIN_CONFIG_ERROR    = 2101
    DATA_LOADER_ERROR     = 2102
    MODEL_CREATION_ERROR  = 2103
    PRETRAIN_LOAD_ERROR   = 2104
    TRAINING_ERROR        = 2105
    EVALUATION_ERROR      = 2106
    CONFIG_SAVE_ERROR     = 2107
    TRAIN_PARAM_ERROR     = 2108
    TRAIN_UNKNOWN         = 2999

    # ── predict: 3101-3999 ──
    CHECKPOINT_MISSING    = 3101
    CONFIG_MISSING        = 3102
    MAPPING_LOAD_ERROR    = 3103
    DATA_FILE_ERROR       = 3104
    DATALOADER_ERROR      = 3105
    MODEL_LOAD_ERROR      = 3106
    PREDICTION_ERROR      = 3107
    SAVE_RESULT_ERROR     = 3108
    PREDICT_PARAM_ERROR   = 3109
    PREDICT_UNKNOWN       = 3999

    # ── prepare_data: 4101-4199 ──
    PREPARE_CONFIG_ERROR   = 4101
    TRACELOG_NOT_FOUND     = 4102
    TRACELOG_PROCESS_ERROR = 4103
    PREPARE_SAVE_ERROR     = 4104
    PREPARE_PARAM_ERROR    = 4105
    PREPARE_UNKNOWN        = 4199


# 错误描述映射
ERROR_DESCRIPTIONS = {
    # preprocess_3D
    ErrorCode.CONFIG_LOAD_ERROR:      "配置文件加载失败，例如路径不存在或格式错误",
    ErrorCode.CONFIG_OVERRIDE_ERROR:  "overrides 参数应用异常",
    ErrorCode.PREPROCESS_INIT_ERROR:  "创建 DataPreprocessor 实例时出错，可能配置缺失",
    ErrorCode.PROCESS_ERROR:          "预处理过程中发生错误（读取/转换/检查数据）",
    ErrorCode.SPLIT_NORMALIZE_ERROR:  "划分或标准化数据时出错",
    ErrorCode.SAVE_DATASET_ERROR:     "保存数据集到磁盘失败（路径或权限问题）",
    ErrorCode.PREPROCESS_UNKNOWN:     "预处理未知错误，请检查日志",
    # data_preprocessing (2D)
    ErrorCode.SUCCESS:                "成功",
    ErrorCode.FILE_NOT_FOUND:         "文件未找到",
    ErrorCode.DATA_EMPTY:             "数据为空",
    ErrorCode.CSV_PARSE_ERROR:        "无法解析CSV编码或文件格式错误",
    ErrorCode.TIME_WINDOW_EMPTY:      "时间窗口截取后无有效数据",
    ErrorCode.DATA_PROCESS_CRASH:     "内部处理崩溃",
    ErrorCode.DATA_IS_EMPTY:          "数据为空（DataFrame为空或None）",
    ErrorCode.TIMESTAMP_MISSING:      "缺失timestamp关键列",
    ErrorCode.CHANNEL_MISSING:        "通道缺失",
    ErrorCode.CHANNEL_ALL_NAN:        "通道数据全空或全为NaN",
    ErrorCode.CHANNEL_EXTRACT_CRASH:  "通道特征提取崩溃",
    ErrorCode.MULTI_CH_INSUFFICIENT:  "多通道特征因列不足跳过",
    ErrorCode.MULTI_CH_EXTRACT_FAIL:  "多通道特征提取失败",
    ErrorCode.FEAT_ABORTED:           "特征提取被中止",
    ErrorCode.FEAT_UNCAUGHT_ERROR:    "计算特征时发生未捕获异常",
    ErrorCode.NO_VALID_SAMPLES:       "没有样本被成功处理",
    ErrorCode.LABEL_NOT_IN_MAP:       "标签未在映射表中定义",
    ErrorCode.EXCEL_READ_ERROR:       "读取Excel失败",
    ErrorCode.BIZ_ANALYSIS_WARN:      "故障模式分析非关键性出错",
    # train
    ErrorCode.TRAIN_CONFIG_ERROR:     "加载YAML配置失败，可能文件不存在或格式有误",
    ErrorCode.DATA_LOADER_ERROR:      "数据集加载或 DataLoader 创建失败",
    ErrorCode.MODEL_CREATION_ERROR:   "模型实例化或模型配置获取出错",
    ErrorCode.PRETRAIN_LOAD_ERROR:    "加载预训练模型失败，检查路径和格式",
    ErrorCode.TRAINING_ERROR:         "训练过程中发生异常",
    ErrorCode.EVALUATION_ERROR:       "测试/评估阶段出错",
    ErrorCode.CONFIG_SAVE_ERROR:      "配置文件保存失败",
    ErrorCode.TRAIN_PARAM_ERROR:      "训练输入参数校验失败",
    ErrorCode.TRAIN_UNKNOWN:          "训练未知错误，请检查日志",
    # predict
    ErrorCode.CHECKPOINT_MISSING:     "检查点文件不存在或无法访问",
    ErrorCode.CONFIG_MISSING:         "配置文件缺失或格式错误，加载失败",
    ErrorCode.MAPPING_LOAD_ERROR:     "故障诊断映射文件读取失败",
    ErrorCode.DATA_FILE_ERROR:        "外部数据文件缺失或内容不符合预期",
    ErrorCode.DATALOADER_ERROR:       "内部数据集或 DataLoader 创建错误",
    ErrorCode.MODEL_LOAD_ERROR:       "模型实例化或权重加载发生错误",
    ErrorCode.PREDICTION_ERROR:       "预测过程中出现异常",
    ErrorCode.SAVE_RESULT_ERROR:      "保存预测结果失败（文件写入出错）",
    ErrorCode.PREDICT_PARAM_ERROR:    "预测输入参数校验失败",
    ErrorCode.PREDICT_UNKNOWN:        "预测未知错误，请查看详细日志",
    # prepare_data
    ErrorCode.PREPARE_CONFIG_ERROR:   "数据准备配置加载失败，可能路径不存在或 YAML 语法错误",
    ErrorCode.TRACELOG_NOT_FOUND:     "指定的 tracelog 文件不存在",
    ErrorCode.TRACELOG_PROCESS_ERROR: "在处理 tracelog 时发生异常",
    ErrorCode.PREPARE_SAVE_ERROR:     "保存输出 .pt 文件失败",
    ErrorCode.PREPARE_PARAM_ERROR:    "数据准备输入参数校验失败",
    ErrorCode.PREPARE_UNKNOWN:        "数据准备未知错误，请查看日志",
}


def log_error(code: ErrorCode, message: str = "") -> None:
    """统一错误日志输出。"""
    desc = ERROR_DESCRIPTIONS.get(code, "")
    log_msg = f"错误代码: {code.value} ({code.name}) -> {desc}"
    if message:
        log_msg += f" | 详细信息: {message}"
    logger.error(log_msg)
"""
日志工具模块
提供统一的日志记录功能
"""
import logging
import os
from datetime import datetime
import time
from pathlib import Path

def _logger(log_path, level=logging.DEBUG):
    """
    创建日志记录器
    
    Args:
        log_path: 日志文件路径
        level: 日志级别
        
    Returns:
        logger: 配置好的日志记录器
    """
    # 确保日志目录存在
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 清除已有的处理器(避免重复)
    logger.handlers = []
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def setup_logger(name='training', log_dir='logs', level=logging.DEBUG):
    """
    设置并返回一个日志记录器
    
    Args:
        name: 日志器名称
        log_dir: 日志文件目录
        level: 日志级别
        
    Returns:
        logger: 配置好的日志记录器
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件路径
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    
    # 使用 _logger 创建日志器
    return _logger(log_file, level)


def set_requires_grad(model, model_dict, requires_grad=True):
    """
    设置模型参数是否需要梯度
    
    Args:
        model: 模型
        model_dict: 模型参数字典
        requires_grad: 是否需要梯度
    """
    for name, param in model.named_parameters():
        if name in model_dict:
            param.requires_grad = requires_grad
    
    if requires_grad:
        print("✓ 所有参数已解冻")
    else:
        print("✓ 指定参数已冻结")


def count_parameters(model):
    """
    计算模型参数量
    
    Args:
        model: 模型
        
    Returns:
        total_params: 总参数量
        trainable_params: 可训练参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def print_model_info(model, logger=None):
    """
    打印模型信息
    
    Args:
        model: 模型
        logger: 日志记录器(可选)
    """
    total_params, trainable_params = count_parameters(model)
    
    info = [
        "\n" + "="*60,
        "模型信息",
        "="*60,
        f"总参数量: {total_params:,}",
        f"可训练参数: {trainable_params:,}",
        f"冻结参数: {total_params - trainable_params:,}",
        "="*60
    ]
    
    for line in info:
        if logger:
            logger.debug(line)
        else:
            print(line)


class LogFormatter:
    """日志格式化工具"""
    
    @staticmethod
    def section(title, char='=', length=60):
        """
        创建分节标题
        
        Args:
            title: 标题文本
            char: 分隔字符
            length: 总长度
            
        Returns:
            formatted_title: 格式化的标题
        """
        if len(title) >= length - 4:
            return f"{char*2} {title} {char*2}"
        
        padding = (length - len(title) - 2) // 2
        return f"{char*padding} {title} {char*padding}"
    
    @staticmethod
    def key_value(key, value, key_width=20):
        """
        创建键值对格式
        
        Args:
            key: 键
            value: 值
            key_width: 键的宽度
            
        Returns:
            formatted_kv: 格式化的键值对
        """
        return f"{key:<{key_width}}: {value}"
    
    @staticmethod
    def metric(name, value, decimals=4):
        """
        格式化指标
        
        Args:
            name: 指标名称
            value: 指标值
            decimals: 小数位数
            
        Returns:
            formatted_metric: 格式化的指标
        """
        if isinstance(value, float):
            return f"{name}: {value:.{decimals}f}"
        else:
            return f"{name}: {value}"
    
    @staticmethod
    def table_row(items, widths):
        """
        创建表格行
        
        Args:
            items: 项目列表
            widths: 每列宽度列表
            
        Returns:
            formatted_row: 格式化的行
        """
        row = ""
        for item, width in zip(items, widths):
            row += f"{str(item):<{width}} "
        return row.rstrip()


def log_epoch_info(logger, epoch, total_epochs, train_loss, train_acc, 
                   val_loss, val_acc, learning_rate=None):
    """
    记录训练轮次信息
    
    Args:
        logger: 日志记录器
        epoch: 当前轮次
        total_epochs: 总轮次
        train_loss: 训练损失
        train_acc: 训练准确率
        val_loss: 验证损失
        val_acc: 验证准确率
        learning_rate: 学习率(可选)
    """
    formatter = LogFormatter()
    
    logger.debug("")
    logger.debug(formatter.section(f"Epoch {epoch}/{total_epochs}"))
    logger.debug(formatter.metric("训练损失", train_loss))
    logger.debug(formatter.metric("训练准确率", train_acc))
    logger.debug(formatter.metric("验证损失", val_loss))
    logger.debug(formatter.metric("验证准确率", val_acc))
    
    if learning_rate is not None:
        logger.debug(formatter.metric("学习率", learning_rate, decimals=6))
    
    logger.debug("="*60)


def log_training_start(logger, config):
    """
    记录训练开始信息
    
    Args:
        logger: 日志记录器
        config: 配置对象/字典
    """
    formatter = LogFormatter()
    
    logger.debug("")
    logger.debug(formatter.section("训练配置", char='=', length=60))
    
    # 提取配置信息
    if isinstance(config, dict):
        batch_size = config.get('batch_size', 'N/A')
        num_epochs = config.get('num_epoch', 'N/A')
        learning_rate = config.get('lr', 'N/A')
        optimizer = config.get('optimizer', 'N/A')
    else:
        batch_size = getattr(config, 'batch_size', 'N/A')
        num_epochs = getattr(config, 'num_epoch', 'N/A')
        learning_rate = getattr(config, 'lr', 'N/A')
        optimizer = getattr(config, 'optimizer', 'N/A')
    
    logger.debug(formatter.key_value("批次大小", batch_size))
    logger.debug(formatter.key_value("训练轮数", num_epochs))
    logger.debug(formatter.key_value("学习率", learning_rate))
    logger.debug(formatter.key_value("优化器", optimizer))
    logger.debug("="*60)


def log_training_complete(logger, start_time, end_time):
    """
    记录训练完成信息
    
    Args:
        logger: 日志记录器
        start_time: 开始时间
        end_time: 结束时间
    """
    formatter = LogFormatter()
    duration = end_time - start_time
    
    logger.debug("")
    logger.debug(formatter.section("训练完成", char='=', length=60))
    logger.debug(formatter.key_value("开始时间", start_time.strftime('%Y-%m-%d %H:%M:%S')))
    logger.debug(formatter.key_value("结束时间", end_time.strftime('%Y-%m-%d %H:%M:%S')))
    logger.debug(formatter.key_value("总耗时", str(duration)))
    logger.debug("="*60)


def preprocess_logger(save_dir, log_name="training"):
    """
    配置全局 Logger
    :param save_dir: 日志保存目录
    :param log_name: 日志文件名前缀
    :return: logger 对象
    """
    # 1. 创建日志目录
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 2. 生成带时间戳的文件名
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = save_path / f"{log_name}_{timestamp}.log"

    # 3. 获取 root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 防止重复添加 handler
    if logger.hasHandlers():
        logger.handlers.clear()

    # 4. 定义格式
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 5. Handler 1: 输出到文件
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 6. Handler 2: 输出到控制台
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 屏蔽 Matplotlib 的字体查找等调试信息
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    # 屏蔽 Pillow (PIL) 的图像处理调试信息
    logging.getLogger('PIL').setLevel(logging.WARNING)
    # 如果以后用到网络请求，也可以屏蔽这个
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    # ========================================

    logger.info(f"日志系统初始化完成。日志文件: {log_file}")
    return logger
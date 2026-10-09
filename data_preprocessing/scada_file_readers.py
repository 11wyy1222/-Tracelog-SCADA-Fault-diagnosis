# -*- coding: utf-8 -*-
"""
SCADA文件读取模块

提供各种压缩格式的SCADA文件读取功能。
支持格式: csv, gz, zip, rar, 7z
"""

import os
import sys
import tempfile
import shutil
import traceback
from pathlib import Path

import pandas as pd
import filetype
import py7zr


def _configure_unrar_library():
    """Point python-unrar at the DLL bundled next to the project."""
    if os.name != "nt":
        return

    configured_path = os.environ.get("UNRAR_LIB_PATH")
    if configured_path and Path(configured_path).is_file():
        return

    dll_relative_path = (
        Path("tools/x64/UnRAR64.dll")
        if sys.maxsize > 2**32
        else Path("tools/UnRAR.dll")
    )
    for parent in Path(__file__).resolve().parents:
        dll_path = parent / dll_relative_path
        if dll_path.is_file():
            os.environ["UNRAR_LIB_PATH"] = str(dll_path)
            return


_configure_unrar_library()
try:
    from unrar import rarfile
except (ImportError, LookupError, OSError):
    # RAR support is optional; other SCADA formats must remain usable if the
    # package or native library is unavailable.
    rarfile = None

from .scada_tag_processor import set_by_order
from .scada_errors import log_to_error_file


# ============================================================================
# 文件类型识别
# ============================================================================

def get_file_suffix_and_type(filepath):
    """
    返回文件后缀和 filetype 识别结果

    参数:
        filepath: 文件路径

    返回:
        (suffix, ft_ext): 文件后缀和filetype识别的扩展名
    """
    suffix = Path(filepath).suffix.lower().replace(".", "")

    ft_ext = None
    try:
        ft_info = filetype.guess(filepath)
        if ft_info is not None:
            ft_ext = ft_info.__dict__.get("_Type__extension", None)
    except Exception:
        ft_ext = None

    return suffix, ft_ext


# ============================================================================
# CSV文件读取
# ============================================================================

def read_csv(filepath, model_tags, time_tags, encoding="utf8", log_enable=False):
    """
    读取CSV文件

    参数:
        filepath: 文件路径
        model_tags: 模型标签列表
        time_tags: 时间标签列表
        encoding: 编码格式
        log_enable: 是否启用日志

    返回:
        (data, encoding): DataFrame和使用的编码
    """
    data = pd.DataFrame()

    encoding_list = set_by_order([encoding, "utf8", "gbk", None])

    for item_encoding in encoding_list:
        try:
            data_head = pd.read_csv(
                filepath,
                nrows=1,
                encoding=item_encoding,
                index_col=False,
                low_memory=False,
            )

            columns = data_head.columns.tolist()

            time_tag_list = list(set(time_tags).intersection(set(columns)))
            if len(time_tag_list) == 0:
                continue

            tag_list = list(set(model_tags).intersection(set(columns)))

            # 将时间标签加入读取列表
            tag_list = list(set(tag_list) | set(time_tag_list))

            data = pd.read_csv(
                filepath,
                usecols=tag_list,
                encoding=item_encoding,
                index_col=False,
                low_memory=False,
            )
            break

        except UnicodeDecodeError:
            if item_encoding != encoding_list[-1]:
                continue
            else:
                if log_enable:
                    list_msg = ["UnicodeDecodeError:", filepath]
                    log_to_error_file(list_msg)

        except Exception as e:
            if e.__class__.__name__ == "ValueError":
                continue
            else:
                if log_enable:
                    list_msg = [
                        "read_csv()执行出现异常：",
                        filepath,
                        e.__class__.__name__,
                        str(traceback.format_exc()),
                    ]
                    log_to_error_file(list_msg)
                break

    return data, item_encoding


# ============================================================================
# 压缩文件读取 (ZIP/GZ)
# ============================================================================

def read_zip_gz(
    filepath,
    model_tags,
    time_tags,
    encoding="utf8",
    compression="gzip",
    log_enable=False,
):
    """
    读取ZIP或GZ压缩文件

    参数:
        filepath: 文件路径
        model_tags: 模型标签列表
        time_tags: 时间标签列表
        encoding: 编码格式
        compression: 压缩格式 ('gzip' 或 'zip')
        log_enable: 是否启用日志

    返回:
        (data, encoding): DataFrame和使用的编码
    """
    data = pd.DataFrame()

    encoding_list = set_by_order([encoding, "utf8", "gbk", None])

    for item_encoding in encoding_list:
        try:
            data_head = pd.read_csv(
                filepath,
                nrows=1,
                encoding=item_encoding,
                compression=compression,
                index_col=False,
                low_memory=False,
            )

            columns = data_head.columns.tolist()

            time_tag_list = list(set(time_tags).intersection(set(columns)))
            if len(time_tag_list) == 0:
                continue

            tag_list = list(set(model_tags).intersection(set(columns)))

            # 将时间标签加入读取列表
            tag_list = list(set(tag_list) | set(time_tag_list))

            data = pd.read_csv(
                filepath,
                usecols=tag_list,
                encoding=item_encoding,
                compression=compression,
                index_col=False,
                low_memory=False,
            )
            break

        except UnicodeDecodeError:
            if item_encoding != encoding_list[-1]:
                continue
            else:
                if log_enable:
                    list_msg = ["UnicodeDecodeError:", filepath]
                    log_to_error_file(list_msg)

        except Exception as e:
            if e.__class__.__name__ == "ValueError":
                continue
            else:
                if log_enable:
                    list_msg = [
                        "read_zip_gz()执行出现异常：",
                        filepath,
                        e.__class__.__name__,
                        str(traceback.format_exc()),
                    ]
                    log_to_error_file(list_msg)
                break

    return data, item_encoding


# ============================================================================
# 解压后文件读取
# ============================================================================

def read_extracted_file(filepath, model_tags, time_tags, log_enable=False):
    """
    读取解压后的文件

    参数:
        filepath: 文件路径
        model_tags: 模型标签列表
        time_tags: 时间标签列表
        log_enable: 是否启用日志

    返回:
        DataFrame
    """
    suffix, ft_ext = get_file_suffix_and_type(filepath)

    if suffix == "csv":
        data_one, _ = read_csv(
            filepath,
            model_tags,
            time_tags,
            encoding="gbk",
            log_enable=log_enable,
        )
        return data_one

    if suffix == "gz" or ft_ext == "gz":
        data_one, _ = read_zip_gz(
            filepath,
            model_tags,
            time_tags,
            encoding="utf8",
            compression="gzip",
            log_enable=log_enable,
        )
        return data_one

    if suffix == "zip" or ft_ext == "zip":
        data_one, _ = read_zip_gz(
            filepath,
            model_tags,
            time_tags,
            encoding="utf8",
            compression="zip",
            log_enable=log_enable,
        )
        return data_one

    return pd.DataFrame()


# ============================================================================
# RAR文件读取
# ============================================================================

def read_rar(filepath, model_tags, time_tags, log_enable=False):
    """
    读取RAR压缩文件

    参数:
        filepath: 文件路径
        model_tags: 模型标签列表
        time_tags: 时间标签列表
        log_enable: 是否启用日志

    返回:
        DataFrame
    """
    if rarfile is None:
        if log_enable:
            log_to_error_file(["缺少 unrar 依赖或 UnRAR DLL，无法读取rar文件：", filepath])
        return pd.DataFrame()

    data_one = pd.DataFrame()
    path_data_orig = tempfile.mkdtemp(prefix="scada_rar_tmp_")

    try:
        rar = rarfile.RarFile(filepath)

        file_list = [x for x in rar.namelist() if not x.endswith("/")]

        if len(file_list) == 0:
            if log_enable:
                log_to_error_file(["rar文件中没有可读取文件：", filepath])
            return pd.DataFrame()

        rar_file = file_list[0]
        rar.extract(rar_file, path_data_orig)

        new_filepath = os.path.join(path_data_orig, rar_file)

        data_one = read_extracted_file(
            new_filepath,
            model_tags,
            time_tags,
            log_enable=log_enable,
        )

    except Exception as e:
        if log_enable:
            list_msg = [
                "read_rar()执行出现异常：",
                filepath,
                e.__class__.__name__,
                str(traceback.format_exc()),
            ]
            log_to_error_file(list_msg)

    finally:
        shutil.rmtree(path_data_orig, ignore_errors=True)

    return data_one


# ============================================================================
# 7Z文件读取
# ============================================================================

def read_7z(filepath, model_tags, time_tags, log_enable=False):
    """
    读取7Z压缩文件

    参数:
        filepath: 文件路径
        model_tags: 模型标签列表
        time_tags: 时间标签列表
        log_enable: 是否启用日志

    返回:
        DataFrame
    """
    data_one = pd.DataFrame()
    path_data_orig = tempfile.mkdtemp(prefix="scada_7z_tmp_")

    try:
        archive = py7zr.SevenZipFile(filepath, mode="r")
        file_list = [x for x in archive.getnames() if not x.endswith("/")]

        if len(file_list) == 0:
            archive.close()
            if log_enable:
                log_to_error_file(["7z文件中没有可读取文件：", filepath])
            return pd.DataFrame()

        target_file = file_list[0]
        archive.extract(path=path_data_orig, targets=[target_file])
        archive.close()

        extracted_filepath = os.path.join(path_data_orig, target_file)

        data_one = read_extracted_file(
            extracted_filepath,
            model_tags,
            time_tags,
            log_enable=log_enable,
        )

        list_cols = data_one.columns.tolist()
        if len(list_cols) < 2:
            list_msg = ["没有找到关键标签点：ALL", extracted_filepath]
            log_to_error_file(list_msg)

    except Exception as e:
        if log_enable:
            list_msg = [
                "read_7z()执行出现异常：",
                filepath,
                e.__class__.__name__,
                str(traceback.format_exc()),
            ]
            log_to_error_file(list_msg)

    finally:
        shutil.rmtree(path_data_orig, ignore_errors=True)

    return data_one


# ============================================================================
# 统一读取接口
# ============================================================================

def read_data(filepath, model_tags, time_tags, encoding="utf8", log_enable=False):
    """
    统一的SCADA数据读取接口

    根据文件类型自动选择合适的读取方法，支持:
        - CSV文件
        - GZ压缩文件
        - ZIP压缩文件
        - RAR压缩文件
        - 7Z压缩文件

    参数:
        filepath: 文件路径
        model_tags: 模型标签列表
        time_tags: 时间标签列表
        encoding: 编码格式
        log_enable: 是否启用日志

    返回:
        DataFrame: 读取的数据
    """
    suffix, ft_ext = get_file_suffix_and_type(filepath)

    # 根据文件类型选择读取方法
    if suffix == "csv":
        data, _ = read_csv(filepath, model_tags, time_tags, encoding, log_enable)
        return data

    if suffix == "gz" or ft_ext == "gz":
        data, _ = read_zip_gz(filepath, model_tags, time_tags, encoding, "gzip", log_enable)
        return data

    if suffix == "zip" or ft_ext == "zip":
        data, _ = read_zip_gz(filepath, model_tags, time_tags, encoding, "zip", log_enable)
        return data

    if suffix == "rar" or ft_ext == "rar":
        data = read_rar(filepath, model_tags, time_tags, log_enable)
        return data

    if suffix == "7z" or ft_ext == "7z":
        data = read_7z(filepath, model_tags, time_tags, log_enable)
        return data

    # 未知格式，尝试作为CSV读取
    if log_enable:
        print(f"⚠ 未知的文件格式: {suffix}, 尝试作为CSV读取")

    try:
        data, _ = read_csv(filepath, model_tags, time_tags, encoding, log_enable)
        return data
    except Exception:
        return pd.DataFrame()

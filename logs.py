# json_logger.py
import json
import datetime
import os
from typing import Any, Dict, Optional

class JSONLogger:
    def __init__(self, filename: str = "log.json"):
        """
        初始化 JSON 日志记录器
        
        Args:
            filename: 日志文件名
            max_entries: 最大日志条目数，超过会自动清理旧日志
        """
        self.filename = filename
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """确保日志文件存在，如果不存在则创建空数组"""
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
    
    def _read_logs(self) -> list:
        """读取现有日志"""
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _write_logs(self, logs: list):
        """写入日志到文件"""
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
    
    def log(self, 
            message: str | None, 
            level: str = "INFO", 
            **extra_fields: Any) -> None:
        """
        记录一条日志到 JSON 文件
        
        Args:
            message: 日志消息
            level: 日志级别 (INFO, WARNING, ERROR, DEBUG)
            **extra_fields: 额外的字段，会一并存入 JSON
        """
        if message == None: message = "None"
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "level": level.upper(),
            "message": message,
            **extra_fields
        }
        logs = self._read_logs()
        print(message)
        logs.append(log_entry)
        self._write_logs(logs)

    def clean(self) -> bool:
        """
        一键清除所有日志记录
        
        Returns:
            bool: 操作是否成功
        """
        try:
            if i := input("是否清除全部日志记录： [y/n]:") != 'y':  raise TypeError("主动退出")
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            print(f"✓ 所有日志记录已清除，文件 '{self.filename}' 已重置为空数组")
            return True
        except Exception as e:
            log(f"✗ 清除日志失败: {e}")
            return False

# 创建全局实例
_logger = JSONLogger()

def log(message: str | None, level: str = "INFO", **kwargs) -> None:
    """
    快速日志函数 - 在其他模块中只需这一条语句即可使用
    
    Args:
        message: 日志消息
        level: 日志级别
        **kwargs: 额外字段
    """
    _logger.log(message, level, **kwargs)

def clean() -> bool:
    """
    一键清除所有日志记录
    
    Returns:
        bool: 操作是否成功
    """
    return _logger.clean()
    

# 便捷方法
def info(message: str, **kwargs) -> None:
    """记录 INFO 级别日志"""
    log(message, "INFO", **kwargs)

def warning(message: str, **kwargs) -> None:
    """记录 WARNING 级别日志"""
    log(message, "WARNING", **kwargs)

def error(message: str, **kwargs) -> None:
    """记录 ERROR 级别日志"""
    log(message, "ERROR", **kwargs)

def debug(message: str, **kwargs) -> None:
    """记录 DEBUG 级别日志"""
    log(message, "DEBUG", **kwargs)
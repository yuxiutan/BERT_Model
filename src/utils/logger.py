"""
日誌管理模組
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logger(name: str) -> logging.Logger:
    """設定日誌記錄器"""
    logger = logging.getLogger(name)
    
    # 避免重複添加handler
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    # 日誌格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台處理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件處理器
    try:
        log_file = Path(os.environ.get('LOG_FILE', '/app/logs/app.log'))
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    except Exception as e:
        logger.warning(f"無法設定文件日誌處理器: {e}")
    
    return logger

class APILogger:
    """API專用日誌記錄器"""
    
    def __init__(self):
        self.logger = setup_logger("api")
    
    def log_request(self, method: str, path: str, status_code: int, response_time: float):
        """記錄API請求"""
        self.logger.info(f"{method} {path} - {status_code} - {response_time:.3f}s")
    
    def log_prediction(self, prediction: str, confidence: float, processing_time: float):
        """記錄模型預測"""
        self.logger.info(f"預測: {prediction}, 信心度: {confidence:.4f}, 處理時間: {processing_time:.3f}s")
    
    def log_low_confidence(self, prediction: str, confidence: float, file_path: str):
        """記錄低信心度事件"""
        self.logger.warning(f"低信心度事件: {prediction} (信心度: {confidence:.4f}) -> 保存至: {file_path}")
    
    def log_retrain_start(self):
        """記錄重訓練開始"""
        self.logger.info("開始自動重訓練程序")
    
    def log_retrain_complete(self, model_version: str, accuracy: float):
        """記錄重訓練完成"""
        self.logger.info(f"重訓練完成 - 新模型版本: {model_version}, 準確率: {accuracy:.4f}")
    
    def log_model_backup(self, old_version: str, new_version: str):
        """記錄模型備份"""
        self.logger.info(f"模型備份完成: {old_version} -> {new_version}")

# 創建全域API日誌記錄器
api_logger = APILogger()

import os

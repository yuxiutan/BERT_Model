"""
配置管理模組
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseSettings, validator

class Settings(BaseSettings):
    """應用程式設定"""
    
    # API設定
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    
    # Wazuh設定
    WAZUH_API_URL: str
    WAZUH_API_USERNAME: str
    WAZUH_API_PASSWORD: str
    WAZUH_INDEX: str = "wazuh-alerts-*"
    
    # Discord設定
    DISCORD_WEBHOOK_URL: Optional[str] = None
    
    # Alert API設定
    ALERT_API_URL: str = "http://100.89.12.61:8999/newalert"
    ALERT_API_TIMEOUT: int = 10
    
    # 模型設定
    MODEL_CONFIDENCE_THRESHOLD: float = 0.3
    MODEL_NAME: str = "bert-base-uncased"
    MAX_LENGTH: int = 512
    NUM_LABELS: int = 2
    
    # 訓練設定
    BATCH_SIZE: int = 2
    LEARNING_RATE: float = 2e-5
    NUM_EPOCHS: int = 3
    DROPOUT_PROB: float = 0.3
    
    # 排程設定
    RETRAIN_TIME: str = "21:00"
    DATA_FETCH_INTERVAL: int = 180  # 秒
    MAX_MODEL_BACKUPS: int = 3
    
    # 檔案路徑設定
    BASE_DIR: Path = Path("/app")
    DATA_DIR: Path = Path("/app/data")
    MODEL_DIR: Path = Path("/app/data/models/current")
    MODEL_BACKUP_DIR: Path = Path("/app/data/models/backups")
    LOW_CONFIDENCE_DIR: Path = Path("/app/data/logs/low_confidence")
    PROCESSED_LOGS_DIR: Path = Path("/app/data/logs/processed")
    REFERENCE_DIR: Path = Path("/app/data/reference")
    
    # 日誌設定
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "/app/logs/app.log"
    
    # 特殊檔案路徑
    REFERENCE_EMBEDDINGS_FILE: str = "reference_embeddings.pkl"
    PROCESSED_DATA_FILE: str = "processed_data.pkl"
    MODEL_CONFIG_FILE: str = "config.json"
    
    @validator('WAZUH_API_URL')
    def validate_wazuh_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('WAZUH_API_URL must start with http:// or https://')
        return v
    
    @validator('RETRAIN_TIME')
    def validate_retrain_time(cls, v):
        try:
            hour, minute = v.split(':')
            hour = int(hour)
            minute = int(minute)
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError()
        except:
            raise ValueError('RETRAIN_TIME must be in HH:MM format (24-hour)')
        return v
    
    @validator('MODEL_CONFIDENCE_THRESHOLD')
    def validate_confidence_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('MODEL_CONFIDENCE_THRESHOLD must be between 0 and 1')
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# 創建全域設定實例
settings = Settings()

# 確保必要目錄存在
def ensure_directories():
    """確保所有必要目錄存在"""
    directories = [
        settings.DATA_DIR,
        settings.MODEL_DIR,
        settings.MODEL_BACKUP_DIR,
        settings.LOW_CONFIDENCE_DIR,
        settings.PROCESSED_LOGS_DIR,
        settings.REFERENCE_DIR,
        Path(settings.LOG_FILE).parent
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# 在模組載入時確保目錄存在
ensure_directories()

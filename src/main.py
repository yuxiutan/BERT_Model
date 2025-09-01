#!/usr/bin/env python3
"""
主程式入口 - 啟動API服務和排程任務
"""

import asyncio
import uvicorn
from pathlib import Path
import sys
import os

# 添加src目錄到Python路徑
sys.path.append(str(Path(__file__).parent))

from utils.config import settings
from utils.logger import setup_logger
from api.app import app
from scheduler.tasks import start_scheduler

# 設定日誌
logger = setup_logger(__name__)

async def startup_tasks():
    """系統啟動時的初始化任務"""
    try:
        logger.info("正在初始化系統...")
        
        # 確保必要目錄存在
        required_dirs = [
            settings.MODEL_DIR,
            settings.MODEL_BACKUP_DIR,
            settings.LOW_CONFIDENCE_DIR,
            settings.PROCESSED_LOGS_DIR,
            Path(settings.LOG_FILE).parent
        ]
        
        for directory in required_dirs:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"確保目錄存在: {directory}")
        
        # 啟動排程器
        start_scheduler()
        logger.info("排程器已啟動")
        
        # 檢查模型檔案
        from models.inference import ModelInference
        inference = ModelInference()
        await inference.initialize()
        logger.info("模型檢查完成")
        
        logger.info("系統初始化完成")
        
    except Exception as e:
        logger.error(f"系統初始化失敗: {e}")
        raise

def main():
    """主函數"""
    try:
        logger.info("=" * 50)
        logger.info("攻擊鏈檢測系統啟動中...")
        logger.info(f"API將在 {settings.API_HOST}:{settings.API_PORT} 啟動")
        logger.info(f"資料獲取間隔: {settings.DATA_FETCH_INTERVAL} 秒")
        logger.info(f"重訓練時間: {settings.RETRAIN_TIME}")
        logger.info(f"信心度閾值: {settings.MODEL_CONFIDENCE_THRESHOLD}")
        logger.info("=" * 50)
        
        # 執行啟動任務
        asyncio.run(startup_tasks())
        
        # 啟動FastAPI服務
        uvicorn.run(
            app,
            host=settings.API_HOST,
            port=settings.API_PORT,
            workers=1,  # 單worker避免排程器重複
            access_log=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("收到中斷信號，正在關閉系統...")
    except Exception as e:
        logger.error(f"系統啟動失敗: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

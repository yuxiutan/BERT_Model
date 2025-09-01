"""
排程任務模組
"""

import asyncio
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from ..utils.config import settings
from ..utils.logger import setup_logger
from ..utils.wazuh_client import WazuhClient
from ..models.inference import ModelInference
from ..models.trainer import ModelTrainer

logger = setup_logger(__name__)

# 全域排程器實例
scheduler = AsyncIOScheduler(timezone='Asia/Taipei')

async def fetch_and_predict_task():
    """定期從Wazuh獲取資料並進行預測的任務"""
    try:
        logger.info("開始執行定期預測任務...")
        
        # 獲取Wazuh資料
        async with WazuhClient() as client:
            alerts = await client.fetch_recent_alerts(minutes=5)
        
        if not alerts:
            logger.info("未獲取到新的告警資料")
            return
        
        # 執行預測
        inference = ModelInference()
        if not inference.is_ready():
            await inference.initialize()
        
        result = await inference.predict_logs(alerts)
        
        logger.info(f"預測完成 - 預測: {result['prediction']}, 信心度: {result['confidence']:.4f}")
        
        # 如果信心度低，保存事件
        if result['confidence'] < settings.MODEL_CONFIDENCE_THRESHOLD:
            await save_low_confidence_event_task(alerts, result['prediction'], result['confidence'])
        
    except Exception as e:
        logger.error(f"定期預測任務失敗: {e}")

async def save_low_confidence_event_task(logs, prediction: str, confidence: float):
    """保存低信心度事件的任務"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = settings.LOW_CONFIDENCE_DIR / f"low_conf_{timestamp}_{confidence:.4f}.json"
        
        event_data = {
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
            "confidence": confidence,
            "logs": logs,
            "metadata": {
                "logs_count": len(logs),
                "processing_time": datetime.now().isoformat()
            }
        }
        
        import json
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(event_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"低信心度事件已保存: {file_path}")
        
    except Exception as e:
        logger.error(f"保存低信心度事件失敗: {e}")

async def retrain_task():
    """每日重訓練任務"""
    try:
        logger.info("開始執行每日重訓練任務...")
        
        trainer = ModelTrainer()
        result = await trainer.retrain_model()
        
        if result["success"]:
            logger.info(f"重訓練成功完成 - 新版本: {result['model_version']}, 準確率: {result['accuracy']:.4f}")
            
            # 重新載入模型（如果有全域推論實例）
            try:
                from ..api.app import model_inference
                if model_inference:
                    await model_inference.initialize()
                    logger.info("模型推論器已重新載入新模型")
            except:
                logger.warning("無法重新載入模型推論器")
                
        else:
            logger.error(f"重訓練失敗: {result['error']}")
            
    except Exception as e:
        logger.error(f"重訓練任務失敗: {e}")

async def cleanup_old_logs_task():
    """清理舊日誌檔案的任務"""
    try:
        from datetime import timedelta
        import os
        
        logger.info("開始清理舊日誌檔案...")
        
        # 清理超過30天的低信心度事件
        cutoff_date = datetime.now() - timedelta(days=30)
        
        for log_file in settings.LOW_CONFIDENCE_DIR.glob("*.json"):
            try:
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff_date:
                    os.remove(log_file)
                    logger.info(f"已清理舊日誌: {log_file}")

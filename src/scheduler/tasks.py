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
            except Exception as e:
                logger.warning(f"清理日誌檔案失敗 {log_file}: {e}")
        
        # 清理處理過的日誌檔案（超過7天）
        cutoff_date = datetime.now() - timedelta(days=7)
        
        for log_file in settings.PROCESSED_LOGS_DIR.glob("*.json"):
            try:
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff_date:
                    os.remove(log_file)
                    logger.info(f"已清理處理過的日誌: {log_file}")
            except Exception as e:
                logger.warning(f"清理處理日誌失敗 {log_file}: {e}")
                
        logger.info("日誌清理任務完成")
        
    except Exception as e:
        logger.error(f"清理日誌任務失敗: {e}")

async def system_health_check_task():
    """系統健康檢查任務"""
    try:
        logger.info("執行系統健康檢查...")
        
        # 檢查模型狀態
        inference = ModelInference()
        model_ready = inference.is_ready()
        
        # 檢查Wazuh連接
        wazuh_ok = False
        try:
            async with WazuhClient() as client:
                await client.fetch_recent_alerts(minutes=1)
                wazuh_ok = True
        except:
            wazuh_ok = False
        
        # 檢查磁碟空間
        import shutil
        total, used, free = shutil.disk_usage(settings.DATA_DIR)
        free_gb = free // (1024**3)
        
        if free_gb < 1:  # 少於1GB空間
            logger.warning(f"磁碟空間不足: {free_gb}GB")
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "model_ready": model_ready,
            "wazuh_connection": wazuh_ok,
            "free_space_gb": free_gb,
            "overall_health": model_ready and wazuh_ok and free_gb > 1
        }
        
        if not status["overall_health"]:
            logger.warning(f"系統健康狀況異常: {status}")
        else:
            logger.info("系統健康檢查通過")
            
    except Exception as e:
        logger.error(f"系統健康檢查失敗: {e}")

def start_scheduler():
    """啟動排程器"""
    try:
        logger.info("正在啟動排程器...")
        
        # 解析重訓練時間
        retrain_hour, retrain_minute = map(int, settings.RETRAIN_TIME.split(':'))
        
        # 添加定期預測任務（每3分鐘）
        scheduler.add_job(
            fetch_and_predict_task,
            trigger=IntervalTrigger(seconds=settings.DATA_FETCH_INTERVAL),
            id='fetch_and_predict',
            name='定期預測任務',
            max_instances=1,  # 防止重複執行
            coalesce=True
        )
        
        # 添加每日重訓練任務
        scheduler.add_job(
            retrain_task,
            trigger=CronTrigger(hour=retrain_hour, minute=retrain_minute),
            id='daily_retrain',
            name='每日重訓練任務',
            max_instances=1
        )
        
        # 添加日誌清理任務（每日凌晨2點）
        scheduler.add_job(
            cleanup_old_logs_task,
            trigger=CronTrigger(hour=2, minute=0),
            id='cleanup_logs',
            name='日誌清理任務',
            max_instances=1
        )
        
        # 添加系統健康檢查任務（每小時）
        scheduler.add_job(
            system_health_check_task,
            trigger=IntervalTrigger(hours=1),
            id='health_check',
            name='系統健康檢查',
            max_instances=1
        )
        
        # 啟動排程器
        scheduler.start()
        
        logger.info("排程器啟動成功")
        logger.info(f"定期預測間隔: {settings.DATA_FETCH_INTERVAL} 秒")
        logger.info(f"重訓練時間: {settings.RETRAIN_TIME}")
        
        # 列出所有排程任務
        for job in scheduler.get_jobs():
            logger.info(f"排程任務: {job.name} - {job.trigger}")
            
    except Exception as e:
        logger.error(f"排程器啟動失敗: {e}")
        raise

def stop_scheduler():
    """停止排程器"""
    try:
        if scheduler.running:
            scheduler.shutdown()
            logger.info("排程器已停止")
    except Exception as e:
        logger.error(f"停止排程器失敗: {e}")

def get_scheduler_status() -> dict:
    """獲取排程器狀態"""
    try:
        if not scheduler.running:
            return {"status": "stopped", "jobs": []}
        
        jobs_info = []
        for job in scheduler.get_jobs():
            jobs_info.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger)
            })
        
        return {
            "status": "running",
            "jobs": jobs_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"獲取排程器狀態失敗: {e}")
        return {"status": "error", "error": str(e)}

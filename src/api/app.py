"""
FastAPI 應用程式 - 攻擊鏈檢測API
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..utils.config import settings
from ..utils.logger import setup_logger, api_logger
from ..models.inference import ModelInference
from ..utils.wazuh_client import WazuhClient

logger = setup_logger(__name__)

# 創建FastAPI應用
app = FastAPI(
    title="攻擊鏈檢測系統",
    description="基於BERT的即時攻擊鏈檢測與分析系統",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全域模型推論實例
model_inference: Optional[ModelInference] = None

# Pydantic模型
class LogData(BaseModel):
    timestamp: Optional[str] = None
    rule_description: str = Field(..., description="規則描述")
    src_ip: Optional[str] = "None"
    dst_ip: Optional[str] = "None"
    rule_id: Optional[str] = None
    rule_level: Optional[int] = None
    agent_id: Optional[str] = None
    full_log: Optional[str] = None

class PredictionRequest(BaseModel):
    logs: List[LogData] = Field(..., description="日誌資料列表")

class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="預測的攻擊鏈類型")
    confidence: float = Field(..., description="信心度分數")
    similarities: Dict[str, float] = Field(..., description="與各攻擊鏈的相似度")
    processing_time: float = Field(..., description="處理時間(秒)")
    alert_generated: bool = Field(..., description="是否產生告警")

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    wazuh_connection: bool

@app.on_event("startup")
async def startup_event():
    """應用啟動事件"""
    global model_inference
    try:
        logger.info("正在初始化模型推論服務...")
        model_inference = ModelInference()
        await model_inference.initialize()
        logger.info("模型推論服務初始化完成")
    except Exception as e:
        logger.error(f"模型初始化失敗: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """應用關閉事件"""
    logger.info("正在關閉攻擊鏈檢測系統...")

async def get_model_inference():
    """依賴注入：獲取模型推論實例"""
    if model_inference is None:
        raise HTTPException(status_code=503, detail="模型尚未初始化")
    return model_inference

@app.get("/", response_class=JSONResponse)
async def root():
    """根路徑"""
    return {
        "message": "攻擊鏈檢測系統 API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康檢查端點"""
    try:
        # 檢查模型狀態
        model_loaded = model_inference is not None and model_inference.is_ready()
        
        # 檢查Wazuh連接（簡單測試）
        wazuh_connection = True
        try:
            async with WazuhClient() as client:
                # 嘗試獲取1分鐘內的資料來測試連接
                await client.fetch_recent_alerts(minutes=1)
        except:
            wazuh_connection = False
        
        status = "healthy" if model_loaded and wazuh_connection else "degraded"
        
        return HealthResponse(
            status=status,
            timestamp=datetime.now().isoformat(),
            model_loaded=model_loaded,
            wazuh_connection=wazuh_connection
        )
    except Exception as e:
        logger.error(f"健康檢查失敗: {e}")
        raise HTTPException(status_code=500, detail="健康檢查失敗")

@app.post("/predict", response_model=PredictionResponse)
async def predict_attack_chain(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    inference: ModelInference = Depends(get_model_inference)
):
    """預測攻擊鏈類型"""
    start_time = datetime.now()
    
    try:
        # 轉換請求格式
        logs = []
        for log_data in request.logs:
            log = {
                "timestamp": log_data.timestamp,
                "rule.description": log_data.rule_description,
                "data.srcip": log_data.src_ip,
                "data.dstip": log_data.dst_ip,
                "rule.id": log_data.rule_id,
                "rule.level": log_data.rule_level,
                "agent.id": log_data.agent_id,
                "full_log": log_data.full_log
            }
            logs.append(log)
        
        # 執行預測
        result = await inference.predict_logs(logs)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 記錄預測結果
        api_logger.log_prediction(
            result["prediction"], 
            result["confidence"], 
            processing_time
        )
        
        # 如果信心度低，背景任務保存資料
        if result["confidence"] < settings.MODEL_CONFIDENCE_THRESHOLD:
            background_tasks.add_task(
                save_low_confidence_event,
                logs,
                result["prediction"],
                result["confidence"]
            )
        
        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            similarities=result["similarities"],
            processing_time=processing_time,
            alert_generated=result["confidence"] >= settings.MODEL_CONFIDENCE_THRESHOLD
        )
        
    except Exception as e:
        logger.error(f"預測失敗: {e}")
        raise HTTPException(status_code=500, detail=f"預測失敗: {str(e)}")

@app.post("/predict/wazuh")
async def predict_from_wazuh(
    background_tasks: BackgroundTasks,
    minutes: int = 5,
    inference: ModelInference = Depends(get_model_inference)
):
    """從Wazuh獲取資料並預測"""
    start_time = datetime.now()
    
    try:
        # 從Wazuh獲取資料
        async with WazuhClient() as client:
            alerts = await client.fetch_recent_alerts(minutes=minutes)
        
        if not alerts:
            return JSONResponse(
                content={
                    "message": "未獲取到資料",
                    "alerts_count": 0,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            )
        
        # 執行預測
        result = await inference.predict_logs(alerts)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 記錄預測結果
        api_logger.log_prediction(
            result["prediction"], 
            result["confidence"], 
            processing_time
        )
        
        # 如果信心度低，背景任務保存資料
        if result["confidence"] < settings.MODEL_CONFIDENCE_THRESHOLD:
            background_tasks.add_task(
                save_low_confidence_event,
                alerts,
                result["prediction"],
                result["confidence"]
            )
        
        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "similarities": result["similarities"],
            "alerts_count": len(alerts),
            "processing_time": processing_time,
            "alert_generated": result["confidence"] >= settings.MODEL_CONFIDENCE_THRESHOLD
        }
        
    except Exception as e:
        logger.error(f"從Wazuh預測失敗: {e}")
        raise HTTPException(status_code=500, detail=f"預測失敗: {str(e)}")

@app.get("/model/status")
async def model_status(inference: ModelInference = Depends(get_model_inference)):
    """獲取模型狀態"""
    try:
        status = await inference.get_status()
        return status
    except Exception as e:
        logger.error(f"獲取模型狀態失敗: {e}")
        raise HTTPException(status_code=500, detail="獲取模型狀態失敗")

@app.post("/model/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """手動觸發模型重訓練"""
    try:
        background_tasks.add_task(perform_retrain)
        return {"message": "重訓練任務已排程", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"觸發重訓練失敗: {e}")
        raise HTTPException(status_code=500, detail="觸發重訓練失敗")

@app.get("/logs/low_confidence")
async def get_low_confidence_logs(limit: int = 100):
    """獲取低信心度事件日誌"""
    try:
        logs_dir = settings.LOW_CONFIDENCE_DIR
        log_files = sorted(logs_dir.glob("*.json"), reverse=True)
        
        all_events = []
        count = 0
        
        for log_file in log_files:
            if count >= limit:
                break
                
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    events = json.load(f)
                    if isinstance(events, list):
                        all_events.extend(events[:limit-count])
                        count += len(events)
                    else:
                        all_events.append(events)
                        count += 1
            except Exception as e:
                logger.warning(f"讀取日誌檔案失敗 {log_file}: {e}")
                continue
        
        return {
            "events": all_events[:limit],
            "total_returned": len(all_events[:limit]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"獲取低信心度日誌失敗: {e}")
        raise HTTPException(status_code=500, detail="獲取低信心度日誌失敗")

async def save_low_confidence_event(logs: List[Dict], prediction: str, confidence: float):
    """背景任務：保存低信心度事件"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = settings.LOW_CONFIDENCE_DIR / f"low_conf_{timestamp}_{confidence:.4f}.json"
        
        event_data = {
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
            "confidence": confidence,
            "logs": logs
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(event_data, f, ensure_ascii=False, indent=2)
        
        api_logger.log_low_confidence(prediction, confidence, str(file_path))
        
    except Exception as e:
        logger.error(f"保存低信心度事件失敗: {e}")

async def perform_retrain():
    """背景任務：執行模型重訓練"""
    try:
        api_logger.log_retrain_start()
        
        from ..models.trainer import ModelTrainer
        trainer = ModelTrainer()
        
        # 執行重訓練
        result = await trainer.retrain_model()
        
        if result["success"]:
            api_logger.log_retrain_complete(
                result["model_version"], 
                result["accuracy"]
            )
        else:
            logger.error(f"重訓練失敗: {result['error']}")
            
    except Exception as e:
        logger.error(f"重訓練背景任務失敗: {e}")

# 異常處理器
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP異常處理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用異常處理器"""
    logger.error(f"未處理的異常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "內部伺服器錯誤",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

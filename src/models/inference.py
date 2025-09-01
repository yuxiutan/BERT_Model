"""
模型推論模組
"""

import json
import pickle
import torch
import numpy as np
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from transformers import BertForSequenceClassification, BertTokenizer, BertModel

from ..utils.config import settings
from ..utils.logger import setup_logger, api_logger

logger = setup_logger(__name__)

class ModelInference:
    """模型推論器"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.bert_model = None
        self.reference_embeddings = None
        self._initialized = False
    
    async def initialize(self):
        """初始化模型和相關組件"""
        try:
            logger.info("正在初始化模型推論器...")
            
            # 檢查是否存在訓練好的模型
            model_dir = settings.MODEL_DIR
            if not (model_dir / "config.json").exists():
                logger.warning("未找到訓練好的模型，開始初始訓練...")
                from .trainer import ModelTrainer
                trainer = ModelTrainer()
                if not trainer.initial_training():
                    raise Exception("初始訓練失敗")
            
            # 載入tokenizer和BERT模型
            self.tokenizer = BertTokenizer.from_pretrained(settings.MODEL_NAME)
            self.bert_model = BertModel.from_pretrained(settings.MODEL_NAME)
            self.bert_model.eval()
            
            # 載入分類模型
            self.model = BertForSequenceClassification.from_pretrained(str(model_dir))
            self.model.eval()
            
            # 載入參考嵌入向量
            ref_embeddings_file = model_dir / settings.REFERENCE_EMBEDDINGS_FILE
            if ref_embeddings_file.exists():
                with open(ref_embeddings_file, "rb") as f:
                    self.reference_embeddings = pickle.load(f)
            else:
                logger.warning("參考嵌入向量檔案不存在")
                self.reference_embeddings = None
            
            self._initialized = True
            logger.info("模型推論器初始化完成")
            
        except Exception as e:
            logger.error(f"模型初始化失敗: {e}")
            raise
    
    def is_ready(self) -> bool:
        """檢查模型是否準備就緒"""
        return (self._initialized and 
                self.model is not None and 
                self.tokenizer is not None and
                self.bert_model is not None)
    
    def extract_features(self, logs: List[Dict[str, Any]]) -> str:
        """從日誌中提取特徵序列"""
        sequences = []
        for log in logs:
            desc = log.get("rule.description", "")
            src_ip = log.get("data.srcip", "None")
            dst_ip = log.get("data.dstip", "None")
            sequence = f"{desc} from {src_ip} to {dst_ip}"
            sequences.append(sequence)
        
        full_sequence = " ".join(sequences)
        return full_sequence
    
    def get_embedding(self, sequence: str) -> np.ndarray:
        """獲取序列的BERT嵌入向量"""
        inputs = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=settings.MAX_LENGTH,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        return embedding
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """計算餘弦相似度"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def predict_attack_chain(self, sequence: str) -> Tuple[str, Dict[str, float], float]:
        """預測攻擊鏈類型"""
        if not self.reference_embeddings:
            # 使用分類模型預測
            return self._predict_with_classifier(sequence)
        
        # 使用相似度方法預測
        emb = self.get_embedding(sequence)
        four_sim = self.cosine_similarity(emb, self.reference_embeddings["four_embedding"])
        apt_sim = self.cosine_similarity(emb, self.reference_embeddings["apt_embedding"])
        
        max_sim = max(four_sim, apt_sim)
        
        if max_sim < 0.5:
            pred_chain = "Unknown"
        else:
            pred_chain = "FourInOne" if four_sim > apt_sim else "APT29"
        
        similarities = {"FourInOne": four_sim, "APT29": apt_sim}
        
        return pred_chain, similarities, max_sim
    
    def _predict_with_classifier(self, sequence: str) -> Tuple[str, Dict[str, float], float]:
        """使用分類模型進行預測"""
        inputs = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=settings.MAX_LENGTH,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
            pred_label = np.argmax(probs)
        
        # 標籤映射
        label_map = {0: "FourInOne", 1: "APT29"}
        pred_chain = label_map.get(pred_label, "Unknown")
        confidence = float(probs[pred_label])
        
        similarities = {
            "FourInOne": float(probs[0]),
            "APT29": float(probs[1])
        }
        
        return pred_chain, similarities, confidence
    
    async def predict_logs(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """對日誌列表進行預測"""
        if not self.is_ready():
            raise Exception("模型尚未初始化")
        
        try:
            # 提取特徵序列
            sequence = self.extract_features(logs)
            
            # 執行預測
            prediction, similarities, confidence = self.predict_attack_chain(sequence)
            
            # 如果信心度很高，發送告警
            if confidence >= settings.MODEL_CONFIDENCE_THRESHOLD:
                await self._send_alert(logs, prediction, confidence, similarities)
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "similarities": similarities,
                "sequence_length": len(sequence),
                "logs_count": len(logs)
            }
            
        except Exception as e:
            logger.error(f"預測失敗: {e}")
            raise
    
    async def _send_alert(self, logs: List[Dict], prediction: str, confidence: float, similarities: Dict[str, float]):
        """發送告警到外部API"""
        try:
            # 準備告警資料
            now = datetime.now()
            formatted_time = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+0000"
            
            # 找到最高信心度的日誌作為主要告警
            if logs:
                main_log = logs[0]  # 簡化處理，取第一個
                
                # 準備個別告警
                alerts = []
                for log in logs:
                    alert = {
                        "timestamp": log.get("timestamp", formatted_time),
                        "rule_id": log.get("rule.id", ""),
                        "rule_level": log.get("rule.level", ""),
                        "description": log.get("rule.description", ""),
                        "src_ip": log.get("data.srcip", "None"),
                        "dst_ip": log.get("data.dstip", "None"),
                        "full_log": log.get("full_log", "")
                    }
                    alerts.append(alert)
                
                # 準備關聯告警
                involved_info = [
                    f"Agent: {log.get('agent.id', 'Unknown')}, "
                    f"Src: {log.get('data.srcip', 'None')}, "
                    f"Dst: {log.get('data.dstip', 'None')}, "
                    f"Confidence: {confidence:.4f}"
                    for log in logs[:5]  # 最多顯示5個
                ]
                
                correlation_alert = {
                    "timestamp": formatted_time,
                    "rule_id": 51110,
                    "rule_level": 12,
                    "description": f"攻擊鏈檢測: {prediction}",
                    "src_ip": main_log.get("data.srcip", "None"),
                    "dst_ip": main_log.get("data.dstip", "None"),
                    "full_log": f"檢測到 {prediction} 攻擊鏈，信心度: {confidence:.4f}，涉及事件: {'; '.join(involved_info)}"
                }
                
                # 組織API資料
                api_data = {
                    "data": [{
                        "correlation_alert": correlation_alert,
                        "alerts": alerts
                    }]
                }
                
                # 發送到告警API
                headers = {"Content-Type": "application/json"}
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        settings.ALERT_API_URL,
                        json=api_data,
                        headers=headers,
                        timeout=settings.ALERT_API_TIMEOUT
                    ) as response:
                        if response.status == 200:
                            logger.info(f"告警發送成功: {prediction} (信心度: {confidence:.4f})")
                        else:
                            logger.warning(f"告警發送失敗，狀態碼: {response.status}")
            
            # 發送Discord通知（如果配置了）
            if settings.DISCORD_WEBHOOK_URL:
                await self._send_discord_notification(prediction, confidence, len(logs))
                
        except Exception as e:
            logger.error(f"發送告警失敗: {e}")
    
    async def _send_discord_notification(self, prediction: str, confidence: float, logs_count: int):
        """發送Discord通知"""
        try:
            import aiohttp
            
            embed = {
                "title": "🚨 攻擊鏈檢測告警",
                "description": f"檢測到 **{prediction}** 攻擊鏈",
                "color": 0xff0000 if confidence > 0.7 else 0xff9900,
                "fields": [
                    {"name": "信心度", "value": f"{confidence:.4f}", "inline": True},
                    {"name": "事件數量", "value": str(logs_count), "inline": True},
                    {"name": "時間", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "inline": True}
                ],
                "footer": {"text": "攻擊鏈檢測系統"}
            }
            
            webhook_data = {"embeds": [embed]}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(settings.DISCORD_WEBHOOK_URL, json=webhook_data) as response:
                    if response.status == 204:
                        logger.info("Discord通知發送成功")
                    else:
                        logger.warning(f"Discord通知發送失敗，狀態碼: {response.status}")
                        
        except Exception as e:
            logger.error(f"Discord通知發送失敗: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """獲取模型狀態資訊"""
        try:
            model_config_file = settings.MODEL_DIR / settings.MODEL_CONFIG_FILE
            
            if model_config_file.exists():
                with open(model_config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                config = {}
            
            return {
                "initialized": self._initialized,
                "model_ready": self.is_ready(),
                "model_version": config.get("version", "unknown"),
                "model_timestamp": config.get("timestamp", "unknown"),
                "confidence_threshold": settings.MODEL_CONFIDENCE_THRESHOLD,
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"獲取模型狀態失敗: {e}")
            return {
                "initialized": False,
                "model_ready": False,
                "error": str(e)
            }

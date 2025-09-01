"""
資料前處理模組
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from transformers import BertTokenizer, BertModel
import torch
from sklearn.model_selection import train_test_split

from ..utils.config import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class DataPreprocessor:
    """資料前處理器"""
    
    def __init__(self):
        self.tokenizer = None
        self.bert_model = None
    
    def load_bert_model(self):
        """載入BERT模型和tokenizer"""
        try:
            self.tokenizer = BertTokenizer.from_pretrained(settings.MODEL_NAME)
            self.bert_model = BertModel.from_pretrained(settings.MODEL_NAME)
            self.bert_model.eval()
            logger.info("BERT模型載入成功")
        except Exception as e:
            logger.error(f"BERT模型載入失敗: {e}")
            raise
    
    def load_logs(self, file_path: Path) -> List[Dict[str, Any]]:
        """載入日誌資料"""
        if not file_path.exists():
            raise FileNotFoundError(f"日誌檔案不存在: {file_path}")
        
        logs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    log = json.loads(line.strip())
                    logs.append(log)
                except json.JSONDecodeError as e:
                    logger.warning(f"跳過無效的JSON行: {e}")
        
        logger.info(f"載入 {len(logs)} 條日誌資料")
        return logs
    
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
    
    def generate_synthetic_sequences(self, base_seq: str, label: int, num_samples: int = 120) -> List[Dict[str, Any]]:
        """生成合成資料序列"""
        data = []
        words = base_seq.split()
        
        for i in range(num_samples):
            # 隨機打亂並截斷
            np.random.shuffle(words)
            perturbed = " ".join(words[:int(len(words) * 0.9)])
            data.append({"sequence": perturbed, "label": label})
        
        logger.info(f"生成 {num_samples} 個標籤為 {label} 的合成序列")
        return data
    
    def tokenize_data(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict], np.ndarray]:
        """對資料進行tokenization"""
        if not self.tokenizer:
            self.load_bert_model()
        
        tokenized_data = []
        for item in data:
            tokenized = self.tokenizer(
                item["sequence"], 
                padding="max_length", 
                truncation=True, 
                max_length=settings.MAX_LENGTH, 
                return_tensors="pt"
            )
            tokenized_data.append(tokenized)
        
        labels = np.array([item["label"] for item in data])
        
        logger.info(f"完成 {len(tokenized_data)} 個序列的tokenization")
        return tokenized_data, labels
    
    def get_embedding(self, sequence: str) -> np.ndarray:
        """獲取序列的BERT嵌入向量"""
        if not self.bert_model or not self.tokenizer:
            self.load_bert_model()
        
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
    
    def prepare_training_data(self) -> bool:
        """準備訓練資料"""
        try:
            logger.info("開始準備訓練資料...")
            
            # 載入參考攻擊鏈資料
            four_in_one_file = settings.REFERENCE_DIR / "attack_chain_FourInOne.json"
            apt29_file = settings.REFERENCE_DIR / "attack_chain_APT29.json"
            
            if not four_in_one_file.exists() or not apt29_file.exists():
                logger.error("參考攻擊鏈檔案不存在")
                return False
            
            # 載入資料
            four_logs = self.load_logs(four_in_one_file)
            apt_logs = self.load_logs(apt29_file)
            
            # 提取特徵
            four_seq = self.extract_features(four_logs)
            apt_seq = self.extract_features(apt_logs)
            
            # 生成合成資料
            synthetic_four = self.generate_synthetic_sequences(four_seq, 0)
            synthetic_apt = self.generate_synthetic_sequences(apt_seq, 1)
            
            # 合併資料
            all_data = synthetic_four + synthetic_apt
            np.random.shuffle(all_data)  # 隨機打亂
            
            # Tokenization
            tokenized_data, labels = self.tokenize_data(all_data)
            
            # 分割訓練/測試集
            train_inputs, test_inputs, train_labels, test_labels = train_test_split(
                tokenized_data, labels, test_size=0.2, random_state=42
            )
            
            # 保存處理後的資料
            processed_data = {
                "train_inputs": train_inputs,
                "test_inputs": test_inputs,
                "train_labels": train_labels,
                "test_labels": test_labels
            }
            
            processed_data_file = settings.MODEL_DIR / settings.PROCESSED_DATA_FILE
            with open(processed_data_file, "wb") as f:
                pickle.dump(processed_data, f)
            
            # 生成並保存參考嵌入
            four_embedding = self.get_embedding(four_seq)
            apt_embedding = self.get_embedding(apt_seq)
            
            reference_embeddings = {
                "four_embedding": four_embedding,
                "apt_embedding": apt_embedding
            }
            
            ref_embeddings_file = settings.MODEL_DIR / settings.REFERENCE_EMBEDDINGS_FILE
            with open(ref_embeddings_file, "wb") as f:
                pickle.dump(reference_embeddings, f)
            
            logger.info("訓練資料準備完成")
            logger.info(f"訓練集大小: {len(train_inputs)}")
            logger.info(f"測試集大小: {len(test_inputs)}")
            
            return True
            
        except Exception as e:
            logger.error(f"準備訓練資料失敗: {e}")
            return False
    
    def prepare_retrain_data(self) -> bool:
        """準備重訓練資料（包含低信心度事件）"""
        try:
            logger.info("開始準備重訓練資料...")
            
            # 載入原始參考資料
            if not self.prepare_training_data():
                return False
            
            # 載入低信心度事件
            low_conf_files = list(settings.LOW_CONFIDENCE_DIR.glob("*.json"))
            additional_logs = []
            
            for file_path in low_conf_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        event_data = json.load(f)
                        if isinstance(event_data, dict) and "logs" in event_data:
                            additional_logs.extend(event_data["logs"])
                except Exception as e:
                    logger.warning(f"讀取低信心度檔案失敗 {file_path}: {e}")
            
            if additional_logs:
                logger.info(f"找到 {len(additional_logs)} 個額外的低信心度事件")
                
                # 這裡可以根據需要對低信心度事件進行標註
                # 目前先跳過，因為需要人工標註或更複雜的自動標註邏輯
                
            logger.info("重訓練資料準備完成")
            return True
            
        except Exception as e:
            logger.error(f"準備重訓練資料失敗: {e}")
            return False

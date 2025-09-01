"""
模型訓練模組
"""

import json
import pickle
import shutil
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from transformers import BertForSequenceClassification, BertTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

from ..utils.config import settings
from ..utils.logger import setup_logger, api_logger
from .preprocessor import DataPreprocessor

logger = setup_logger(__name__)

class ModelTrainer:
    """模型訓練器"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.model = None
        self.tokenizer = None
    
    def backup_current_model(self) -> str:
        """備份當前模型"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_version = f"model_v{timestamp}"
            backup_dir = settings.MODEL_BACKUP_DIR / backup_version
            
            # 創建備份目錄
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 備份模型檔案
            current_model_dir = settings.MODEL_DIR
            if current_model_dir.exists():
                # 複製所有模型檔案
                for file_path in current_model_dir.iterdir():
                    if file_path.is_file():
                        shutil.copy2(file_path, backup_dir / file_path.name)
                
                logger.info(f"模型備份完成: {backup_version}")
                api_logger.log_model_backup("current", backup_version)
                
                # 清理舊的備份（保留最新3個）
                self._cleanup_old_backups()
                
                return backup_version
            else:
                logger.warning("當前模型目錄不存在，跳過備份")
                return ""
                
        except Exception as e:
            logger.error(f"模型備份失敗: {e}")
            return ""
    
    def _cleanup_old_backups(self):
        """清理舊的模型備份"""
        try:
            backup_dirs = sorted(
                [d for d in settings.MODEL_BACKUP_DIR.iterdir() if d.is_dir()],
                key=lambda x: x.name,
                reverse=True
            )
            
            # 保留最新的3個備份
            if len(backup_dirs) > settings.MAX_MODEL_BACKUPS:
                for old_backup in backup_dirs[settings.MAX_MODEL_BACKUPS:]:
                    shutil.rmtree(old_backup)
                    logger.info(f"刪除舊備份: {old_backup.name}")
                    
        except Exception as e:
            logger.error(f"清理舊備份失敗: {e}")
    
    def load_training_data(self) -> tuple:
        """載入訓練資料"""
        try:
            processed_data_file = settings.MODEL_DIR / settings.PROCESSED_DATA_FILE
            
            if not processed_data_file.exists():
                logger.info("處理後的資料不存在，開始準備訓練資料...")
                if not self.preprocessor.prepare_training_data():
                    raise Exception("準備訓練資料失敗")
            
            with open(processed_data_file, "rb") as f:
                data = pickle.load(f)
            
            return (
                data["train_inputs"],
                data["test_inputs"], 
                data["train_labels"],
                data["test_labels"]
            )
            
        except Exception as e:
            logger.error(f"載入訓練資料失敗: {e}")
            raise
    
    def create_data_loader(self, inputs: List, labels: np.ndarray, shuffle: bool = True) -> DataLoader:
        """創建DataLoader"""
        try:
            input_ids = torch.cat([item['input_ids'] for item in inputs], dim=0)
            attention_masks = torch.cat([item['attention_mask'] for item in inputs], dim=0)
            labels_tensor = torch.tensor(labels)
            
            dataset = TensorDataset(input_ids, attention_masks, labels_tensor)
            dataloader = DataLoader(
                dataset, 
                batch_size=settings.BATCH_SIZE, 
                shuffle=shuffle
            )
            
            return dataloader
            
        except Exception as e:
            logger.error(f"創建DataLoader失敗: {e}")
            raise
    
    def train_model(self, train_loader: DataLoader, test_loader: DataLoader) -> Dict[str, Any]:
        """訓練模型"""
        try:
            logger.info("開始訓練模型...")
            
            # 初始化模型
            self.model = BertForSequenceClassification.from_pretrained(
                settings.MODEL_NAME,
                num_labels=settings.NUM_LABELS,
                hidden_dropout_prob=settings.DROPOUT_PROB
            )
            
            # 設定優化器
            optimizer = AdamW(self.model.parameters(), lr=settings.LEARNING_RATE)
            
            # 訓練迴圈
            self.model.train()
            train_losses = []
            
            for epoch in range(settings.NUM_EPOCHS):
                total_loss = 0
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{settings.NUM_EPOCHS}")
                
                for batch in progress_bar:
                    b_input_ids, b_attention_mask, b_labels = batch
                    
                    optimizer.zero_grad()
                    outputs = self.model(
                        input_ids=b_input_ids,
                        attention_mask=b_attention_mask,
                        labels=b_labels
                    )
                    
                    loss = outputs.loss
                    total_loss += loss.item()
                    
                    loss.backward()
                    optimizer.step()
                    
                    progress_bar.set_postfix({"batch_loss": loss.item()})
                
                avg_loss = total_loss / len(train_loader)
                train_losses.append(avg_loss)
                logger.info(f"Epoch {epoch+1} 平均損失: {avg_loss:.4f}")
            
            # 評估模型
            accuracy = self.evaluate_model(test_loader)
            
            # 保存訓練結果
            training_result = {
                "train_losses": train_losses,
                "final_accuracy": accuracy,
                "num_epochs": settings.NUM_EPOCHS,
                "timestamp": datetime.now().isoformat()
            }
            
            return training_result
            
        except Exception as e:
            logger.error(f"模型訓練失敗: {e}")
            raise
    
    def evaluate_model(self, test_loader: DataLoader) -> float:
        """評估模型性能"""
        try:
            self.model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    b_input_ids, b_attention_mask, b_labels = batch
                    
                    outputs = self.model(
                        input_ids=b_input_ids,
                        attention_mask=b_attention_mask
                    )
                    
                    preds = torch.argmax(outputs.logits, dim=1).numpy()
                    all_preds.extend(preds)
                    all_labels.extend(b_labels.numpy())
            
            accuracy = accuracy_score(all_labels, all_preds)
            logger.info(f"模型準確率: {accuracy:.4f}")
            
            return accuracy
            
        except Exception as e:
            logger.error(f"模型評估失敗: {e}")
            return 0.0
    
    def save_model(self, version: str = None) -> str:
        """保存模型"""
        try:
            if version is None:
                version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存模型
            self.model.save_pretrained(settings.MODEL_DIR)
            
            # 保存配置資訊
            model_config = {
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "model_name": settings.MODEL_NAME,
                "num_labels": settings.NUM_LABELS,
                "max_length": settings.MAX_LENGTH,
                "confidence_threshold": settings.MODEL_CONFIDENCE_THRESHOLD
            }
            
            config_file = settings.MODEL_DIR / settings.MODEL_CONFIG_FILE
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(model_config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"模型保存完成: {version}")
            return version
            
        except Exception as e:
            logger.error(f"模型保存失敗: {e}")
            raise
    
    async def retrain_model(self) -> Dict[str, Any]:
        """執行模型重訓練"""
        try:
            logger.info("開始執行模型重訓練...")
            
            # 1. 備份當前模型
            backup_version = self.backup_current_model()
            
            # 2. 準備重訓練資料
            if not self.preprocessor.prepare_retrain_data():
                return {
                    "success": False,
                    "error": "準備重訓練資料失敗"
                }
            
            # 3. 載入訓練資料
            train_inputs, test_inputs, train_labels, test_labels = self.load_training_data()
            
            # 4. 創建DataLoader
            train_loader = self.create_data_loader(train_inputs, train_labels)
            test_loader = self.create_data_loader(test_inputs, test_labels, shuffle=False)
            
            # 5. 訓練模型
            training_result = self.train_model(train_loader, test_loader)
            
            # 6. 保存新模型
            new_version = self.save_model()
            
            logger.info("模型重訓練完成")
            
            return {
                "success": True,
                "model_version": new_version,
                "backup_version": backup_version,
                "accuracy": training_result["final_accuracy"],
                "train_losses": training_result["train_losses"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"模型重訓練失敗: {e}")
            
            # 嘗試恢復備份
            try:
                if backup_version:
                    self.restore_backup(backup_version)
                    logger.info("已恢復備份模型")
            except Exception as restore_error:
                logger.error(f"恢復備份失敗: {restore_error}")
            
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def restore_backup(self, backup_version: str):
        """恢復指定版本的備份"""
        try:
            backup_dir = settings.MODEL_BACKUP_DIR / backup_version
            
            if not backup_dir.exists():
                raise FileNotFoundError(f"備份版本不存在: {backup_version}")
            
            # 清空當前模型目錄
            if settings.MODEL_DIR.exists():
                shutil.rmtree(settings.MODEL_DIR)
            settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            
            # 復制備份檔案
            for file_path in backup_dir.iterdir():
                if file_path.is_file():
                    shutil.copy2(file_path, settings.MODEL_DIR / file_path.name)
            
            logger.info(f"模型恢復完成: {backup_version}")
            
        except Exception as e:
            logger.error(f"模型恢復失敗: {e}")
            raise
    
    def get_available_backups(self) -> List[str]:
        """獲取可用的備份版本"""
        try:
            backup_dirs = sorted(
                [d.name for d in settings.MODEL_BACKUP_DIR.iterdir() if d.is_dir()],
                reverse=True
            )
            return backup_dirs
        except Exception as e:
            logger.error(f"獲取備份列表失敗: {e}")
            return []
    
    def initial_training(self) -> bool:
        """初始訓練（第一次部署時使用）"""
        try:
            logger.info("開始初始模型訓練...")
            
            # 準備資料
            if not self.preprocessor.prepare_training_data():
                return False
            
            # 載入資料並訓練
            train_inputs, test_inputs, train_labels, test_labels = self.load_training_data()
            train_loader = self.create_data_loader(train_inputs, train_labels)
            test_loader = self.create_data_loader(test_inputs, test_labels, shuffle=False)
            
            # 訓練模型
            training_result = self.train_model(train_loader, test_loader)
            
            # 保存模型
            version = self.save_model("initial_v1.0")
            
            logger.info(f"初始訓練完成，模型版本: {version}")
            logger.info(f"初始模型準確率: {training_result['final_accuracy']:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"初始訓練失敗: {e}")
            return False

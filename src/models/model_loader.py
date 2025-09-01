"""
預訓練模型載入模組
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from ..utils.config import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class PretrainedModelLoader:
    """預訓練模型載入器"""
    
    def __init__(self):
        self.pretrained_path = Path("/app/pretrained_model")
        self.current_model_path = settings.MODEL_DIR
    
    def check_pretrained_model(self) -> bool:
        """檢查是否存在預訓練模型"""
        required_files = ["config.json"]  # 最基本需要config.json
        
        if not self.pretrained_path.exists():
            logger.info("預訓練模型目錄不存在")
            return False
        
        for file_name in required_files:
            if not (self.pretrained_path / file_name).exists():
                logger.info(f"預訓練模型缺少必要檔案: {file_name}")
                return False
        
        logger.info("發現預訓練模型")
        return True
    
    def load_pretrained_model(self) -> bool:
        """載入預訓練模型到當前模型目錄"""
        try:
            if not self.check_pretrained_model():
                return False
            
            logger.info("開始載入預訓練模型...")
            
            # 確保當前模型目錄存在
            self.current_model_path.mkdir(parents=True, exist_ok=True)
            
            # 複製所有模型檔案
            copied_files = []
            for file_path in self.pretrained_path.iterdir():
                if file_path.is_file():
                    dest_path = self.current_model_path / file_path.name
                    shutil.copy2(file_path, dest_path)
                    copied_files.append(file_path.name)
                    logger.info(f"複製模型檔案: {file_path.name}")
            
            # 驗證關鍵檔案
            if self._validate_model_files():
                logger.info(f"預訓練模型載入成功，共複製 {len(copied_files)} 個檔案")
                self._log_model_info()
                return True
            else:
                logger.error("模型檔案驗證失敗")
                return False
                
        except Exception as e:
            logger.error(f"載入預訓練模型失敗: {e}")
            return False
    
    def _validate_model_files(self) -> bool:
        """驗證模型檔案完整性"""
        try:
            config_file = self.current_model_path / "config.json"
            
            if not config_file.exists():
                logger.error("config.json 檔案不存在")
                return False
            
            # 檢查配置檔案格式
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 檢查必要的配置項
            required_keys = ["model_type", "num_labels"]
            for key in required_keys:
                if key not in config:
                    logger.warning(f"配置檔案缺少必要項目: {key}")
            
            # 檢查模型權重檔案
            weight_files = [
                "pytorch_model.bin",
                "model.safetensors",
                "tf_model.h5"
            ]
            
            has_weights = any(
                (self.current_model_path / weight_file).exists() 
                for weight_file in weight_files
            )
            
            if not has_weights:
                logger.warning("未找到模型權重檔案，可能需要重新訓練")
            
            return True
            
        except Exception as e:
            logger.error(f"驗證模型檔案失敗: {e}")
            return False
    
    def _log_model_info(self):
        """記錄模型資訊"""
        try:
            config_file = self.current_model_path / "config.json"
            
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                logger.info("預訓練模型資訊:")
                logger.info(f"  模型類型: {config.get('model_type', 'unknown')}")
                logger.info(f"  標籤數量: {config.get('num_labels', 'unknown')}")
                logger.info(f"  隱藏層大小: {config.get('hidden_size', 'unknown')}")
                logger.info(f"  架構ID: {config.get('architectures', 'unknown')}")
                
                # 檢查是否有自訂配置
                model_config_file = self.current_model_path / "model_config.json"
                if model_config_file.exists():
                    with open(model_config_file, 'r', encoding='utf-8') as f:
                        model_config = json.load(f)
                    logger.info(f"  模型版本: {model_config.get('version', 'unknown')}")
                    logger.info(f"  訓練時間: {model_config.get('timestamp', 'unknown')}")
                    
        except Exception as e:
            logger.warning(f"讀取模型資訊失敗: {e}")
    
    def get_model_metadata(self) -> Optional[Dict[str, Any]]:
        """獲取模型元資料"""
        try:
            metadata = {}
            
            # 讀取基本配置
            config_file = self.current_model_path / "config.json"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                metadata["model_config"] = config
            
            # 讀取自訂配置
            model_config_file = self.current_model_path / "model_config.json"  
            if model_config_file.exists():
                with open(model_config_file, 'r', encoding='utf-8') as f:
                    model_config = json.load(f)
                metadata["custom_config"] = model_config
            
            # 檢查檔案狀態
            files = {}
            for file_path in self.current_model_path.iterdir():
                if file_path.is_file():
                    files[file_path.name] = {
                        "size": file_path.stat().st_size,
                        "modified": file_path.stat().st_mtime
                    }
            metadata["files"] = files
            
            return metadata
            
        except Exception as e:
            logger.error(f"獲取模型元資料失敗: {e}")
            return None

def initialize_model_from_pretrained() -> bool:
    """從預訓練模型初始化當前模型"""
    loader = PretrainedModelLoader()
    
    # 如果當前模型目錄已有模型，跳過載入
    if (settings.MODEL_DIR / "config.json").exists():
        logger.info("當前模型目錄已有模型，跳過預訓練模型載入")
        return True
    
    # 載入預訓練模型
    return loader.load_pretrained_model()

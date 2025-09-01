"""
Wazuh API 客戶端
"""

import json
import asyncio
import aiohttp
import ssl
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from .config import settings
from .logger import setup_logger

logger = setup_logger(__name__)

class WazuhClient:
    """Wazuh API 客戶端"""
    
    def __init__(self):
        self.base_url = settings.WAZUH_API_URL
        self.username = settings.WAZUH_API_USERNAME
        self.password = settings.WAZUH_API_PASSWORD
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """異步上下文管理器進入"""
        await self._create_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """異步上下文管理器退出"""
        await self._close_session()
    
    async def _create_session(self):
        """創建HTTP會話"""
        # 忽略SSL證書驗證（僅用於測試環境）
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        auth = aiohttp.BasicAuth(self.username, self.password)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            auth=auth,
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    async def _close_session(self):
        """關閉HTTP會話"""
        if self.session:
            await self.session.close()
    
    async def fetch_recent_alerts(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """獲取最近N分鐘的告警資料"""
        if not self.session:
            await self._create_session()
        
        try:
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "range": {
                                    "timestamp": {
                                        "gte": f"now-{minutes}m",
                                        "lte": "now"
                                    }
                                }
                            },
                            {
                                "terms": {
                                    "agent.name": [
                                        "DESKTOP-66TG6SE",
                                        "DESKTOP-66G2GGG",
                                        "connector-node"
                                    ]
                                }
                            }
                        ]
                    }
                },
                "_source": [
                    "timestamp",
                    "agent.ip",
                    "agent.name", 
                    "agent.id",
                    "rule.id",
                    "rule.mitre.id",
                    "rule.level",
                    "rule.description",
                    "data.srcip",
                    "data.dstip",
                    "full_log"
                ],
                "sort": [
                    {"timestamp": {"order": "asc"}}
                ],
                "size": 10000
            }
            
            url = f"{self.base_url}/{settings.WAZUH_INDEX}/_search"
            headers = {"Content-Type": "application/json"}
            
            async with self.session.post(url, json=query, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    hits = result.get("hits", {}).get("hits", [])
                    
                    # 轉換格式
                    alerts = []
                    for hit in hits:
                        source = hit.get("_source", {})
                        alert = {
                            "timestamp": source.get("timestamp"),
                            "agent.ip": source.get("agent", {}).get("ip"),
                            "agent.name": source.get("agent", {}).get("name"),
                            "agent.id": source.get("agent", {}).get("id"),
                            "rule.id": source.get("rule", {}).get("id"),
                            "rule.mitre.id": source.get("rule", {}).get("mitre", {}).get("id", "T0000"),
                            "rule.level": source.get("rule", {}).get("level"),
                            "rule.description": source.get("rule", {}).get("description"),
                            "data.srcip": source.get("data", {}).get("srcip"),
                            "data.dstip": source.get("data", {}).get("dstip"),
                            "full_log": source.get("full_log")
                        }
                        alerts.append(alert)
                    
                    logger.info(f"成功獲取 {len(alerts)} 條告警資料")
                    return alerts
                    
                else:
                    logger.error(f"Wazuh API請求失敗: {response.status}")
                    return []
                    
        except asyncio.TimeoutError:
            logger.error("Wazuh API請求超時")
            return []
        except Exception as e:
            logger.error(f"Wazuh API請求異常: {e}")
            return []
    
    async def save_alerts_to_file(self, alerts: List[Dict[str, Any]], file_path: Path):
        """將告警資料保存到檔案"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 以JSON Lines格式保存
            with open(file_path, 'a', encoding='utf-8') as f:
                for alert in alerts:
                    f.write(json.dumps(alert, ensure_ascii=False) + '\n')
            
            logger.info(f"告警資料已保存至: {file_path}")
            
        except Exception as e:
            logger.error(f"保存告警資料失敗: {e}")
    
    async def fetch_and_save_recent_data(self) -> Path:
        """獲取並保存最近的告警資料"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = settings.PROCESSED_LOGS_DIR / f"wazuh_alerts_{timestamp}.json"
        
        alerts = await self.fetch_recent_alerts(minutes=5)
        if alerts:
            await self.save_alerts_to_file(alerts, file_path)
            return file_path
        else:
            logger.warning("未獲取到任何告警資料")
            return None

async def test_wazuh_connection():
    """測試Wazuh連接"""
    try:
        async with WazuhClient() as client:
            alerts = await client.fetch_recent_alerts(minutes=1)
            logger.info(f"Wazuh連接測試成功，獲取到 {len(alerts)} 條資料")
            return True
    except Exception as e:
        logger.error(f"Wazuh連接測試失敗: {e}")
        return False

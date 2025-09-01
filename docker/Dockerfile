FROM python:3.9-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    bash \
    cron \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# 安裝Python依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式碼
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY data/reference/ ./data/reference/

# 複製預訓練模型（如果存在）
# 注意：您需要將預訓練模型放在 pretrained_model/ 目錄中
COPY pretrained_model/ ./data/models/current/ 2>/dev/null || true

# 創建必要目錄
RUN mkdir -p /app/data/models/current \
    && mkdir -p /app/data/models/backups \
    && mkdir -p /app/data/logs/low_confidence \
    && mkdir -p /app/data/logs/processed \
    && mkdir -p /app/logs

# 設定腳本執行權限
RUN chmod +x /app/scripts/fetch_wazuh_data.sh

# 創建非root用戶
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 8000

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 啟動命令
CMD ["python", "-m", "src.main"]

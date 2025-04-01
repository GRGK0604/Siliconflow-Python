# 构建阶段
FROM python:3.9-slim as builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# 安装依赖到虚拟环境
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 最终阶段 - 使用更小的slim镜像
FROM python:3.9-slim

WORKDIR /app

# 从构建阶段复制虚拟环境
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# 创建数据目录并设置权限
RUN mkdir -p /app/data && chmod -R 777 /app/data

# 清理不必要的包和缓存
RUN find /venv -name "__pycache__" -type d -exec rm -rf {} +

# 仅复制必要的项目文件
COPY main.py config.py models.py utils.py db.py ./
COPY static/ ./static/
COPY doc/ ./doc/

# 使用非root用户运行
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 7898

# 启动应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7898"] 

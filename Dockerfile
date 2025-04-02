FROM python:3.9-alpine

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 安装构建依赖
RUN apk add --no-cache --virtual .build-deps gcc musl-dev

# 复制和安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    apk del .build-deps && \
    find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    rm -rf /root/.cache/pip/*

# 复制项目文件
COPY . .

# 创建数据目录并设置权限
RUN mkdir -p /app/data && chmod 777 /app/data

# 暴露应用端口
EXPOSE 7898

# 启动应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7898"]

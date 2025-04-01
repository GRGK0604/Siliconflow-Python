FROM python:3.9-alpine

WORKDIR /app

# 创建数据目录
RUN mkdir -p /app/data && chmod -R 777 /app/data

# 安装必要的系统依赖
RUN apk add --no-cache --virtual .build-deps \
    gcc \
    musl-dev \
    python3-dev

# 复制依赖文件并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apk del .build-deps

# 仅复制必要的项目文件
COPY main.py config.py models.py utils.py db.py ./
COPY static/ ./static/
COPY doc/ ./doc/

# 暴露正确的端口
EXPOSE 7898

# 启动应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7898"] 

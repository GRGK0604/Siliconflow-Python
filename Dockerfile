FROM python:3.9-alpine

WORKDIR /app

# 安装必要的系统依赖
RUN apk add --no-cache --virtual .build-deps \
    gcc \
    musl-dev \
    python3-dev \
    && mkdir -p /app/data

# 复制依赖文件并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apk del .build-deps

# 复制项目文件
COPY . .

# 权限设置
RUN chmod -R 777 /app/data

# 暴露正确的端口
EXPOSE 7898

# 启动应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7898"] 

# 构建阶段
FROM python:3.9-alpine AS builder

WORKDIR /build

# 安装构建依赖
RUN apk add --no-cache gcc musl-dev python3-dev

# 复制依赖文件
COPY requirements.txt .

# 安装依赖到指定目录
RUN pip install --no-cache-dir --target=/install -r requirements.txt

# 运行阶段
FROM python:3.9-alpine

WORKDIR /app

# 从构建阶段复制安装好的依赖
COPY --from=builder /install /usr/local/lib/python3.9/site-packages

# 创建数据目录
RUN mkdir -p /app/data && chmod -R 777 /app/data

# 仅复制必要的项目文件
COPY main.py config.py models.py utils.py db.py ./
COPY requirements.txt ./
COPY static/ ./static/
COPY doc/ ./doc/

# 暴露正确的端口
EXPOSE 7898

# 启动应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7898"] 
# 构建阶段
FROM python:3.9-alpine AS builder

# 设置工作目录
WORKDIR /app

# 安装构建依赖
RUN apk add --no-cache gcc musl-dev

# 复制并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --no-compile --user -r requirements.txt

# 最终阶段
FROM python:3.9-alpine

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH=/root/.local/bin:$PATH

# 安装运行时必需的依赖
RUN apk add --no-cache libstdc++

# 复制构建阶段的Python包
COPY --from=builder /root/.local /root/.local

# 复制项目文件 - 只复制必需文件
COPY main.py db.py utils.py models.py config.py requirements.txt ./
COPY static ./static/
COPY doc ./doc/

# 创建数据目录并设置权限
RUN mkdir -p /app/data && chmod 777 /app/data

# 删除不必要的文件
RUN find /root/.local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /root/.local -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true && \
    find /root/.local -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true && \
    find /root/.local -name "*.pyc" -delete

# 暴露应用端口
EXPOSE 7898

# 启动应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7898"]

# 使用官方Python基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（按需添加）
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖列表并安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用程序文件
COPY main.py .
COPY config.py .
COPY static/ ./static/
COPY admin.png keylists.png login.png ./

# 创建持久化数据目录并设置权限
RUN mkdir -p /root/siliconflow && chmod 777 /root/siliconflow

# 添加非root用户（安全增强）
RUN useradd -m appuser && chown -R appuser /app /root/siliconflow
USER appuser

# 暴露应用端口
EXPOSE 7898

# 设置容器启动命令
CMD ["python", "main.py"]

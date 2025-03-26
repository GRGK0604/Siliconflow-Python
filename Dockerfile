FROM python:3.10-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建日志目录和备份目录
RUN mkdir -p logs log_backups

# 暴露端口
EXPOSE 7898

# 启动应用
CMD ["python", "main.py"] 
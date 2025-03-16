# Dockerfile
FROM python:3.8-slim

WORKDIR /app

# 复制所有必要文件
COPY main.py config.py requirements.txt ./
COPY static ./static
COPY *.html ./

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口（根据你的应用实际端口修改）
EXPOSE 7898

CMD ["python", "main.py"]

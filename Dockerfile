FROM python:3.9-slim
 
 # 设置工作目录
 WORKDIR /app
 
 # 设置环境变量，确保Python输出直接显示到终端
 ENV PYTHONDONTWRITEBYTECODE=1
 ENV PYTHONUNBUFFERED=1
 
 # 安装项目依赖
 COPY requirements.txt /app/
 RUN pip install --no-cache-dir -r requirements.txt
 
 # 复制项目文件
 COPY . /app/
 
 # 创建数据目录并设置权限
 RUN mkdir -p /app/data && chmod 777 /app/data
 
 # 暴露应用端口
 EXPOSE 7898
 
 # 启动应用
 CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7898"]

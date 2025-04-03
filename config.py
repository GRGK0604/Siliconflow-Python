# if you want to add api key check, set this to the api key:
import os

# 从环境变量读取API_KEY，如果没有则设为None
API_KEY = os.environ.get("API_KEY", None)

# Admin credentials for login
# 从环境变量读取管理员账号密码，如果没有则使用默认值
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "password")  # You should change this to a secure password

# 自动刷新密钥设置
# 自动刷新时间间隔（秒），默认3600秒（1小时）
# 设置为0则禁用自动刷新
AUTO_REFRESH_INTERVAL = int(os.environ.get("AUTO_REFRESH_INTERVAL", "3600"))

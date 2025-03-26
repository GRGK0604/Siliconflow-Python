"""
数据库配置文件 - 支持MySQL和SQLite
"""

import os

# 数据库类型: 'mysql' 或 'sqlite'
DB_TYPE = 'mysql'

# 从环境变量获取MySQL配置
MYSQL_CONFIG = {
    'host': os.environ.get('MYSQL_HOST', 'localhost'),
    'port': int(os.environ.get('MYSQL_PORT', 3306)),
    'user': os.environ.get('MYSQL_USER', 'root'),
    'password': os.environ.get('MYSQL_PASSWORD', 'your_password'),
    'database': os.environ.get('MYSQL_DATABASE', 'siliconfig'),
    'charset': 'utf8mb4',
    'pool_size': 10,  # 连接池大小
    'max_overflow': 20,  # 连接池最大溢出数
    'pool_timeout': 30,  # 连接池获取连接超时时间(秒)
    'pool_recycle': 3600,  # 连接回收时间(秒)
}

# SQLite配置（如需备用）
SQLITE_DB_PATH = "db/siliconfig.db"

# 日志清理配置
LOG_AUTO_CLEAN = True  # 是否启用自动清理日志
LOG_RETENTION_DAYS = 30  # 保留最近30天的日志
LOG_CLEAN_INTERVAL_HOURS = 24  # 每24小时清理一次
LOG_BACKUP_ENABLED = False  # 是否在清理前备份日志
LOG_BACKUP_DIR = "log_backups"  # 日志备份目录 
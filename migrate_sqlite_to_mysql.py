#!/usr/bin/env python3
"""
SQLite到MySQL数据迁移工具

此脚本将SQLite数据库中的数据迁移到MySQL数据库
使用前请确保:
1. MySQL服务已启动
2. MySQL中已创建相应的数据库 (默认为siliconfig)
3. MySQL用户有足够的权限
4. 所需的Python库已安装 (pymysql, sqlalchemy)

使用方法:
python migrate_sqlite_to_mysql.py

设置数据库连接信息，请修改db_config.py文件中的配置
"""

import os
import sys
import time
import sqlite3
import logging
from typing import List, Dict, Any, Optional

import pymysql
from sqlalchemy import create_engine, text

# 导入配置和数据库管理器
import db_config
from db_manager import Base

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("migration")

def create_mysql_db_if_not_exists():
    """确保MySQL数据库存在"""
    config = db_config.MYSQL_CONFIG
    
    # 创建没有指定数据库的连接
    connection_url = (f"mysql+pymysql://{config['user']}:{config['password']}@"
                     f"{config['host']}:{config['port']}/?charset={config['charset']}")
    engine = create_engine(connection_url)
    
    with engine.connect() as conn:
        # 检查数据库是否存在
        result = conn.execute(text(f"SHOW DATABASES LIKE '{config['database']}'"))
        if result.rowcount == 0:
            logger.info(f"正在创建MySQL数据库: {config['database']}")
            conn.execute(text(f"CREATE DATABASE `{config['database']}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
            logger.info(f"MySQL数据库 {config['database']} 创建成功")
        else:
            logger.info(f"MySQL数据库 {config['database']} 已存在")

def setup_mysql_tables():
    """在MySQL中创建表结构"""
    config = db_config.MYSQL_CONFIG
    connection_url = (f"mysql+pymysql://{config['user']}:{config['password']}@"
                     f"{config['host']}:{config['port']}/{config['database']}?"
                     f"charset={config['charset']}")
    engine = create_engine(connection_url)
    
    # 创建表
    logger.info("正在创建MySQL表结构...")
    Base.metadata.create_all(engine)
    logger.info("MySQL表结构创建成功")
    
    return engine

def get_sqlite_connection():
    """获取SQLite数据库连接"""
    sqlite_path = db_config.SQLITE_DB_PATH
    
    # 检查SQLite数据库文件是否存在
    if not os.path.exists(sqlite_path):
        logger.error(f"SQLite数据库文件不存在: {sqlite_path}")
        sys.exit(1)
    
    logger.info(f"连接SQLite数据库: {sqlite_path}")
    return sqlite3.connect(sqlite_path)

def migrate_table(sqlite_conn, mysql_engine, table_name, columns):
    """迁移单个表的数据"""
    logger.info(f"开始迁移表: {table_name}")
    cursor = sqlite_conn.cursor()
    
    # 查询SQLite表中的所有数据
    column_list = ", ".join(columns)
    cursor.execute(f"SELECT {column_list} FROM {table_name}")
    rows = cursor.fetchall()
    
    if not rows:
        logger.info(f"表 {table_name} 中没有数据，跳过")
        return 0
    
    # 构建插入语句
    placeholders = ", ".join(["%s"] * len(columns))
    insert_sql = f"INSERT INTO {table_name} ({column_list}) VALUES ({placeholders})"
    
    # 使用MySQL连接批量插入
    conn = mysql_engine.raw_connection()
    try:
        with conn.cursor() as cursor:
            # 一次插入100条记录
            batch_size = 100
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i+batch_size]
                cursor.executemany(insert_sql, batch)
                conn.commit()
                logger.info(f"已迁移 {table_name} 表数据: {i+len(batch)}/{len(rows)}")
        
        logger.info(f"表 {table_name} 数据迁移完成，共 {len(rows)} 条记录")
        return len(rows)
    except Exception as e:
        logger.error(f"表 {table_name} 数据迁移失败: {str(e)}")
        conn.rollback()
        return 0
    finally:
        conn.close()

def migrate_data():
    """主迁移函数"""
    start_time = time.time()
    
    # 创建数据库和表结构
    create_mysql_db_if_not_exists()
    mysql_engine = setup_mysql_tables()
    
    # 连接SQLite数据库
    sqlite_conn = get_sqlite_connection()
    
    # 确定要迁移的表和字段
    tables = {
        "api_keys": ["key", "add_time", "balance", "usage_count", "enabled"],
        "logs": ["id", "used_key", "model", "call_time", "input_tokens", "output_tokens", "total_tokens"],
        "sessions": ["session_id", "username", "created_at"]
    }
    
    # 依次迁移每张表
    total_records = 0
    for table_name, columns in tables.items():
        count = migrate_table(sqlite_conn, mysql_engine, table_name, columns)
        total_records += count
    
    # 关闭连接
    sqlite_conn.close()
    
    # 完成
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"数据迁移完成，共迁移 {total_records} 条记录，耗时 {duration:.2f} 秒")

if __name__ == "__main__":
    logger.info("开始进行SQLite到MySQL的数据迁移...")
    
    try:
        migrate_data()
        logger.info("迁移成功！请修改db_config.py中的DB_TYPE为'mysql'来使用MySQL数据库")
    except Exception as e:
        logger.error(f"迁移过程发生错误: {str(e)}")
        sys.exit(1) 
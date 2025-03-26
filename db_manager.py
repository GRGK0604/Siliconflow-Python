"""
数据库管理模块 - 提供MySQL连接管理和ORM模型
"""

import time
import logging
from contextlib import contextmanager
from typing import Dict, Any, Generator, Optional

import pymysql
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from fastapi import HTTPException

# 导入数据库配置
from db_config import DB_TYPE, MYSQL_CONFIG

# 设置日志
logger = logging.getLogger("db_manager")

# 创建ORM基类
Base = declarative_base()

# 创建模型类
class ApiKey(Base):
    __tablename__ = "api_keys"
    
    key = Column(String(255), primary_key=True)
    add_time = Column(Float, nullable=False)
    balance = Column(Float, default=0.0)
    usage_count = Column(Integer, default=0)
    enabled = Column(Boolean, default=True)

class Log(Base):
    __tablename__ = "logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    used_key = Column(String(255), nullable=False)
    model = Column(String(50))
    call_time = Column(Float, nullable=False)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    
    # 添加索引以提高查询性能
    __table_args__ = (
        Index('idx_logs_call_time', 'call_time'),
    )

class Session(Base):
    __tablename__ = "sessions"
    
    session_id = Column(String(255), primary_key=True)
    username = Column(String(50), nullable=False)
    created_at = Column(Float, nullable=False)
    
    # 添加索引以提高查询性能
    __table_args__ = (
        Index('idx_sessions_created_at', 'created_at'),
    )

# 创建数据库引擎和会话工厂
def create_engine_and_session():
    """创建数据库引擎和会话工厂"""
    config = MYSQL_CONFIG
    
    # 构建连接URL
    connection_url = (
        f"mysql+pymysql://{config['user']}:{config['password']}@"
        f"{config['host']}:{config['port']}/{config['database']}?"
        f"charset={config['charset']}"
    )
    
    # 创建引擎，配置连接池
    engine = create_engine(
        connection_url,
        pool_size=config.get('pool_size', 10),
        max_overflow=config.get('max_overflow', 20),
        pool_timeout=config.get('pool_timeout', 30),
        pool_recycle=config.get('pool_recycle', 3600),
    )
    
    # 创建会话工厂
    SessionFactory = sessionmaker(bind=engine)
    
    return engine, SessionFactory

# 创建引擎和会话工厂
engine, SessionFactory = create_engine_and_session()

# 初始化数据库表
def init_db():
    """初始化数据库表结构"""
    Base.metadata.create_all(engine)
    logger.info("数据库表结构初始化完成")

# 数据库会话上下文管理器
@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """获取数据库会话的上下文管理器"""
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        if isinstance(e, pymysql.err.OperationalError):
            logger.error(f"数据库操作错误: {str(e)}")
            raise HTTPException(status_code=503, detail="数据库繁忙，请稍后重试")
        else:
            logger.error(f"数据库会话错误: {str(e)}")
            raise
    finally:
        session.close()

# 在模块加载时初始化数据库
init_db() 
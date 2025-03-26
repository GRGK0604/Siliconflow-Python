"""
数据库管理模块 - 提供MySQL连接管理和ORM模型
"""

import time
import logging
from contextlib import contextmanager
from typing import Dict, Any, Generator, Optional

# 替换pymysql为aiomysql用于异步操作
import aiomysql
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text, ForeignKey, Index, desc, asc, func, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError
# 导入异步SQLAlchemy支持
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from fastapi import HTTPException
import os

# 导入数据库配置
from db_config import DB_TYPE, MYSQL_CONFIG, SQLITE_DB_PATH

# 设置日志
logger = logging.getLogger("db_manager")

# 创建ORM基类
Base = declarative_base()

# 创建模型类
class ApiKey(Base):
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True)
    key = Column(String(255), unique=True, nullable=False)
    add_time = Column(Float, default=time.time)
    balance = Column(Float, default=0)
    usage_count = Column(Integer, default=0)
    enabled = Column(Boolean, default=True)
    
    # 与日志的关系
    logs = relationship("Log", back_populates="api_key")

class Log(Base):
    __tablename__ = "logs"
    
    id = Column(Integer, primary_key=True)
    used_key = Column(String(255), ForeignKey('api_keys.key'))
    model = Column(String(50))
    call_time = Column(Float, default=time.time)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    
    # 与API密钥的关系
    api_key = relationship("ApiKey", back_populates="logs")

    # 添加索引以提高查询性能
    __table_args__ = (
        Index('idx_logs_call_time', 'call_time'),
    )

class Session(Base):
    __tablename__ = "sessions"
    
    session_id = Column(String(32), primary_key=True)
    username = Column(String(50))
    created_at = Column(Float, default=time.time)
    
    # 添加索引以提高查询性能
    __table_args__ = (
        Index('idx_sessions_created_at', 'created_at'),
    )

# 根据配置创建数据库引擎
if DB_TYPE == 'mysql':
    # MySQL连接字符串 - 同步版本
    connection_string = (
        f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@"
        f"{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}?"
        f"charset={MYSQL_CONFIG['charset']}"
    )
    
    # 异步连接字符串
    async_connection_string = (
        f"mysql+aiomysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@"
        f"{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}?"
        f"charset={MYSQL_CONFIG['charset']}"
    )
    
    # 创建MySQL引擎 - 同步版本（仅用于初始化表结构和迁移）
    engine = create_engine(
        connection_string,
        pool_size=MYSQL_CONFIG.get('pool_size', 10),
        max_overflow=MYSQL_CONFIG.get('max_overflow', 20),
        pool_timeout=MYSQL_CONFIG.get('pool_timeout', 30),
        pool_recycle=MYSQL_CONFIG.get('pool_recycle', 3600),
        echo=False
    )
    
    # 创建异步引擎（用于实际操作）
    async_engine = create_async_engine(
        async_connection_string,
        pool_size=MYSQL_CONFIG.get('pool_size', 10),
        max_overflow=MYSQL_CONFIG.get('max_overflow', 20),
        pool_timeout=MYSQL_CONFIG.get('pool_timeout', 30),
        pool_recycle=MYSQL_CONFIG.get('pool_recycle', 3600),
        echo=False
    )
else:
    # 确保SQLite数据库目录存在
    os.makedirs(os.path.dirname(os.path.abspath(SQLITE_DB_PATH)), exist_ok=True)
    
    # SQLite连接字符串 - 同步版本
    connection_string = f"sqlite:///{SQLITE_DB_PATH}"
    
    # SQLite异步连接字符串
    async_connection_string = f"sqlite+aiosqlite:///{SQLITE_DB_PATH}"
    
    # 创建SQLite引擎 - 同步版本（仅用于初始化表结构和迁移）
    engine = create_engine(
        connection_string,
        echo=False,
        connect_args={"check_same_thread": False}
    )
    
    # 创建异步引擎（用于实际操作）
    async_engine = create_async_engine(
        async_connection_string,
        echo=False,
        connect_args={"check_same_thread": False} 
    )

# 创建所有表 - 使用同步引擎
Base.metadata.create_all(engine)

# 创建会话工厂 - 同步版本（仅用于迁移和健康检查）
Session_factory = sessionmaker(bind=engine)

# 创建异步会话工厂
Async_Session_factory = async_sessionmaker(bind=async_engine)

# 上下文管理器，用于处理数据库会话 - 同步版本
@contextmanager
def get_db_session():
    """提供数据库会话的上下文管理器（同步版本，避免在异步代码中使用）"""
    session = Session_factory()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

# 上下文管理器，用于处理数据库会话 - 异步版本
async def get_async_db_session():
    """提供数据库会话的异步上下文管理器"""
    session = None
    try:
        session = Async_Session_factory()
        yield session
        await session.commit()
    except Exception as e:
        if session:
            await session.rollback()
        raise e
    finally:
        if session:
            await session.close()

# 初始化数据库表
def init_db():
    """初始化数据库表结构"""
    Base.metadata.create_all(engine)
    logger.info("数据库表结构初始化完成")

def check_db_health():
    """检查数据库连接健康状态 - 同步版本，仅用于健康检查"""
    health_data = {
        "status": "healthy",
        "type": DB_TYPE,
        "details": {}
    }
    
    try:
        # 尝试执行简单查询
        with get_db_session() as session:
            if DB_TYPE == 'mysql':
                # MySQL特定信息
                result = session.execute("SELECT VERSION()").scalar()
                health_data["details"]["version"] = result
                
                # 检查连接池状态
                pool_status = {
                    "size": engine.pool.size(),
                    "checkedin": engine.pool.checkedin(),
                    "checkedout": engine.pool.checkedout(),
                    "overflow": engine.pool.overflow()
                }
                health_data["details"]["pool"] = pool_status
            else:
                # SQLite特定信息
                result = session.execute("SELECT sqlite_version()").scalar()
                health_data["details"]["version"] = result
                health_data["details"]["path"] = SQLITE_DB_PATH
            
            # 检查表计数
            tables_count = {}
            api_key_count = session.execute(select(func.count()).select_from(ApiKey)).scalar()
            log_count = session.execute(select(func.count()).select_from(Log)).scalar()
            session_count = session.execute(select(func.count()).select_from(Session)).scalar()
            
            tables_count["api_keys"] = api_key_count
            tables_count["logs"] = log_count
            tables_count["sessions"] = session_count
            
            health_data["details"]["tables_count"] = tables_count
            
    except OperationalError as e:
        health_data["status"] = "unhealthy"
        health_data["error"] = f"Database connection error: {str(e)}"
    except SQLAlchemyError as e:
        health_data["status"] = "degraded"
        health_data["error"] = f"Database query error: {str(e)}"
    except Exception as e:
        health_data["status"] = "unknown"
        health_data["error"] = f"Unexpected error: {str(e)}"
    
    return health_data 
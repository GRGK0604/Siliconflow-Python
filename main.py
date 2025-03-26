from fastapi import FastAPI, Request, HTTPException, Depends, Response, Cookie
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from config import API_KEY, ADMIN_USERNAME, ADMIN_PASSWORD
import random
import time
import asyncio
import aiohttp
import json
import secrets
from contextlib import asynccontextmanager, contextmanager
from uvicorn.config import LOGGING_CONFIG
from datetime import datetime, timedelta
import os
import logging
import logging.handlers
from sqlalchemy import select, func, desc, asc, text
from sqlalchemy.exc import SQLAlchemyError
import sqlite3
from decimal import Decimal
from io import BytesIO
import uuid

# 自定义JSON编码器，以处理Decimal类型
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

# 用于将对象转换为JSON的辅助函数
def jsonable_encoder(obj):
    """将对象转换为JSON可序列化的对象"""
    return json.loads(json.dumps(obj, cls=CustomJSONEncoder))

# 导入数据库配置和管理器
from db_config import (
    DB_TYPE, 
    LOG_AUTO_CLEAN, 
    LOG_RETENTION_DAYS,
    LOG_CLEAN_INTERVAL_HOURS,
    LOG_BACKUP_ENABLED,
    LOG_BACKUP_DIR,
    SQLITE_DB_PATH
)
from db_manager import get_db_session, get_async_db_session, ApiKey, Log, Session as DbSession, check_db_health, close_async_engine

# 设置日志
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# 创建文件处理器
os.makedirs("logs", exist_ok=True)
file_handler = logging.handlers.RotatingFileHandler(
    "logs/app.log", 
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# 添加处理器到logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 配置日志格式
LOGGING_CONFIG["formatters"]["default"]["fmt"] = (
    "%(asctime)s - %(levelprefix)s %(message)s"
)

# 自动清理日志的函数
async def auto_clean_logs():
    """定时清理过期的日志记录"""
    logger.info(f"日志自动清理任务已启动，将每 {LOG_CLEAN_INTERVAL_HOURS} 小时运行一次")
    
    while True:
        try:
            if LOG_AUTO_CLEAN:
                # 计算保留日志的时间戳（当前时间减去保留天数）
                retention_timestamp = time.time() - (LOG_RETENTION_DAYS * 86400)  # 86400秒 = 1天
                
                # 如果启用了备份，先备份日志
                if LOG_BACKUP_ENABLED:
                    await backup_logs(retention_timestamp)
                
                # 删除旧日志
                try:
                    async for session in get_async_db_session():
                        try:
                            # 先计算要删除的记录数
                            count_query = select(func.count()).select_from(Log).where(Log.call_time < retention_timestamp)
                            result = await session.execute(count_query)
                            count_to_delete = result.scalar() or 0
                            
                            if count_to_delete > 0:
                                # 删除旧日志
                                delete_query = Log.__table__.delete().where(Log.call_time < retention_timestamp)
                                result = await session.execute(delete_query)
                                deleted_count = result.rowcount
                                
                                # 记录清理操作
                                logger.info(f"自动清理日志：已删除 {deleted_count} 条过期日志记录")
                        except Exception as db_error:
                            logger.error(f"数据库操作出错: {str(db_error)}")
                except Exception as session_error:
                    logger.error(f"获取数据库会话出错: {str(session_error)}")
            
            # 等待下一次执行
            next_run = datetime.now() + timedelta(hours=LOG_CLEAN_INTERVAL_HOURS)
            logger.info(f"下一次日志清理将在 {next_run.strftime('%Y-%m-%d %H:%M:%S')} 进行")
            
            # 添加检查任务是否应该停止的功能
            try:
                await asyncio.sleep(LOG_CLEAN_INTERVAL_HOURS * 3600)  # 转换为秒
            except asyncio.CancelledError:
                logger.info("日志清理任务被取消")
                break
        
        except Exception as e:
            logger.error(f"日志自动清理出错: {str(e)}")
            
            # 同样添加检查是否应该停止的功能
            try:
                await asyncio.sleep(3600)  # 发生错误时等待1小时后重试
            except asyncio.CancelledError:
                logger.info("日志清理任务被取消")
                break

# 备份日志到文件
async def backup_logs(cutoff_timestamp):
    """将要删除的日志备份到文件中"""
    if not LOG_BACKUP_ENABLED:
        return
    
    try:
        # 确保备份目录存在
        os.makedirs(LOG_BACKUP_DIR, exist_ok=True)
        
        backup_filename = os.path.join(
            LOG_BACKUP_DIR, 
            f"logs_before_{datetime.fromtimestamp(cutoff_timestamp).strftime('%Y%m%d')}.csv"
        )
        
        backup_count = 0
        
        # 查询需要备份的日志
        try:
            async for session in get_async_db_session():
                try:
                    # 查询旧日志
                    query = select(Log).where(Log.call_time < cutoff_timestamp)
                    result = await session.execute(query)
                    logs = result.scalars().all()
                    
                    # 写入CSV文件
                    if logs:
                        with open(backup_filename, 'w', encoding='utf-8') as f:
                            # 写入CSV头
                            f.write("id,used_key,model,call_time,input_tokens,output_tokens,total_tokens\n")
                            
                            # 写入数据
                            for log in logs:
                                try:
                                    # 安全地写入每条记录，处理可能的空值
                                    log_id = log.id or ""
                                    used_key = log.used_key.replace(",", "_") if log.used_key else ""
                                    model = log.model.replace(",", "_") if log.model else ""
                                    call_time = log.call_time or 0
                                    input_tokens = log.input_tokens or 0
                                    output_tokens = log.output_tokens or 0
                                    total_tokens = log.total_tokens or 0
                                    
                                    f.write(f"{log_id},{used_key},{model},{call_time},{input_tokens},{output_tokens},{total_tokens}\n")
                                    backup_count += 1
                                except Exception as record_error:
                                    logger.error(f"写入日志记录时出错: {str(record_error)}")
                                    continue
                        
                        logger.info(f"已备份 {backup_count} 条日志记录到 {backup_filename}")
                except Exception as session_error:
                    logger.error(f"查询数据库时出错: {str(session_error)}")
        except Exception as db_error:
            logger.error(f"获取数据库会话时出错: {str(db_error)}")
    
    except Exception as e:
        logger.error(f"备份日志失败: {str(e)}", exc_info=True)

# 全局HTTP会话，用于所有外部请求
class HTTPClientSession:
    def __init__(self):
        self._session = None
        self._lock = asyncio.Lock()
    
    async def get_session(self):
        """获取全局共享的HTTP会话，如果不存在则创建新的"""
        if self._session is None:
            async with self._lock:
                if self._session is None:
                    # 配置TCP连接池
                    conn = aiohttp.TCPConnector(
                        limit=100,  # 限制最大连接数
                        limit_per_host=20,  # 每个主机的最大连接数
                        keepalive_timeout=60.0,  # 保持连接的超时时间
                        enable_cleanup_closed=True  # 自动清理已关闭的连接
                    )
                    self._session = aiohttp.ClientSession(
                        connector=conn,
                        timeout=aiohttp.ClientTimeout(total=60)
                    )
                    logger.info("已创建全局HTTP会话")
        return self._session
    
    async def close(self):
        """关闭HTTP会话，释放资源"""
        if self._session is not None:
            await self._session.close()
            self._session = None
            logger.info("已关闭全局HTTP会话")

# 创建全局HTTP会话实例
http_client = HTTPClientSession()

# 修改应用生命周期管理器，确保在应用关闭时关闭HTTP会话
@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用程序的生命周期，初始化和清理资源"""
    app_start_time = time.time()
    app.state.startup_timestamp = app_start_time
    logger.info(f"应用启动，初始化清理任务... 启动时间: {datetime.fromtimestamp(app_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建日志清理任务
    cleanup_task = asyncio.create_task(auto_clean_logs())
    
    # 将任务保存在应用状态中，以便稍后引用
    app.state.cleanup_task = cleanup_task
    
    try:
        # 应用正常运行部分
        yield
    except Exception as e:
        logger.error(f"应用运行期间发生错误: {str(e)}", exc_info=True)
    finally:
        # 应用关闭时执行清理
        app_end_time = time.time()
        run_duration = app_end_time - app_start_time
        logger.info(f"应用关闭，清理资源... 运行时长: {int(run_duration//3600)}小时{int((run_duration%3600)//60)}分{int(run_duration%60)}秒")
        
        # 取消清理任务
        if hasattr(app.state, 'cleanup_task') and not app.state.cleanup_task.done():
            logger.info("正在取消日志清理任务...")
            app.state.cleanup_task.cancel()
            try:
                await asyncio.wait_for(app.state.cleanup_task, timeout=5.0)
                logger.info("日志清理任务已取消")
            except asyncio.TimeoutError:
                logger.warning("日志清理任务取消超时")
            except asyncio.CancelledError:
                logger.info("日志清理任务已取消")
            except Exception as e:
                logger.error(f"取消日志清理任务时出错: {str(e)}")
        
        # 关闭HTTP客户端会话
        try:
            logger.info("正在关闭HTTP客户端会话...")
            await http_client.close()
        except Exception as e:
            logger.error(f"关闭HTTP客户端时出错: {str(e)}")
        
        # 关闭数据库连接
        try:
            logger.info("正在关闭数据库连接...")
            await close_async_engine()
            logger.info("数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库引擎时出错: {str(e)}")
        
        # 仅在SQLite模式下关闭连接
        if DB_TYPE == 'sqlite' and 'conn' in globals():
            try:
                conn.close()
                logger.info("SQLite数据库连接已关闭")
            except Exception as e:
                logger.error(f"关闭SQLite连接时出错: {str(e)}")
        
        logger.info("所有资源已清理完毕，应用已安全关闭")

# 创建FastAPI应用
app = FastAPI(
    title="SiliconFlow API",
    description="SiliconFlow API代理服务",
    version="1.0.0",
    lifespan=lifespan  # 使用lifespan上下文管理器
)

# 自定义异常处理中间件
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    """全局错误处理中间件，捕获并处理所有请求中的异常"""
    # 提取请求的基本信息
    request_method = request.method
    request_path = request.url.path
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    client_host = request.client.host if request.client else "未知IP"
    
    # 尝试从X-Forwarded-For获取真实IP
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        client_host = forwarded_for.split(",")[0].strip()
    
    # 记录请求日志
    logger.info(f"开始处理请求: {request_id} | {request_method} {request_path} | 客户端: {client_host}")
    
    start_time = time.time()
    
    try:
        # 正常处理请求
        response = await call_next(request)
        
        # 记录成功请求日志
        process_time = round((time.time() - start_time) * 1000, 2)
        logger.info(f"请求完成: {request_id} | {request_method} {request_path} | 状态码: {response.status_code} | 处理时间: {process_time}ms")
        
        return response
    
    # 捕获并处理各种类型的异常
    except Exception as e:
        # 计算处理时间
        process_time = round((time.time() - start_time) * 1000, 2)
        
        # 确定HTTP状态码和错误消息
        status_code = 500
        error_message = "服务器内部错误"
        
        # 根据异常类型分类处理
        if isinstance(e, HTTPException):
            status_code = e.status_code
            error_message = str(e.detail)
        elif isinstance(e, RequestValidationError):
            status_code = 422
            error_message = "请求验证失败"
        elif isinstance(e, TimeoutError) or isinstance(e, asyncio.TimeoutError):
            status_code = 504
            error_message = "请求处理超时"
        elif isinstance(e, ConnectionError) or isinstance(e, aiohttp.ClientConnectionError):
            status_code = 502
            error_message = "连接服务失败"
        
        # 只在非4xx错误时记录详细的异常信息，避免记录过多常规客户端错误
        if status_code < 400 or status_code >= 500:
            logger.error(
                f"处理请求时出错: {request_id} | {request_method} {request_path} | "
                f"状态码: {status_code} | 错误: {error_message} | 类型: {type(e).__name__} | 处理时间: {process_time}ms", 
                exc_info=True
            )
        else:
            logger.warning(
                f"客户端请求错误: {request_id} | {request_method} {request_path} | "
                f"状态码: {status_code} | 错误: {error_message} | 处理时间: {process_time}ms"
            )
        
        # 创建错误响应
        error_details = {"type": type(e).__name__, "request_id": request_id}
        
        # 仅在开发环境或内部错误时包含堆栈跟踪
        if status_code >= 500:
            import traceback
            error_details["traceback"] = traceback.format_exc().splitlines()[-5:]  # 只包含最后5行
        
        # 返回JSON格式的错误响应
        return JSONResponse(
            status_code=status_code,
            content={
                "error": error_message,
                "details": error_details,
                "timestamp": time.time()
            }
        )

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"))

# 确保备份目录存在（如果启用）
if LOG_BACKUP_ENABLED:
    os.makedirs(LOG_BACKUP_DIR, exist_ok=True)

BASE_URL = "https://api.siliconflow.cn"  # adjust if needed

# 确保数据库目录存在
os.makedirs(os.path.dirname(os.path.abspath(SQLITE_DB_PATH)), exist_ok=True)

# SQLite 数据库连接 - 仅在使用SQLite模式时需要
if DB_TYPE == 'sqlite':
    # SQLite连接参数
    DB_CHECK_SAME_THREAD = False
    DB_TIMEOUT = 30

    conn = sqlite3.connect(
        SQLITE_DB_PATH, 
        check_same_thread=DB_CHECK_SAME_THREAD,
        timeout=DB_TIMEOUT
    )
    conn.execute("PRAGMA journal_mode=WAL")  # 使用WAL模式提高并发性能
    conn.execute("PRAGMA busy_timeout=10000")  # 设置繁忙超时以减少"database is locked"错误

    # 创建一个上下文管理器来处理数据库连接和游标
    @contextmanager
    def get_cursor():
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except sqlite3.OperationalError as e:
            conn.rollback()
            if "database is locked" in str(e):
                print(f"数据库锁定错误: {str(e)}")
                raise HTTPException(status_code=503, detail="数据库繁忙，请稍后重试")
            else:
                print(f"数据库操作错误: {str(e)}")
                raise HTTPException(status_code=500, detail=f"数据库操作错误: {str(e)}")
        except Exception as e:
            conn.rollback()
            print(f"游标操作失败: {str(e)}")
            raise e
        finally:
            cursor.close()

    # 初始化数据库表
    with get_cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            key TEXT PRIMARY KEY,
            add_time REAL,
            balance REAL,
            usage_count INTEGER,
            enabled INTEGER DEFAULT 1
        )
        """)

        # Create logs table for recording completion calls
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            used_key TEXT,
            model TEXT,
            call_time REAL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            total_tokens INTEGER
        )
        """)

        # Create sessions table for storing user sessions
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            username TEXT,
            created_at REAL
        )
        """)

        # 添加索引以提高查询性能
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_call_time ON logs(call_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at)")

# Custom exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 401:
        # 对于 API 请求，返回 JSON 响应
        if request.url.path.startswith("/v1/") or request.headers.get("accept") == "application/json":
            return JSONResponse(
                status_code=401,
                content={"detail": "未授权访问，请先登录"}
            )
        # 对于网页请求，显示 401 页面
        return FileResponse("static/401.html", status_code=401)
    elif exc.status_code == 404:
        return FileResponse("static/404.html", status_code=404)
    elif exc.status_code == 500:
        return FileResponse("static/500.html", status_code=500)
    
    # 对于其他错误，返回 JSON 或通用错误页面
    if request.url.path.startswith("/v1/") or request.headers.get("accept") == "application/json":
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": str(exc.detail)}
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc.detail)}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)}
    )


# Authentication functions
async def create_session(username: str) -> str:
    """创建新的会话并保存到数据库"""
    session_id = secrets.token_hex(16)
    
    try:
        async for db_session in get_async_db_session():
            # 创建会话对象
            db_session_obj = DbSession(
                session_id=session_id,
                username=username,
                created_at=time.time()
            )
            # 添加到会话
            db_session.add(db_session_obj)
            # 提交事务
            await db_session.commit()
            
            logger.info(f"已创建会话: {session_id} 用户: {username}")
            return session_id
    except Exception as e:
        logger.error(f"创建会话时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"创建会话失败: {str(e)}")
    
    # 如果执行到这里，说明创建会话失败
    logger.error("创建会话失败，未知原因")
    raise HTTPException(status_code=500, detail="创建会话失败")


# 检查会话是否有效
# 设置会话清理周期(秒)
SESSION_CLEANUP_INTERVAL = 3600  # 每小时清理一次过期会话

# 上次清理会话的时间
last_session_cleanup_time = time.time()

async def cleanup_expired_sessions():
    """清理过期的会话"""
    global last_session_cleanup_time
    current_time = time.time()
    
    # 只有在距离上次清理超过指定间隔时才执行清理
    if current_time - last_session_cleanup_time < SESSION_CLEANUP_INTERVAL:
        return
    
    try:
        logger.info("正在清理过期会话...")
        async for session in get_async_db_session():
            # 删除24小时前的过期会话
            delete_query = DbSession.__table__.delete().where(DbSession.created_at < current_time - 86400)
            result = await session.execute(delete_query)
            deleted_count = result.rowcount
            if deleted_count > 0:
                logger.info(f"已清理 {deleted_count} 个过期会话")
        
        # 更新上次清理时间
        last_session_cleanup_time = current_time
    except Exception as e:
        logger.error(f"清理过期会话时出错: {str(e)}")


async def validate_session(session_id: str = Cookie(None)) -> bool:
    """验证会话是否有效"""
    if not session_id:
        logger.debug("会话验证失败: 没有提供session_id")
        return False
    
    try:
        # 定期清理过期会话
        await cleanup_expired_sessions()
        
        # 验证当前会话
        async for db_session in get_async_db_session():
            # 首先查询会话是否存在
            query = select(DbSession).where(DbSession.session_id == session_id)
            result = await db_session.execute(query)
            session_obj = result.scalar_one_or_none()
            
            if not session_obj:
                logger.debug(f"会话验证失败: 找不到会话ID {session_id}")
                return False
            
            # 会话存在，更新会话时间戳，延长会话有效期
            current_time = time.time()
            update_query = DbSession.__table__.update().where(
                DbSession.session_id == session_id
            ).values(created_at=current_time)
            
            await db_session.execute(update_query)
            await db_session.commit()
            logger.debug(f"会话验证成功: {session_id}")
            return True
    
    except Exception as e:
        logger.error(f"验证会话时出错: {str(e)}", exc_info=True)
        return False
    
    # 如果代码执行到这里，说明出现了未知问题
    logger.error(f"会话验证失败: 未知错误 {session_id}")
    return False


async def require_auth(session_id: str = Cookie(None)):
    """验证用户是否已登录，如果未登录则抛出401异常"""
    if not session_id:
        raise HTTPException(status_code=401, detail="未授权访问，请先登录")
    
    valid_session = await validate_session(session_id)
    if not valid_session:
        raise HTTPException(status_code=401, detail="会话已过期，请重新登录")
    
    return True


# 全局API密钥验证缓存
api_key_cache = {}
API_KEY_CACHE_TTL = 3600  # 缓存有效期1小时(秒)

async def validate_key_async(api_key: str, retries=2, timeout=10):
    """验证API密钥是否有效，使用硅基流动API的/v1/user/info接口进行验证
    
    Args:
        api_key: API密钥字符串
        retries: 重试次数，默认2次
        timeout: 每次请求超时时间(秒)，默认10秒
    """
    logger.debug(f"正在验证API密钥: {mask_key(api_key)}")
    
    # 检查缓存
    cache_key = f"key_{api_key}"
    now = time.time()
    if cache_key in api_key_cache:
        cache_time, result = api_key_cache[cache_key]
        # 如果缓存未过期，直接返回结果
        if now - cache_time < API_KEY_CACHE_TTL:
            logger.debug(f"使用缓存的API密钥验证结果: {mask_key(api_key)}")
            return result
    
    # 构造请求头
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # 实现重试机制
    for attempt in range(retries + 1):
        try:
            # 获取共享会话
            session = await http_client.get_session()
            
            # 使用共享会话发送验证请求 - 使用user/info接口
            async with session.get(
                "https://api.siliconflow.cn/v1/user/info", 
                headers=headers, 
                timeout=timeout
            ) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                        # 从响应中获取余额
                        balance = data.get("data", {}).get("totalBalance", 0)
                        logger.debug(f"API密钥验证成功: {mask_key(api_key)}, 余额: {balance}")
                        
                        # 构造成功结果
                        result = {
                            "valid": True,
                            "status_code": response.status,
                            "models": {
                                "models": [
                                    {"name": "GLM-4"},
                                    {"name": "GLM-3-Turbo"},
                                    {"name": "Qwen-7B"}
                                ]
                            },
                            "balance": float(balance)
                        }
                        
                        # 缓存结果
                        api_key_cache[cache_key] = (now, result)
                        return result
                    except json.JSONDecodeError:
                        logger.warning(f"API响应不是有效的JSON: {mask_key(api_key)}")
                        result = {
                            "valid": False,
                            "status_code": response.status,
                            "error": "API响应不是有效的JSON",
                            "balance": 0.0
                        }
                        return result
                else:
                    try:
                        data = await response.json()
                        error_message = data.get("message", "验证失败")
                        logger.warning(f"API密钥验证失败: {mask_key(api_key)}, 状态码: {response.status}, 错误: {error_message}")
                        
                        # 可重试的错误
                        if response.status in [429, 500, 502, 503, 504] and attempt < retries:
                            retry_delay = 0.5 * (2 ** attempt)  # 指数退避策略
                            logger.info(f"将在{retry_delay:.1f}秒后重试验证API密钥 (尝试 {attempt+2}/{retries+1})")
                            await asyncio.sleep(retry_delay)
                            continue
                        
                        # 其他错误
                        result = {
                            "valid": False,
                            "status_code": response.status,
                            "error": error_message,
                            "balance": 0.0
                        }
                        
                        # 缓存结果
                        api_key_cache[cache_key] = (now, result)
                        return result
                    except Exception as text_error:
                        logger.error(f"读取错误响应时出错: {str(text_error)}")
                        result = {
                            "valid": False,
                            "status_code": response.status,
                            "error": f"无法解析错误响应: {str(text_error)}",
                            "balance": 0.0
                        }
                        return result
        
        except asyncio.TimeoutError:
            logger.warning(f"验证API密钥超时: {mask_key(api_key)} (尝试 {attempt+1}/{retries+1})")
            if attempt < retries:
                retry_delay = 0.5 * (2 ** attempt)
                logger.info(f"将在{retry_delay:.1f}秒后重试验证API密钥")
                await asyncio.sleep(retry_delay)
                continue
            
            # 如果所有重试都超时
            result = {
                "valid": False,
                "status_code": 408,
                "error": "请求超时，无法验证API密钥",
                "balance": 0.0
            }
            return result
        
        except Exception as e:
            logger.error(f"验证API密钥时发生错误: {str(e)}", exc_info=True)
            if attempt < retries:
                retry_delay = 0.5 * (2 ** attempt)
                logger.info(f"将在{retry_delay:.1f}秒后重试验证API密钥")
                await asyncio.sleep(retry_delay)
                continue
            
            # 如果所有重试都失败
            result = {
                "valid": False,
                "status_code": 500,
                "error": f"验证过程中出错: {str(e)}",
                "balance": 0.0
            }
            return result
    
    # 这一行通常不会执行到，因为重试循环应该已经返回结果
    # 但为了健壮性和代码完整性，我们仍然提供一个默认返回值
    return {
        "valid": False,
        "status_code": 500,
        "error": "未知错误，验证API密钥失败",
        "balance": 0.0
    }


def insert_api_key(api_key: str, balance: float):
    with get_db_session() as session:
        new_key = ApiKey(
            key=api_key,
            add_time=time.time(),
            balance=balance,
            usage_count=0
        )
        session.add(new_key)


async def log_completion(
    used_key: str,
    model: str,
    call_time: float,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
):
    """记录API调用的token使用情况"""
    try:
        async for session in get_async_db_session():
            log_entry = Log(
                used_key=used_key,
                model=model,
                call_time=call_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens
            )
            session.add(log_entry)
            await session.commit()
    except Exception as e:
        logger.error(f"记录日志时出错: {str(e)}")
        # 出错时继续运行，不影响主流程

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """处理聊天完成请求，转发到上游API"""
    start_time = time.time()
    
    # 检查是否配置了 API_KEY
    if API_KEY is not None:
        # 获取请求头中的 Authorization
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="未提供有效的 API 密钥",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        api_key = auth_header.replace("Bearer ", "").strip()
        
        # 验证 API 密钥是否与配置的密钥匹配
        if api_key != API_KEY:
            raise HTTPException(status_code=401, detail="无效的 API 密钥")
    
    # 获取可用的密钥
    selected_key = None
    try:
        # 获取并验证可用的密钥
        async for session in get_async_db_session():
            # 修改这里，使用ApiKey.key作为查询字段，而不是id
            query = select(ApiKey.key).where(ApiKey.enabled == True)
            result = await session.execute(query)
            keys = result.scalars().all()
            
            if not keys:
                raise HTTPException(status_code=503, detail="没有可用的API密钥")
            
            # 随机选择一个密钥
            selected_key = random.choice(keys)
            
            # 更新使用计数
            update_query = ApiKey.__table__.update().where(ApiKey.key == selected_key).values(usage_count=ApiKey.usage_count + 1)
            await session.execute(update_query)
    except SQLAlchemyError as e:
        logger.error(f"数据库操作出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"数据库错误: {str(e)}")
    except Exception as e:
        logger.error(f"选择API密钥时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")
    
    # 解析请求体
    try:
        body = await request.json()
    except json.JSONDecodeError:
        logger.error("无法解析JSON请求体")
        raise HTTPException(status_code=400, detail="无效的JSON请求")
    
    # 获取模型名称，确保它不为空
    model = body.get("model", "")
    if not model:
        model = "GLM-4"  # 设置默认模型名称为硅基流动支持的模型
        body["model"] = model
    
    # 记录请求时间
    call_time_stamp = time.time()
    
    # 检查是否是流式请求
    is_stream = body.get("stream", False)
    
    # 准备转发请求的头部
    headers = {
        "Authorization": f"Bearer {selected_key}",
        "Content-Type": "application/json",
    }
    
    # 记录请求开始
    logger.info(f"开始处理请求: 模型={model}, 流式={is_stream}")
    
    if is_stream:
        # 处理流式响应
        return StreamingResponse(
            stream_response(selected_key, body, headers),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        # 处理非流式响应
        try:
            # 设置更合理的超时
            timeout = aiohttp.ClientTimeout(total=300, sock_connect=10, sock_read=300)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                try:
                    async with session.post(
                        f"{BASE_URL}/v1/chat/completions",
                        headers=headers,
                        json=body,
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"请求失败: {response.status} - {error_text}")
                            raise HTTPException(
                                status_code=response.status,
                                detail=f"API请求失败: {error_text}"
                            )
                        
                        response_data = await response.json()
                        
                        # 计算 token 使用量
                        usage = response_data.get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                        
                        # 记录使用情况
                        await log_completion(
                            selected_key,
                            model,
                            call_time_stamp,
                            prompt_tokens,
                            completion_tokens,
                            total_tokens,
                        )
                        
                        request_time = round((time.time() - start_time) * 1000)
                        logger.info(f"请求完成: 模型={model}, 输入tokens={prompt_tokens}, 输出tokens={completion_tokens}, 总tokens={total_tokens}, 处理时间={request_time}ms")
                        
                        return JSONResponse(response_data)
                
                except aiohttp.ClientError as e:
                    logger.error(f"HTTP客户端错误: {str(e)}", exc_info=True)
                    raise HTTPException(status_code=502, detail=f"上游服务请求失败: {str(e)}")
                except asyncio.TimeoutError:
                    logger.error(f"请求超时")
                    raise HTTPException(status_code=504, detail="请求超时")
        
        except Exception as e:
            logger.error(f"处理请求时出错: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"请求处理失败: {str(e)}")

async def direct_stream_response(api_key: str, body: bytes, headers: dict):
    """直接透传流式响应，不做任何处理"""
    model = "unknown"  # 默认模型名称
    call_time_stamp = time.time()
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    
    try:
        # 尝试从请求体中解析模型名称
        try:
            request_data = json.loads(body.decode('utf-8'))
            if 'model' in request_data:
                model = request_data['model']
        except:
            pass
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{BASE_URL}/v1/chat/completions",
                    headers=headers,
                    data=body,  # 使用原始请求体
                    timeout=300,
                ) as resp:
                    # 直接透传响应
                    async for chunk in resp.content:
                        yield chunk
                        
                        # 尝试解析chunk以获取token信息
                        try:
                            chunk_str = chunk.decode("utf-8").strip()
                            if chunk_str.startswith("data: ") and "usage" in chunk_str:
                                data_str = chunk_str[6:].strip()
                                if data_str and data_str != "[DONE]":
                                    data = json.loads(data_str)
                                    usage = data.get("usage", {})
                                    if usage:
                                        prompt_tokens = usage.get("prompt_tokens", 0)
                                        completion_tokens = usage.get("completion_tokens", 0)
                                        total_tokens = usage.get("total_tokens", 0)
                        except:
                            pass
                
                # 记录使用情况
                if prompt_tokens > 0 or completion_tokens > 0 or total_tokens > 0:
                    await log_completion(
                        api_key,
                        model,
                        call_time_stamp,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens or (prompt_tokens + completion_tokens)
                    )
                    logger.info(f"流式请求完成: 模型={model}, 输入tokens={prompt_tokens}, 输出tokens={completion_tokens}, 总tokens={total_tokens}")
            
            except aiohttp.ClientError as e:
                logger.error(f"HTTP客户端错误: {str(e)}")
                error_json = json.dumps({"error": {"message": f"上游服务请求失败: {str(e)}", "type": "upstream_error"}})
                yield f"data: {error_json}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
            
            except asyncio.TimeoutError:
                logger.error("请求超时")
                error_json = json.dumps({"error": {"message": "请求超时", "type": "timeout_error"}})
                yield f"data: {error_json}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
                    
    except Exception as e:
        logger.error(f"处理流式响应时出错: {str(e)}", exc_info=True)
        error_json = json.dumps({"error": {"message": str(e), "type": "server_error"}})
        yield f"data: {error_json}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

@app.post("/v1/embeddings")
async def embeddings(request: Request):
    if API_KEY is not None:
        request_api_key = request.headers.get("Authorization")
        if request_api_key != f"Bearer {API_KEY}":
            raise HTTPException(status_code=403, detail="无效的API_KEY")
    
    async for session in get_async_db_session():
        query = select(ApiKey.key).where(ApiKey.enabled == True)
        result = await session.execute(query)
        keys = result.scalars().all()
    
    if not keys:
        raise HTTPException(status_code=503, detail="没有可用的api-key")
    
    selected = random.choice(keys)
    forward_headers = dict(request.headers)
    forward_headers["Authorization"] = f"Bearer {selected}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BASE_URL}/v1/embeddings",
                headers=forward_headers,
                data=await request.body(),
                timeout=30,
            ) as resp:
                data = await resp.json()
                return JSONResponse(content=data, status_code=resp.status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"请求转发失败: {str(e)}")


@app.get("/v1/models")
async def list_models(request: Request):
    try:
        # 获取可用的key列表
        async for session in get_async_db_session():
            query = select(ApiKey.key).where(ApiKey.enabled == True)
            result = await session.execute(query)
            keys = result.scalars().all()
        
        if not keys:
            logger.error("没有可用的API密钥用于models请求")
            raise HTTPException(status_code=503, detail="没有可用的api-key")
        
        # 随机选择一个密钥
        selected = random.choice(keys)
        
        # 转发请求
        forward_headers = dict(request.headers)
        forward_headers["Authorization"] = f"Bearer {selected}"
        
        logger.info("开始获取可用模型列表")
        
        async with aiohttp.ClientSession() as session:
            # 设置更合理的超时
            timeout = aiohttp.ClientTimeout(total=30, sock_connect=10, sock_read=30)
            
            try:
                async with session.get(
                    f"{BASE_URL}/v1/models", 
                    headers=forward_headers, 
                    timeout=timeout
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"获取模型列表失败: {resp.status} - {error_text}")
                        return JSONResponse(
                            content={"error": {"message": error_text}},
                            status_code=resp.status
                        )
                    
                    data = await resp.json()
                    logger.info(f"获取模型列表成功")
                    return JSONResponse(content=data, status_code=resp.status)
            
            except aiohttp.ClientError as e:
                logger.error(f"获取模型列表HTTP客户端错误: {str(e)}", exc_info=True)
                raise HTTPException(status_code=502, detail=f"上游服务请求失败: {str(e)}")
            except asyncio.TimeoutError:
                logger.error(f"获取模型列表请求超时")
                raise HTTPException(status_code=504, detail="请求超时")
    
    except Exception as e:
        logger.error(f"处理models请求时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"请求转发失败: {str(e)}")


@app.get("/stats")
async def get_stats(authorized: bool = Depends(require_auth)):
    """获取基础统计信息"""
    try:
        stats = {}
        async for session in get_async_db_session():
            # 获取密钥计数
            key_count_result = await session.execute(select(func.count()).select_from(ApiKey))
            stats['key_count'] = key_count_result.scalar() or 0
            
            # 获取总余额
            total_balance_result = await session.execute(select(func.sum(ApiKey.balance)).select_from(ApiKey))
            stats['total_balance'] = total_balance_result.scalar() or 0
            
            # 获取调用总次数
            total_usage_result = await session.execute(select(func.sum(ApiKey.usage_count)).select_from(ApiKey))
            stats['total_usage'] = total_usage_result.scalar() or 0
            
            # 获取模型使用情况统计
            model_usage_query = select(
                Log.model, 
                func.count().label('count'), 
                func.sum(Log.input_tokens).label('input_tokens'),
                func.sum(Log.output_tokens).label('output_tokens'),
                func.sum(Log.total_tokens).label('total_tokens')
            ).group_by(Log.model)
            
            model_usage_result = await session.execute(model_usage_query)
            model_usage = [
                {
                    'model': model or 'unknown',
                    'count': count,
                    'input_tokens': input_tokens or 0,
                    'output_tokens': output_tokens or 0,
                    'total_tokens': total_tokens or 0
                }
                for model, count, input_tokens, output_tokens, total_tokens in model_usage_result
            ]
            
            stats['model_usage'] = model_usage
            
            # 获取近30天使用趋势 - 根据数据库类型使用不同的日期函数
            thirty_days_ago = datetime.now() - timedelta(days=30)
            timestamp = thirty_days_ago.timestamp()
            
            # 根据数据库类型构建不同的日期格式化查询
            if DB_TYPE == 'sqlite':
                date_func = func.strftime('%Y-%m-%d', func.datetime(Log.call_time, 'unixepoch'))
            else:  # mysql
                date_func = func.date(func.from_unixtime(Log.call_time))
            
            daily_usage_query = select(
                date_func.label('date'),
                func.count().label('count'),
                func.sum(Log.total_tokens).label('tokens')
            ).where(Log.call_time >= timestamp).group_by('date').order_by('date')
            
            try:
                daily_usage_result = await session.execute(daily_usage_query)
                daily_usage = [
                    {
                        'date': str(date),  # 确保日期以字符串形式返回
                        'count': count,
                        'tokens': tokens or 0
                    }
                    for date, count, tokens in daily_usage_result
                ]
                stats['daily_usage'] = daily_usage
            except Exception as e:
                logger.error(f"获取每日使用统计出错: {str(e)}")
                stats['daily_usage'] = []  # 出错时返回空列表
        
        # 使用自定义编码器处理Decimal类型
        return JSONResponse(content=jsonable_encoder(stats))
    
    except Exception as e:
        logger.error(f"获取统计信息时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@app.get("/export_keys")
async def export_keys(authorized: bool = Depends(require_auth), format: str = "text"):
    """导出所有API密钥的信息，支持text和json两种格式"""
    async for session in get_async_db_session():
        query = select(ApiKey)
        result = await session.execute(query)
        keys = result.scalars().all()
        
        if format.lower() == "json":
            # JSON格式导出
            data = [
                {
                    "key": key.key,
                    "add_time": key.add_time,
                    "balance": key.balance,
                    "usage_count": key.usage_count,
                    "enabled": key.enabled
                }
                for key in keys
            ]
            
            # 使用自定义编码器处理Decimal类型
            headers = {"Content-Disposition": "attachment; filename=keys.json"}
            return JSONResponse(content=jsonable_encoder(data), headers=headers)
        else:
            # 文本格式导出（默认）
            key_texts = [key.key for key in keys]
            keys_text = "\n".join(key_texts)
            headers = {"Content-Disposition": "attachment; filename=keys.txt"}
            return Response(content=keys_text, media_type="text/plain", headers=headers)


@app.get("/logs")
async def logs(
    page: int = 1, 
    page_size: int = 50, 
    sort_field: str = "call_time", 
    sort_order: str = "desc",
    authorized: bool = Depends(require_auth)
):
    """分页获取使用日志"""
    try:
        async for session in get_async_db_session():
            # 确定总记录数
            count_query = select(func.count()).select_from(Log)
            result = await session.execute(count_query)
            total = result.scalar() or 0
            
            # 确定排序
            if sort_field not in ["call_time", "input_tokens", "output_tokens", "total_tokens"]:
                sort_field = "call_time"  # 默认排序字段
                
            # 获取排序字段
            sort_column = getattr(Log, sort_field)
            if sort_order.lower() == "asc":
                order_by = asc(sort_column)
            else:
                order_by = desc(sort_column)
            
            # 查询对应页的记录
            offset = (page - 1) * page_size
            query = select(Log).order_by(order_by).offset(offset).limit(page_size)
            result = await session.execute(query)
            logs = result.scalars().all()
            
            # 构建分页结果
            data = {
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size,
                "items": [
                    {
                        "id": log.id,
                        "used_key": log.used_key,
                        "model": log.model,
                        "call_time": log.call_time,
                        "timestamp": datetime.fromtimestamp(log.call_time).strftime("%Y-%m-%d %H:%M:%S"),
                        "input_tokens": log.input_tokens,
                        "output_tokens": log.output_tokens,
                        "total_tokens": log.total_tokens
                    }
                    for log in logs
                ]
            }
            
            # 使用自定义JSON编码器序列化
            return JSONResponse(content=jsonable_encoder(data))
            
    except Exception as e:
        logger.error(f"获取使用日志时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取使用日志失败: {str(e)}")


@app.post("/clear_logs")
async def clear_logs(authorized: bool = Depends(require_auth)):
    try:
        async for session in get_async_db_session():
            delete_query = Log.__table__.delete()
            await session.execute(delete_query)
        
        return JSONResponse({"message": "日志已清空"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空日志失败: {str(e)}")


@app.get("/api/keys")
async def api_keys(
    page: int = 1, 
    page_size: int = 10, 
    sort_field: str = "add_time", 
    sort_order: str = "desc",
    authorized: bool = Depends(require_auth)
):
    """分页获取API密钥列表"""
    try:
        async for session in get_async_db_session():
            # 确定总记录数
            count_query = select(func.count()).select_from(ApiKey)
            result = await session.execute(count_query)
            total = result.scalar() or 0
            
            # 确定排序
            if sort_field not in ["add_time", "balance", "usage_count"]:
                sort_field = "add_time"  # 默认排序字段
                
            # 获取排序字段
            sort_column = getattr(ApiKey, sort_field)
            if sort_order.lower() == "asc":
                order_by = asc(sort_column)
            else:
                order_by = desc(sort_column)
            
            # 查询对应页的记录
            offset = (page - 1) * page_size
            query = select(ApiKey).order_by(order_by).offset(offset).limit(page_size)
            result = await session.execute(query)
            keys = result.scalars().all()
            
            # 构建分页结果
            data = {
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size,
                "items": [
                    {
                        "key": key.key,
                        "add_time": key.add_time,
                        "balance": key.balance,
                        "usage_count": key.usage_count,
                        "enabled": key.enabled
                    }
                    for key in keys
                ]
            }
            
            # 使用自定义JSON编码器序列化
            return JSONResponse(content=jsonable_encoder(data))
            
    except Exception as e:
        logger.error(f"获取API密钥列表时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取API密钥列表失败: {str(e)}")

@app.post("/api/refresh_key")
async def refresh_single_key(request: Request, authorized: bool = Depends(require_auth)):
    data = await request.json()
    key = data.get("key")
    
    if not key:
        raise HTTPException(status_code=400, detail="未提供API密钥")
    
    # 验证密钥
    validation_result = await validate_key_async(key)
    
    # 检查验证结果
    if validation_result.get("valid", False):
        # 获取余额
        balance = validation_result.get("balance", 0.0)
        
        # 更新数据库中的密钥余额
        async for session in get_async_db_session():
            query = select(ApiKey).where(ApiKey.key == key)
            result = await session.execute(query)
            api_key = result.scalar_one_or_none()
            if api_key:
                # 更新密钥余额
                api_key.balance = balance
                # 更新最后使用时间
                if hasattr(api_key, 'last_used'):
                    api_key.last_used = time.time()
                await session.commit()
                
                logger.info(f"密钥刷新成功: {mask_key(key)}, 余额: {balance}")
        
        return JSONResponse({
            "message": "密钥刷新成功", 
            "valid": True,
            "balance": balance
        })
    else:
        # 密钥无效处理
        error_message = validation_result.get("error", "密钥无效")
        status_code = validation_result.get("status_code", 400)
        
        # 禁用无效密钥
        try:
            async for session in get_async_db_session():
                # 查找密钥
                query = select(ApiKey).where(ApiKey.key == key)
                result = await session.execute(query)
                api_key = result.scalar_one_or_none()
                
                if api_key:
                    # 禁用密钥而不是删除
                    api_key.enabled = False
                    await session.commit()
                    logger.warning(f"已禁用无效密钥: {mask_key(key)}")
        except Exception as e:
            logger.error(f"禁用无效密钥时出错: {str(e)}")
        
        raise HTTPException(
            status_code=status_code, 
            detail=f"密钥验证失败: {error_message}"
        )

@app.get("/api/key_info")
async def key_info(key: str, authorized: bool = Depends(require_auth)):
    """获取单个API密钥的详细信息"""
    try:
        async for session in get_async_db_session():
            # 查询API密钥
            key_query = select(ApiKey).where(ApiKey.key == key)
            result = await session.execute(key_query)
            api_key = result.scalar_one_or_none()
            
            if not api_key:
                raise HTTPException(status_code=404, detail="API密钥不存在")
            
            # 查询最近的使用记录
            log_query = select(Log).where(Log.used_key == key).order_by(desc(Log.call_time)).limit(10)
            result = await session.execute(log_query)
            logs = result.scalars().all()
            
            # 构建API密钥信息
            key_info = {
                "key": api_key.key,
                "add_time": api_key.add_time,
                "balance": api_key.balance,
                "usage_count": api_key.usage_count,
                "enabled": api_key.enabled,
                "recent_logs": [
                    {
                        "id": log.id,
                        "model": log.model,
                        "call_time": log.call_time,
                        "timestamp": datetime.fromtimestamp(log.call_time).strftime("%Y-%m-%d %H:%M:%S"),
                        "input_tokens": log.input_tokens,
                        "output_tokens": log.output_tokens,
                        "total_tokens": log.total_tokens
                    }
                    for log in logs
                ]
            }
            
            # 使用自定义JSON编码器处理Decimal类型
            return JSONResponse(content=jsonable_encoder(key_info))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取API密钥信息时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取API密钥信息失败: {str(e)}")

# Add a general exception handler for uncaught exceptions
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    # Log the error here if needed
    logger.error(f"未处理的异常: {str(exc)}", exc_info=True)
    
    # For API endpoints, return JSON error
    if request.url.path.startswith("/v1/") or request.headers.get("accept") == "application/json":
        return JSONResponse(
            status_code=500,
            content={"detail": "服务器内部错误"}
        )
    # For web pages, return the 500 error page
    return FileResponse("static/500.html", status_code=500)


@app.get("/keys")
async def keys_page(authorized: bool = Depends(require_auth)):
    response = FileResponse("static/keys.html")
    # 添加缓存控制头
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/log_cleanup_config")
async def get_log_cleanup_config(authorized: bool = Depends(require_auth)):
    """获取日志清理配置"""
    return JSONResponse({
        "enabled": LOG_AUTO_CLEAN,
        "retention_days": LOG_RETENTION_DAYS,
        "interval_hours": LOG_CLEAN_INTERVAL_HOURS,
        "backup_enabled": LOG_BACKUP_ENABLED,
        "backup_dir": LOG_BACKUP_DIR
    })


@app.post("/refresh")
async def refresh_all_keys(request: Request, authorized: bool = Depends(require_auth)):
    """刷新所有API密钥，更新余额信息（并发执行）"""
    try:
        # 获取所有API密钥
        refreshed_keys = 0
        failed_keys = 0
        max_concurrent = 10  # 最大并发数
        
        async for session in get_async_db_session():
            # 查询所有API密钥
            query = select(ApiKey)
            result = await session.execute(query)
            keys = result.scalars().all()
            
            total_keys = len(keys)
            if total_keys == 0:
                return JSONResponse(
                    content={
                        "success": True,
                        "message": "没有找到API密钥",
                        "refreshed": 0,
                        "failed": 0,
                        "total": 0
                    }
                )
            
            logger.info(f"开始刷新 {total_keys} 个API密钥，最大并发数: {max_concurrent}")
            
            # 创建一个信号量来限制并发数
            semaphore = asyncio.Semaphore(max_concurrent)
            
            # 创建任务列表和结果跟踪字典
            tasks = []
            results = {"refreshed": 0, "failed": 0}
            
            # 定义单个密钥刷新协程
            async def refresh_single_key(key_obj):
                # 使用信号量限制并发数
                async with semaphore:
                    try:
                        # 验证API密钥
                        validation_result = await validate_key_async(key_obj.key)
                        
                        # 构建更新结果
                        update_result = {
                            "key": mask_key(key_obj.key),
                            "valid": validation_result.get("valid", False),
                            "status": "success" if validation_result.get("valid", False) else "failed"
                        }
                        
                        if validation_result.get("valid", False):
                            # 获取余额，更新密钥
                            balance = validation_result.get("balance", 0.0)
                            update_result["balance"] = balance
                            
                            # 在数据库中更新密钥
                            async for update_session in get_async_db_session():
                                try:
                                    # 重新查询以避免并发问题
                                    key_query = select(ApiKey).where(ApiKey.key == key_obj.key)
                                    key_result = await update_session.execute(key_query)
                                    db_key = key_result.scalar_one_or_none()
                                    
                                    if db_key:
                                        db_key.balance = balance
                                        # 更新最后使用时间
                                        if hasattr(db_key, 'last_used'):
                                            db_key.last_used = time.time()
                                        await update_session.commit()
                                        
                                        logger.info(f"密钥刷新成功: {mask_key(key_obj.key)}, 余额: {balance}")
                                        async with results_lock:
                                            results["refreshed"] += 1
                                except Exception as db_error:
                                    logger.error(f"数据库更新密钥时出错: {str(db_error)}")
                                    update_result["status"] = "db_error"
                                    update_result["error"] = str(db_error)
                                    async with results_lock:
                                        results["failed"] += 1
                        else:
                            # 密钥验证失败，禁用密钥
                            error_message = validation_result.get("error", "密钥无效")
                            logger.warning(f"API密钥无效，将被禁用: {mask_key(key_obj.key)}, 错误: {error_message}")
                            
                            update_result["error"] = error_message
                            
                            # 禁用密钥
                            async for update_session in get_async_db_session():
                                try:
                                    # 重新查询以避免并发问题
                                    key_query = select(ApiKey).where(ApiKey.key == key_obj.key)
                                    key_result = await update_session.execute(key_query)
                                    db_key = key_result.scalar_one_or_none()
                                    
                                    if db_key:
                                        db_key.enabled = False  # 禁用密钥而不是删除
                                        await update_session.commit()
                                except Exception as db_error:
                                    logger.error(f"禁用无效密钥时出错: {str(db_error)}")
                                    update_result["db_error"] = str(db_error)
                            
                            async with results_lock:
                                results["failed"] += 1
                    
                    except Exception as e:
                        logger.error(f"刷新API密钥时出错: {mask_key(key_obj.key)}, 错误: {str(e)}")
                        async with results_lock:
                            results["failed"] += 1
                        return {"key": mask_key(key_obj.key), "status": "error", "error": str(e)}
                    
                    return update_result
            
            # 创建一个锁用于更新结果计数
            results_lock = asyncio.Lock()
            
            # 创建所有刷新任务
            for key_obj in keys:
                task = asyncio.create_task(refresh_single_key(key_obj))
                tasks.append(task)
            
            # 等待所有任务完成
            refresh_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            valid_results = [r for r in refresh_results if isinstance(r, dict)]
            error_results = [r for r in refresh_results if isinstance(r, Exception)]
            
            if error_results:
                for error in error_results:
                    logger.error(f"刷新任务异常: {str(error)}")
            
            refreshed_keys = results["refreshed"]
            failed_keys = results["failed"]
            
            # 记录结果
            logger.info(f"刷新API密钥完成: 成功 {refreshed_keys} 个, 失败 {failed_keys} 个, 总计 {total_keys} 个")
        
        return JSONResponse(
            content={
                "success": True,
                "message": f"刷新完成: 成功 {refreshed_keys} 个, 失败 {failed_keys} 个, 总计 {total_keys} 个",
                "refreshed": refreshed_keys,
                "failed": failed_keys,
                "total": total_keys
            }
        )
    
    except Exception as e:
        logger.error(f"刷新所有API密钥时发生错误: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"刷新API密钥时发生错误: {str(e)}"
            }
        )


@app.post("/log_cleanup_config")
async def update_log_cleanup_config(request: Request, authorized: bool = Depends(require_auth)):
    """更新日志清理配置"""
    global LOG_AUTO_CLEAN, LOG_RETENTION_DAYS, LOG_CLEAN_INTERVAL_HOURS, LOG_BACKUP_ENABLED, LOG_BACKUP_DIR
    
    try:
        data = await request.json()
        
        # 更新配置
        if "enabled" in data:
            LOG_AUTO_CLEAN = bool(data["enabled"])
        
        if "retention_days" in data:
            retention_days = int(data["retention_days"])
            if retention_days < 1:
                raise ValueError("保留天数必须大于0")
            LOG_RETENTION_DAYS = retention_days
        
        if "interval_hours" in data:
            interval_hours = int(data["interval_hours"])
            if interval_hours < 1:
                raise ValueError("清理间隔必须大于0小时")
            LOG_CLEAN_INTERVAL_HOURS = interval_hours
        
        # 新增备份配置
        if "backup_enabled" in data:
            LOG_BACKUP_ENABLED = bool(data["backup_enabled"])
        
        if "backup_dir" in data and data["backup_dir"]:
            LOG_BACKUP_DIR = str(data["backup_dir"])
            # 测试目录是否可写
            try:
                os.makedirs(LOG_BACKUP_DIR, exist_ok=True)
                test_file = os.path.join(LOG_BACKUP_DIR, ".test_write")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                raise ValueError(f"备份目录不可写: {str(e)}")
        
        # 返回更新后的配置
        return JSONResponse({
            "message": "日志清理配置已更新",
            "config": {
                "enabled": LOG_AUTO_CLEAN,
                "retention_days": LOG_RETENTION_DAYS,
                "interval_hours": LOG_CLEAN_INTERVAL_HOURS,
                "backup_enabled": LOG_BACKUP_ENABLED,
                "backup_dir": LOG_BACKUP_DIR
            }
        })
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新配置失败: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查接口，验证系统各组件状态"""
    # 准备响应结构
    health_status = {
        "status": "healthy",
        "details": {
            "api_service": "healthy",
            "database": "healthy",
            "uptime": 0,
            "memory_usage": 0,
            "cpu_usage": 0,
            "active_keys": 0,
            "version": "1.0.0"
        },
        "timestamp": time.time()
    }
    
    # 尝试检查数据库连接
    db_start_time = time.time()
    db_error = None
    
    try:
        if DB_TYPE == 'sqlite':
            with get_db_session() as session:
                result = session.execute(text("SELECT 1")).scalar()
                if result != 1:
                    raise Exception("数据库连接测试失败")
        else:  # MySQL
            async with new_session() as session:
                result = await session.execute(text("SELECT 1"))
                if result.scalar() != 1:
                    raise Exception("数据库连接测试失败")
                
                # 检查活跃密钥数量
                active_keys_query = select(func.count(ApiKey.id)).where(ApiKey.enabled == True)
                active_keys_result = await session.execute(active_keys_query)
                health_status["details"]["active_keys"] = active_keys_result.scalar() or 0
                
        # 记录数据库响应时间
        db_response_time = time.time() - db_start_time
        health_status["details"]["database_response_time"] = round(db_response_time * 1000, 2)  # 毫秒
        
    except Exception as e:
        db_error = str(e)
        health_status["status"] = "degraded"
        health_status["details"]["database"] = "unhealthy"
        health_status["details"]["database_error"] = db_error
    
    # 检查外部API连通性
    api_start_time = time.time()
    api_error = None
    
    try:
        # 使用共享会话发送HTTP GET请求测试
        session = await http_client.get_session()
        timeout = aiohttp.ClientTimeout(total=5)  # 5秒超时
        async with session.get("https://api.siliconflow.cn/v1/models", timeout=timeout) as response:
            if response.status != 200:
                raise Exception(f"API连接测试失败，状态码: {response.status}")
            
            # 记录API响应时间
            api_response_time = time.time() - api_start_time
            health_status["details"]["api_response_time"] = round(api_response_time * 1000, 2)  # 毫秒
    except Exception as e:
        api_error = str(e)
        health_status["status"] = "degraded"
        health_status["details"]["api_service"] = "unhealthy"
        health_status["details"]["api_error"] = api_error
    
    # 获取系统运行时间
    if hasattr(app, "state") and hasattr(app.state, "startup_timestamp"):
        uptime_seconds = time.time() - app.state.startup_timestamp
        days, remainder = divmod(uptime_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        health_status["details"]["uptime"] = {
            "total_seconds": int(uptime_seconds),
            "formatted": f"{int(days)}天 {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒"
        }
    
    # 获取内存使用情况
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        health_status["details"]["memory_usage"] = {
            "rss_mb": round(memory_info.rss / (1024 * 1024), 2),  # RSS内存(MB)
            "vms_mb": round(memory_info.vms / (1024 * 1024), 2),   # 虚拟内存(MB)
            "percent": round(process.memory_percent(), 2)          # 内存使用百分比
        }
        
        # CPU使用情况
        health_status["details"]["cpu_usage"] = {
            "percent": round(process.cpu_percent(interval=0.1), 2),  # CPU使用百分比
            "threads": len(process.threads()),                        # 线程数
            "system_load": [round(x, 2) for x in psutil.getloadavg()]  # 系统负载
        }
    except ImportError:
        health_status["details"]["memory_usage"] = "未安装psutil，无法获取内存信息"
        health_status["details"]["cpu_usage"] = "未安装psutil，无法获取CPU信息"
    except Exception as e:
        health_status["details"]["system_error"] = str(e)
    
    # 设置响应的HTTP状态码
    status_code = 200 if health_status["status"] == "healthy" else 503
    
    return JSONResponse(content=health_status, status_code=status_code)


@app.get("/metrics")
async def metrics(authorized: bool = Depends(require_auth)):
    """提供应用指标数据，需要管理员权限"""
    try:
        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "api_keys": {},
            "usage": {},
            "today": {},
            "model_usage": [],
            "daily_trend": [],
            "system": {
                "uptime_seconds": round(time.time() - app.state.startup_timestamp, 2) if hasattr(app.state, "startup_timestamp") else None,
                "process_memory_mb": round(get_process_memory_mb(), 2)
            }
        }
        
        async for session in get_async_db_session():
            # 获取密钥指标
            key_count_query = select(func.count()).select_from(ApiKey)
            result = await session.execute(key_count_query)
            key_count = result.scalar() or 0
            
            key_stats_query = select(
                func.sum(ApiKey.balance).label("total_balance"),
                func.avg(ApiKey.balance).label("avg_balance"),
                func.sum(ApiKey.usage_count).label("total_usage")
            ).select_from(ApiKey)
            result = await session.execute(key_stats_query)
            key_stats_result = result.one_or_none()
            
            # 获取日志指标
            log_count_query = select(func.count()).select_from(Log)
            result = await session.execute(log_count_query)
            log_count = result.scalar() or 0
            
            tokens_query = select(
                func.sum(Log.input_tokens).label("total_input_tokens"),
                func.sum(Log.output_tokens).label("total_output_tokens"),
                func.sum(Log.total_tokens).label("total_tokens")
            ).select_from(Log)
            result = await session.execute(tokens_query)
            tokens_result = result.one_or_none()
            
            # 获取今日统计
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            today_query = select(
                func.count().label("today_requests"),
                func.sum(Log.total_tokens).label("today_tokens")
            ).select_from(Log).where(Log.call_time >= today_start)
            result = await session.execute(today_query)
            today_result = result.one_or_none()
            
            # 获取模型使用情况
            model_usage_query = select(
                Log.model,
                func.count().label("count"),
                func.sum(Log.total_tokens).label("total_tokens")
            ).select_from(Log).group_by(Log.model).order_by(desc("count"))
            result = await session.execute(model_usage_query)
            model_usage_result = result.all()
            
            # 获取最近一周的每日趋势
            week_ago = datetime.now() - timedelta(days=7)
            week_timestamp = week_ago.timestamp()
            
            # 根据数据库类型构建不同的日期格式化查询
            if DB_TYPE == 'sqlite':
                date_func = func.strftime('%Y-%m-%d', func.datetime(Log.call_time, 'unixepoch'))
            else:  # mysql
                date_func = func.date(func.from_unixtime(Log.call_time))
            
            try:
                daily_trend_query = select(
                    date_func.label('date'),
                    func.count().label('count'),
                    func.sum(Log.total_tokens).label('tokens')
                ).where(Log.call_time >= week_timestamp).group_by('date').order_by('date')
                
                result = await session.execute(daily_trend_query)
                daily_trend = [
                    {
                        "date": str(date),
                        "count": count,
                        "tokens": tokens or 0
                    }
                    for date, count, tokens in result
                ]
                metrics_data["daily_trend"] = daily_trend
            except Exception as e:
                logger.error(f"获取每日趋势统计出错: {str(e)}")
                metrics_data["daily_trend"] = []  # 出错时返回空列表
            
            # 构建响应
            metrics_data["api_keys"] = {
                "count": key_count,
                "total_balance": key_stats_result[0] if key_stats_result else 0,
                "average_balance": key_stats_result[1] if key_stats_result else 0,
                "total_usage": key_stats_result[2] if key_stats_result else 0
            }
            
            metrics_data["usage"] = {
                "total_requests": log_count,
                "total_input_tokens": tokens_result[0] if tokens_result else 0,
                "total_output_tokens": tokens_result[1] if tokens_result else 0,
                "total_tokens": tokens_result[2] if tokens_result else 0,
            }
            
            metrics_data["today"] = {
                "requests": today_result[0] if today_result else 0,
                "tokens": today_result[1] if today_result else 0
            }
            
            metrics_data["model_usage"] = [
                {
                    "model": item[0] or "unknown",
                    "request_count": item[1],
                    "total_tokens": item[2] or 0
                }
                for item in model_usage_result
            ]
        
        # 使用自定义编码器处理Decimal类型
        return JSONResponse(content=jsonable_encoder(metrics_data))
    
    except Exception as e:
        logger.error(f"获取指标时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取指标失败: {str(e)}")


def get_process_memory_mb():
    """获取当前进程的内存使用量（MB）"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # 转换为MB
    except ImportError:
        return 0  # 如果psutil不可用，返回0
    except Exception:
        return 0  # 出错时返回0

@app.post("/v1/{path:path}")
async def proxy_request(request: Request, path: str):
    """代理转发所有API请求到原始API接口"""
    # 获取客户端原始请求的数据和头信息
    request_data = await request.body()
    raw_data = request_data.decode('utf-8') if request_data else None
    request_headers = dict(request.headers)
    method = request.method
    
    # 提取模型名称，用于统计和日志
    model_name = "unknown"
    content_type = request_headers.get("content-type", "")
    
    if content_type and "application/json" in content_type and raw_data:
        try:
            json_data = json.loads(raw_data)
            # 针对不同路径提取模型信息
            if path == "chat/completions" and "model" in json_data:
                model_name = json_data["model"]
            elif path == "embeddings" and "model" in json_data:
                model_name = json_data["model"]
        except json.JSONDecodeError:
            logger.warning("无法解析请求JSON数据以提取模型名称")
    
    # 检查授权头
    auth_header = request_headers.get("authorization", "")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.warning("缺少有效的Authorization头")
        return JSONResponse(
            status_code=401,
            content={"error": "缺少有效的Authorization头"}
        )
    
    # 从数据库中随机选择一个API密钥
    async with new_session() as session:
        try:
            # 以抽奖方式随机选择一个密钥
            stmt = select(Key).where(Key.active == 1).order_by(func.random()).limit(1)
            result = await session.execute(stmt)
            key = result.scalar_one_or_none()
            
            if not key:
                logger.error("数据库中没有可用的API密钥")
                return JSONResponse(
                    status_code=500,
                    content={"error": "没有可用的API密钥"}
                )
            
            # 提取客户端真实IP
            client_ip = request.client.host if request.client else "未知IP"
            # 尝试从X-Forwarded-For或其他代理头获取真实IP
            forwarded_for = request_headers.get("x-forwarded-for")
            if forwarded_for:
                # 取第一个IP作为客户端真实IP
                client_ip = forwarded_for.split(",")[0].strip()
            
            # 提取自定义请求ID或生成一个
            request_id = request_headers.get("x-request-id", str(uuid.uuid4()))
            
            # 准备转发请求
            target_url = f"https://api.siliconflow.cn/v1/{path}"
            # 继承原始请求头，但修改Authorization头
            headers = {**request_headers}
            headers["Authorization"] = f"Bearer {key.key}"
            
            # 移除可能导致问题的代理头
            for header in ["host", "content-length", "transfer-encoding"]:
                if header in headers:
                    del headers[header]
            
            # 添加请求跟踪ID
            headers["x-request-id"] = request_id
            
            # 记录请求信息
            logger.info(f"转发请求: {request_id} | 客户端: {client_ip} | 路径: {path} | 模型: {model_name} | 密钥: {mask_key(key.key)}")
            
            # 估算Token数量
            token_count = 0
            if raw_data and path == "chat/completions":
                try:
                    token_count = estimate_chat_tokens(json.loads(raw_data))
                    logger.info(f"估算请求Token: {token_count} | 请求ID: {request_id}")
                except Exception as e:
                    logger.warning(f"Token估算失败: {str(e)}")
            
            # 使用全局HTTP会话发送请求
            session = await http_client.get_session()
            target_timeout = aiohttp.ClientTimeout(total=600)  # 设置10分钟超时
            
            # 根据请求方法发送不同类型的请求
            async with session.request(
                method=method,
                url=target_url,
                headers=headers,
                data=request_data,
                timeout=target_timeout,
                allow_redirects=True
            ) as response:
                # 读取响应数据
                response_data = await response.read()
                response_headers = dict(response.headers)
                
                # 计算完成Token(对于流式输出以外的内容)
                completion_tokens = 0
                if not response_headers.get("content-type", "").startswith("text/event-stream") and path == "chat/completions":
                    try:
                        response_json = json.loads(response_data)
                        if "usage" in response_json and "completion_tokens" in response_json["usage"]:
                            completion_tokens = response_json["usage"]["completion_tokens"]
                            logger.info(f"完成Token: {completion_tokens} | 请求ID: {request_id}")
                    except Exception as e:
                        logger.warning(f"无法从响应中提取Token使用信息: {str(e)}")
                
                # 记录日志
                log_entry = Log(
                    model=model_name,
                    path=path,
                    key_id=key.id,
                    key=mask_key(key.key),
                    client_ip=client_ip,
                    request_id=request_id,
                    request_tokens=token_count,
                    completion_tokens=completion_tokens,
                    status_code=response.status,
                    call_time=int(time.time())
                )
                
                try:
                    session.add(log_entry)
                    await session.commit()
                except Exception as e:
                    logger.error(f"记录日志失败: {str(e)}", exc_info=True)
                    await session.rollback()
                
                # 更新密钥使用计数
                try:
                    key.usage_count += 1
                    key.last_used = int(time.time())
                    await session.commit()
                except Exception as e:
                    logger.error(f"更新密钥使用计数失败: {str(e)}", exc_info=True)
                    await session.rollback()
                
                # 构建响应
                logger.info(f"请求完成: {request_id} | 状态码: {response.status} | 总Token: {token_count + completion_tokens}")
                
                # 处理流式响应
                if response_headers.get("content-type", "").startswith("text/event-stream"):
                    return StreamingResponse(
                        content=BytesIO(response_data),
                        status_code=response.status,
                        headers=response_headers
                    )
                
                # 处理普通响应
                return Response(
                    content=response_data,
                    status_code=response.status,
                    headers=response_headers
                )
        
        except NoResultFound:
            logger.error("没有找到活跃的API密钥")
            return JSONResponse(
                status_code=500,
                content={"error": "没有可用的API密钥"}
            )
        except Exception as e:
            logger.error(f"代理请求过程中发生错误: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": f"服务器内部错误: {str(e)}"}
            )

@app.get("/")
async def root(session_id: str = Cookie(None)):
    """根路径处理，返回登录页面或重定向到管理页面"""
    # 检查用户是否已登录
    if session_id:
        try:
            valid_session = await validate_session(session_id)
            if valid_session:
                # 用户已登录，重定向到 admin 页面
                return RedirectResponse(url="/admin")
        except Exception as e:
            logger.error(f"检查会话时出错: {str(e)}")
            # 出错时继续显示登录页面
    
    # 用户未登录或会话无效，显示登录页面
    return FileResponse("static/index.html")


@app.post("/login")
async def login(request: Request):
    """处理用户登录请求"""
    try:
        data = await request.json()
        username = data.get("username")
        password = data.get("password")
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            # 创建会话
            session_id = await create_session(username)
            
            # 构建响应
            response = JSONResponse({"message": "登录成功"})
            
            # 设置cookie
            response.set_cookie(
                key="session_id",
                value=session_id,
                httponly=True,
                max_age=86400,  # 24小时
                samesite="lax"
            )
            
            return response
        else:
            # 登录失败
            logger.warning(f"登录失败: 用户名 {username}")
            return JSONResponse(
                status_code=401,
                content={"detail": "用户名或密码错误"}
            )
    except Exception as e:
        logger.error(f"登录处理出错: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": f"登录处理出错: {str(e)}"}
        )


@app.get("/admin")
async def admin_page(authorized: bool = Depends(require_auth)):
    """管理员页面，需要身份验证"""
    response = FileResponse("static/admin.html")
    # 添加缓存控制头
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/logout")
async def logout(session_id: str = Cookie(None)):
    """注销登录，删除会话"""
    # 清除数据库中的会话
    if session_id:
        try:
            async for session in get_async_db_session():
                # 删除会话
                delete_query = DbSession.__table__.delete().where(DbSession.session_id == session_id)
                await session.execute(delete_query)
                await session.commit()
                logger.info(f"会话已删除: {session_id}")
        except Exception as e:
            logger.error(f"删除会话时出错: {str(e)}")
    
    # 创建重定向响应
    response = RedirectResponse(url="/", status_code=303)  # 使用 303 See Other 状态码
    
    # 添加缓存控制头，防止浏览器缓存
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    
    # 删除 cookie
    response.delete_cookie(
        key="session_id",
        path="/",  # 确保删除所有路径下的 cookie
        httponly=True
    )
    
    return response

def mask_key(key: str) -> str:
    """对API密钥进行掩码，只保留前4位和后4位，中间用星号替代"""
    if not key or len(key) < 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7898)

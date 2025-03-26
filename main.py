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
from contextlib import asynccontextmanager
from uvicorn.config import LOGGING_CONFIG
from datetime import datetime, timedelta
import os
import logging
import logging.handlers
from sqlalchemy import select, func, desc, asc
from sqlalchemy.exc import SQLAlchemyError
import sqlite3

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
from db_manager import get_db_session, get_async_db_session, ApiKey, Log, Session as DbSession, check_db_health

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

app = FastAPI(
    title="SiliconFlow API",
    description="SiliconFlow API代理服务",
    version="1.0.0"
)

# Mount static files
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
        
        # 查询需要备份的日志
        async for session in get_async_db_session():
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
                        f.write(f"{log.id},{log.used_key},{log.model},{log.call_time},{log.input_tokens},{log.output_tokens},{log.total_tokens}\n")
                
                logger.info(f"已备份 {len(logs)} 条日志记录到 {backup_filename}")
    
    except Exception as e:
        logger.error(f"备份日志失败: {str(e)}")

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
                async for session in get_async_db_session():
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
            
            # 等待下一次执行
            next_run = datetime.now() + timedelta(hours=LOG_CLEAN_INTERVAL_HOURS)
            logger.info(f"下一次日志清理将在 {next_run.strftime('%Y-%m-%d %H:%M:%S')} 进行")
            
            await asyncio.sleep(LOG_CLEAN_INTERVAL_HOURS * 3600)  # 转换为秒
        
        except Exception as e:
            logger.error(f"日志自动清理出错: {str(e)}")
            await asyncio.sleep(3600)  # 发生错误时等待1小时后重试

# 启动时运行自动清理任务
@app.on_event("startup")
async def startup_event():
    # 记录启动时间戳
    app.state.startup_timestamp = time.time()
    
    logger.info("应用启动，初始化清理任务...")
    asyncio.create_task(auto_clean_logs())

# 应用关闭时确保连接关闭
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("应用关闭，清理资源...")
    # 仅在SQLite模式下关闭连接
    if DB_TYPE == 'sqlite' and 'conn' in globals():
        conn.close()
        print("SQLite数据库连接已关闭")

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
    session_id = secrets.token_hex(16)
    async for session in get_async_db_session():
        db_session = DbSession(
            session_id=session_id,
            username=username,
            created_at=time.time()
        )
        session.add(db_session)
    
    return session_id


async def validate_session(session_id: str = Cookie(None)) -> bool:
    if not session_id:
        return False
    
    # 清理旧会话
    try:
        async for session in get_async_db_session():
            # 删除过期会话（24小时前）
            delete_query = DbSession.__table__.delete().where(DbSession.created_at < time.time() - 86400)
            await session.execute(delete_query)
    except Exception as e:
        logger.error(f"清理过期会话时出错: {str(e)}")
    
    # 检查当前会话
    try:
        async for session in get_async_db_session():
            query = select(DbSession).where(DbSession.session_id == session_id)
            result = await session.execute(query)
            return bool(result.scalar_one_or_none())
    except Exception as e:
        logger.error(f"验证会话时出错: {str(e)}")
        return False


async def require_auth(session_id: str = Cookie(None)):
    """验证用户是否已登录，如果未登录则抛出401异常"""
    if not session_id:
        raise HTTPException(status_code=401, detail="未授权访问，请先登录")
    
    # 清理过期会话
    try:
        async for session in get_async_db_session():
            # 删除过期会话（24小时前）
            delete_query = DbSession.__table__.delete().where(DbSession.created_at < time.time() - 86400)
            await session.execute(delete_query)
    except Exception as e:
        logger.error(f"清理过期会话时出错: {str(e)}")
    
    # 检查当前会话
    try:
        async for session in get_async_db_session():
            query = select(DbSession).where(DbSession.session_id == session_id)
            result = await session.execute(query)
            
            if not result.scalar_one_or_none():
                raise HTTPException(status_code=401, detail="会话已过期，请重新登录")
    except SQLAlchemyError as e:
        logger.error(f"检查会话时出错: {str(e)}")
        raise HTTPException(status_code=401, detail="验证会话时出错，请重新登录")
    
    return True


async def validate_key_async(api_key: str):
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.siliconflow.cn/v1/user/info", headers=headers, timeout=10
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    return True, data.get("data", {}).get("totalBalance", 0)
                else:
                    data = await r.json()
                    return False, data.get("message", "验证失败")
    except Exception as e:
        return False, f"请求失败: {str(e)}"


def insert_api_key(api_key: str, balance: float):
    with get_db_session() as session:
        new_key = ApiKey(
            key=api_key,
            add_time=time.time(),
            balance=balance,
            usage_count=0
        )
        session.add(new_key)


def log_completion(
    used_key: str,
    model: str,
    call_time: float,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
):
    with get_db_session() as session:
        log_entry = Log(
            used_key=used_key,
            model=model,
            call_time=call_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens
        )
        session.add(log_entry)


@app.get("/")
async def root(session_id: str = Cookie(None)):
    # 检查用户是否已登录
    if session_id:
        try:
            with get_db_session() as session:
                query = select(DbSession).where(DbSession.session_id == session_id)
                result = session.execute(query).scalar_one_or_none()
                if result:
                    # 用户已登录，重定向到 admin 页面
                    return RedirectResponse(url="/admin")
        except Exception as e:
            logger.error(f"检查会话时出错: {str(e)}")
            # 出错时继续显示登录页面
    
    # 用户未登录或会话无效，显示登录页面
    return FileResponse("static/index.html")


@app.get("/admin")
async def admin_page(authorized: bool = Depends(require_auth)):
    response = FileResponse("static/admin.html")
    # 添加缓存控制头
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.post("/login")
async def login(request: Request):
    data = await request.json()
    username = data.get("username")
    password = data.get("password")
    
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session_id = await create_session(username)
        response = JSONResponse({"message": "登录成功"})
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            max_age=86400,  # 24 hours
            samesite="lax"
        )
        return response
    else:
        raise HTTPException(status_code=401, detail="用户名或密码错误")


@app.get("/logout")
async def logout(session_id: str = Cookie(None)):
    # 清除数据库中的会话
    if session_id:
        with get_db_session() as session:
            delete_query = DbSession.__table__.delete().where(DbSession.session_id == session_id)
            session.execute(delete_query)
    
    # 创建重定向响应
    response = RedirectResponse(url="/", status_code=303)  # 使用 303 See Other 状态码
    
    # 添加缓存控制头，防止浏览器缓存
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    
    # 正确地删除 cookie
    response.delete_cookie(
        key="session_id",
        path="/",  # 确保删除所有路径下的 cookie
        secure=False,  # 根据您的环境设置
        httponly=True,
        samesite="lax"
    )
    
    return response


@app.post("/import_keys")
async def import_keys(request: Request, authorized: bool = Depends(require_auth)):
    data = await request.json()
    keys_text = data.get("keys", "")
    keys = [k.strip() for k in keys_text.splitlines() if k.strip()]
    if not keys:
        raise HTTPException(status_code=400, detail="未提供有效的api-key")
    
    tasks = []
    # 检查重复的键
    duplicate_keys = []
    with get_db_session() as session:
        for key in keys:
            query = select(ApiKey).where(ApiKey.key == key)
            existing_key = session.execute(query).scalar_one_or_none()
            if existing_key:
                duplicate_keys.append(key)
    
    # 准备任务
    for key in keys:
        if key in duplicate_keys:
            tasks.append(asyncio.sleep(0, result=("duplicate", key)))
        else:
            tasks.append(validate_key_async(key))
    
    results = await asyncio.gather(*tasks)
    imported_count = 0
    duplicate_count = len(duplicate_keys)
    invalid_count = 0
    
    for idx, result in enumerate(results):
        if result[0] == "duplicate":
            continue
        else:
            valid, balance = result
            if valid and float(balance) > 0:
                insert_api_key(keys[idx], balance)
                imported_count += 1
            else:
                invalid_count += 1
    
    return JSONResponse(
        {
            "message": f"导入成功 {imported_count} 个，有重复 {duplicate_count} 个，无效 {invalid_count} 个"
        }
    )


@app.post("/refresh")
async def refresh_keys(authorized: bool = Depends(require_auth)):
    with get_db_session() as session:
        query = select(ApiKey)
        all_keys = [key.key for key in session.execute(query).scalars().all()]

    # Create tasks for parallel validation
    tasks = [validate_key_async(key) for key in all_keys]
    results = await asyncio.gather(*tasks)

    removed = 0
    for key, (valid, balance) in zip(all_keys, results):
        if valid and float(balance) > 0:
            with get_db_session() as session:
                query = select(ApiKey).where(ApiKey.key == key)
                api_key = session.execute(query).scalar_one_or_none()
                if api_key:
                    api_key.balance = balance
        else:
            with get_db_session() as session:
                delete_query = ApiKey.__table__.delete().where(ApiKey.key == key)
                session.execute(delete_query)
                removed += 1

    return JSONResponse(
        {"message": f"刷新完成，共移除 {removed} 个余额用尽或无效的key"}
    )


async def stream_response(api_key: str, req_json: dict, headers: dict):
    """Stream the chat completion response from the API."""
    completion_tokens = 0
    prompt_tokens = 0
    total_tokens = 0
    call_time_stamp = time.time()
    model = req_json.get("model", "unknown")
    
    async with aiohttp.ClientSession() as session:
        try:
            logger.info(f"开始流式请求: 模型={model}")
            async with session.post(
                f"{BASE_URL}/v1/chat/completions",
                headers=headers,
                json=req_json,
                timeout=300,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"流式请求失败: {resp.status} - {error_text}")
                    yield f"data: {json.dumps({'error': error_text})}\n\n".encode('utf-8')
                    return
                
                # 逐行读取响应
                async for line in resp.content:
                    if line:
                        # 确保每行都是正确的 SSE 格式
                        try:
                            line = line.decode('utf-8').strip()
                            if line:  # 忽略空行
                                if not line.startswith('data: '):
                                    line = f"data: {line}"
                                yield f"{line}\n\n".encode('utf-8')
                                
                                # 尝试解析完成的 tokens
                                if line.startswith('data: '):
                                    try:
                                        data = json.loads(line[6:])
                                        if 'usage' in data:
                                            usage = data['usage']
                                            prompt_tokens = usage.get('prompt_tokens', 0)
                                            completion_tokens = usage.get('completion_tokens', 0)
                                            total_tokens = usage.get('total_tokens', 0)
                                    except json.JSONDecodeError:
                                        pass
                        except Exception as e:
                            logger.error(f"处理响应行时出错: {str(e)}")
                            continue
                
                # 发送结束标记
                yield "data: [DONE]\n\n".encode('utf-8')
                
                # 记录使用情况
                log_completion(
                    api_key,
                    model,
                    call_time_stamp,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                )
                
                logger.info(f"流式请求完成: 模型={model}, 输入tokens={prompt_tokens}, 输出tokens={completion_tokens}, 总tokens={total_tokens}")
                
        except asyncio.TimeoutError:
            logger.error("流式请求超时")
            yield f"data: {json.dumps({'error': '请求超时'})}\n\n".encode('utf-8')
        except Exception as e:
            logger.error(f"流式请求出错: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n".encode('utf-8')

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
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
    
    # 获取并验证可用的密钥
    with get_db_session() as session:
        query = select(ApiKey.key)
        keys = session.execute(query).scalars().all()
    
    if not keys:
        raise HTTPException(status_code=500, detail="没有可用的API密钥")
    
    # 随机选择一个密钥
    selected_key = random.choice(keys)
    
    # 更新使用计数
    with get_db_session() as session:
        query = select(ApiKey).where(ApiKey.key == selected_key)
        api_key = session.execute(query).scalar_one_or_none()
        if api_key:
            api_key.usage_count += 1
    
    # 解析请求体
    try:
        body = await request.json()
    except json.JSONDecodeError:
        logger.error("无法解析JSON请求体")
        raise HTTPException(status_code=400, detail="无效的JSON请求")
    
    # 获取模型名称，确保它不为空
    model = body.get("model", "")
    if not model:
        model = "gpt-4o"  # 设置默认模型名称
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
            async with aiohttp.ClientSession() as session:
                # 设置更合理的超时
                timeout = aiohttp.ClientTimeout(total=300, sock_connect=10, sock_read=300)
                
                async with session.post(
                    f"{BASE_URL}/v1/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=timeout,
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
                    total_tokens = prompt_tokens + completion_tokens
                    
                    # 记录使用情况
                    log_completion(
                        selected_key,
                        model,
                        call_time_stamp,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                    )
                    
                    logger.info(f"请求完成: 模型={model}, 输入tokens={prompt_tokens}, 输出tokens={completion_tokens}, 总tokens={total_tokens}")
                    
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
    model = "unknown"
    call_time_stamp = time.time()
    prompt_tokens = 0
    completion_tokens = 0
    
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
                    
                    # 尝试解析最后一个块以获取token信息
                    try:
                        chunk_str = chunk.decode("utf-8")
                        if chunk_str.startswith("data: ") and "usage" in chunk_str:
                            data = json.loads(chunk_str[6:])
                            usage = data.get("usage", {})
                            if usage:
                                prompt_tokens = usage.get("prompt_tokens", 0)
                                completion_tokens = usage.get("completion_tokens", 0)
                                total_tokens = usage.get("total_tokens", 0)
                    except:
                        pass
                
                # 记录使用情况
                log_completion(
                    api_key,
                    model,
                    call_time_stamp,
                    prompt_tokens,
                    completion_tokens,
                    prompt_tokens + completion_tokens,
                )
                
        except Exception as e:
            error_json = json.dumps({"error": {"message": str(e), "type": "server_error"}})
            yield f"data: {error_json}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

@app.post("/v1/embeddings")
async def embeddings(request: Request):
    if API_KEY is not None:
        request_api_key = request.headers.get("Authorization")
        if request_api_key != f"Bearer {API_KEY}":
            raise HTTPException(status_code=403, detail="无效的API_KEY")
    
    with get_db_session() as session:
        query = select(ApiKey.key)
        keys = session.execute(query).scalars().all()
    
    if not keys:
        raise HTTPException(status_code=500, detail="没有可用的api-key")
    
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
        with get_db_session() as session:
            query = select(ApiKey.key)
            keys = session.execute(query).scalars().all()
        
        if not keys:
            logger.error("没有可用的API密钥用于models请求")
            raise HTTPException(status_code=500, detail="没有可用的api-key")
        
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
async def stats(authorized: bool = Depends(require_auth)):
    try:
        key_count = 0
        total_balance = 0
        total_calls = 0
        total_tokens = 0
        
        async for session in get_async_db_session():
            # 获取密钥数量和总余额
            count_query = select(func.count(), func.sum(ApiKey.balance)).select_from(ApiKey)
            result = await session.execute(count_query)
            key_count, total_balance = result.one_or_none() or (0, 0)
            
            # 获取总调用次数（logs表中的记录数）
            calls_query = select(func.count()).select_from(Log)
            result = await session.execute(calls_query)
            total_calls = result.scalar() or 0
            
            # 获取总tokens消耗（logs表中total_tokens字段的总和）
            tokens_query = select(func.sum(Log.total_tokens)).select_from(Log)
            result = await session.execute(tokens_query)
            total_tokens = result.scalar() or 0
        
        return JSONResponse({
            "key_count": key_count, 
            "total_balance": total_balance or 0,
            "total_calls": total_calls,
            "total_tokens": total_tokens
        })
    except Exception as e:
        print(f"获取统计信息时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@app.get("/export_keys")
async def export_keys(authorized: bool = Depends(require_auth)):
    all_keys = []
    async for session in get_async_db_session():
        query = select(ApiKey.key)
        result = await session.execute(query)
        all_keys = result.scalars().all()
    
    keys = "\n".join(all_keys)
    headers = {"Content-Disposition": "attachment; filename=keys.txt"}
    return Response(content=keys, media_type="text/plain", headers=headers)


@app.get("/logs")
async def get_logs(page: int = 1, authorized: bool = Depends(require_auth)):
    page_size = 10
    offset = (page - 1) * page_size
    
    try:
        total = 0
        formatted_logs = []
        
        async for session in get_async_db_session():
            # 获取总记录数
            count_query = select(func.count()).select_from(Log)
            result = await session.execute(count_query)
            total = result.scalar() or 0
            
            # 获取分页数据
            query = (
                select(Log.used_key, Log.model, Log.call_time, Log.input_tokens, Log.output_tokens, Log.total_tokens)
                .order_by(desc(Log.call_time))
                .limit(page_size)
                .offset(offset)
            )
            result = await session.execute(query)
            
            # 格式化日志数据
            for log in result:
                formatted_logs.append({
                    "api_key": log[0],
                    "model": log[1] or "未知",  # 如果 model 为 None 或空字符串，则显示"未知"
                    "call_time": log[2],
                    "input_tokens": log[3],
                    "output_tokens": log[4],
                    "total_tokens": log[5]
                })
        
        return JSONResponse({
            "logs": formatted_logs,
            "total": total,
            "page": page,
            "page_size": page_size
        })
    except Exception as e:
        print(f"获取日志时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取日志失败: {str(e)}")


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
async def get_keys(
    page: int = 1, sort_field: str = "add_time", sort_order: str = "desc", 
    authorized: bool = Depends(require_auth)
):
    allowed_fields = ["add_time", "balance", "usage_count"]
    allowed_orders = ["asc", "desc"]

    if sort_field not in allowed_fields:
        sort_field = "add_time"
    if sort_order not in allowed_orders:
        sort_order = "desc"

    page_size = 10
    offset = (page - 1) * page_size

    total = 0
    key_list = []
    
    async for session in get_async_db_session():
        # 获取总记录数
        count_query = select(func.count()).select_from(ApiKey)
        result = await session.execute(count_query)
        total = result.scalar() or 0
        
        # 构建排序条件
        if sort_order == "asc":
            order_by_clause = asc(getattr(ApiKey, sort_field))
        else:
            order_by_clause = desc(getattr(ApiKey, sort_field))
        
        # 获取分页数据
        query = (
            select(ApiKey)
            .order_by(order_by_clause)
            .limit(page_size)
            .offset(offset)
        )
        result = await session.execute(query)
        
        # 格式化键数据
        key_list = [
            {
                "key": key.key,
                "add_time": key.add_time,
                "balance": key.balance,
                "usage_count": key.usage_count
            }
            for key in result.scalars().all()
        ]

    return JSONResponse(
        {"keys": key_list, "total": total, "page": page, "page_size": page_size}
    )

@app.post("/api/refresh_key")
async def refresh_single_key(request: Request, authorized: bool = Depends(require_auth)):
    data = await request.json()
    key = data.get("key")
    
    if not key:
        raise HTTPException(status_code=400, detail="未提供API密钥")
    
    # 验证密钥
    valid, balance = await validate_key_async(key)
    
    if valid and float(balance) > 0:
        async for session in get_async_db_session():
            query = select(ApiKey).where(ApiKey.key == key)
            result = await session.execute(query)
            api_key = result.scalar_one_or_none()
            if api_key:
                api_key.balance = balance
        return JSONResponse({"message": "密钥刷新成功", "balance": balance})
    else:
        async for session in get_async_db_session():
            delete_query = ApiKey.__table__.delete().where(ApiKey.key == key)
            await session.execute(delete_query)
        raise HTTPException(status_code=400, detail="密钥无效或余额为零，已从系统中移除")

@app.get("/api/key_info")
async def get_key_info(key: str, authorized: bool = Depends(require_auth)):
    result_key = None
    async for session in get_async_db_session():
        query = select(ApiKey).where(ApiKey.key == key)
        result = await session.execute(query)
        result_key = result.scalar_one_or_none()
    
    if not result_key:
        raise HTTPException(status_code=404, detail="密钥不存在")
    
    return JSONResponse({
        "key": result_key.key,
        "balance": result_key.balance,
        "usage_count": result_key.usage_count,
        "add_time": result_key.add_time
    })

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
    """健康检查端点，用于监控应用状态"""
    app_start_time = time.time()
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": app.version,
        "uptime": round(time.time() - app.state.startup_timestamp, 2) if hasattr(app.state, "startup_timestamp") else None,
    }
    
    # 检查数据库状态 - 使用同步检查，因为这是专门为健康检查设计的函数
    db_health = check_db_health()
    health_data["database"] = db_health
    
    # 如果数据库不健康，整体状态就不健康
    if db_health["status"] != "healthy":
        health_data["status"] = "degraded"
    
    # 计算API响应时间
    health_data["response_time_ms"] = round((time.time() - app_start_time) * 1000, 2)
    
    return JSONResponse(content=health_data)


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
            
        return JSONResponse(content=metrics_data)
    
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
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7898)

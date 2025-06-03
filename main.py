from fastapi import FastAPI, Request, HTTPException, Depends, Response, Cookie, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from config import API_KEY, ADMIN_USERNAME, ADMIN_PASSWORD, AUTO_REFRESH_INTERVAL
import json
import random
import time
import asyncio
import aiohttp
import uuid
from uvicorn.config import LOGGING_CONFIG
from typing import Optional, List, Dict, Any, Tuple
import logging
import aiosqlite

# Import our new modules
from db import AsyncDBPool, DB_PATH
from models import (
    LoginRequest, APIKeyImport, APIKeyInfo, APIKeyResponse, LogEntry, 
    LogsResponse, StatsResponse, MessageResponse, ErrorResponse, ChatCompletionRequest,
    APIKeyRefresh
)
from utils import (
    validate_key_async, generate_session_id, invalidate_stats_cache, 
    make_api_request, stream_response, BASE_URL
)

LOGGING_CONFIG["formatters"]["default"]["fmt"] = (
    "%(asctime)s - %(levelprefix)s %(message)s"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global _monitor_task, _db_maintenance_task

    # 启动时的初始化
    logger.info("应用启动，开始初始化...")

    # Initialize the database
    db = await AsyncDBPool.get_instance()
    await db.initialize()

    # 启动自动刷新密钥的后台任务
    await start_auto_refresh_task()

    # 启动自动刷新任务监控器
    _monitor_task = asyncio.create_task(monitor_auto_refresh_task())

    # 启动数据库维护任务
    _db_maintenance_task = asyncio.create_task(db_maintenance_task())

    logger.info("应用初始化完成")

    yield  # 应用运行期间

    # 关闭时的清理工作
    logger.info("开始应用关闭清理...")

    # 标记自动刷新任务停止
    AUTO_REFRESH_TASK_STATUS["running"] = False

    # 取消所有后台任务
    tasks_to_cancel = []
    if _auto_refresh_task and not _auto_refresh_task.done():
        tasks_to_cancel.append(_auto_refresh_task)
    if _monitor_task and not _monitor_task.done():
        tasks_to_cancel.append(_monitor_task)
    if _db_maintenance_task and not _db_maintenance_task.done():
        tasks_to_cancel.append(_db_maintenance_task)

    for task in tasks_to_cancel:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"取消任务时出错: {str(e)}")

    # 清理utils模块中的全局session
    try:
        from utils import _session, _connector
        if _session and not _session.closed:
            await _session.close()
        if _connector:
            await _connector.close()
        logger.info("网络连接已清理")
    except Exception as e:
        logger.error(f"清理网络连接时出错: {str(e)}")

    logger.info("应用关闭清理完成")

app = FastAPI(
    title="SiliconFlow API",
    description="A proxy server for the Silicon Flow API with key management",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("siliconflow")

# 自动刷新任务状态跟踪
AUTO_REFRESH_TASK_STATUS = {
    "running": False,
    "last_run_time": 0,
    "last_error": None,
    "run_count": 0,
    "error_count": 0,
    "total_keys_processed": 0,
    "total_keys_updated": 0,
    "total_keys_removed": 0,
    "average_processing_time": 0,
    "last_batch_stats": {}
}

# 任务实例管理
_auto_refresh_task = None
_monitor_task = None
_db_maintenance_task = None
_task_lock = asyncio.Lock()

# 重复的 lifespan 函数已删除

# 自动刷新密钥的定时任务
async def auto_refresh_keys_task():
    """定时自动刷新所有API密钥的余额"""
    global _auto_refresh_task

    # 如果设置为0，则禁用自动刷新
    if AUTO_REFRESH_INTERVAL <= 0:
        logger.info("自动刷新密钥功能已禁用")
        return

    logger.info(f"启动自动刷新任务，间隔：{AUTO_REFRESH_INTERVAL}秒")

    try:
        # 更新任务状态
        AUTO_REFRESH_TASK_STATUS["running"] = True
        AUTO_REFRESH_TASK_STATUS["last_error"] = None

        while True:
            try:
                # 每隔配置的时间执行一次刷新
                await asyncio.sleep(AUTO_REFRESH_INTERVAL)

                logger.info("开始自动刷新API密钥余额...")
                batch_start_time = time.time()

                # 更新运行状态
                AUTO_REFRESH_TASK_STATUS["run_count"] += 1
                AUTO_REFRESH_TASK_STATUS["last_run_time"] = time.time()

                # 获取所有密钥
                db = await AsyncDBPool.get_instance()
                all_keys = await db.get_key_list()

                if not all_keys:
                    logger.info("没有发现API密钥，跳过刷新")
                    continue

                AUTO_REFRESH_TASK_STATUS["total_keys_processed"] += len(all_keys)

                # 使用并发处理密钥验证，但限制并发数避免过载
                removed = 0
                updated = 0
                error_count = 0

                # 分批处理密钥，每批最多10个并发
                batch_size = 10
                for i in range(0, len(all_keys), batch_size):
                    batch_keys = all_keys[i:i + batch_size]

                    # 创建并发验证任务
                    tasks = []
                    for key in batch_keys:
                        tasks.append(validate_and_update_key(key, db))

                    # 等待当前批次完成
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                    # 处理结果
                    for j, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            error_count += 1
                            logger.error(f"刷新密钥 {batch_keys[j][:8]}*** 时出错: {str(result)}")
                        else:
                            action, success = result
                            if success:
                                if action == "updated":
                                    updated += 1
                                elif action == "removed":
                                    removed += 1
                            else:
                                error_count += 1

                    # 在批次之间稍作延迟，避免API限制
                    if i + batch_size < len(all_keys):
                        await asyncio.sleep(0.5)

                    # 如果错误过多，记录警告但继续处理
                    if error_count > len(batch_keys) * 0.8:  # 当前批次错误率超过80%
                        logger.warning(f"当前批次错误率过高: {error_count}/{len(batch_keys)}")
                        await asyncio.sleep(2)  # 稍作延迟

                # 更新统计信息
                AUTO_REFRESH_TASK_STATUS["total_keys_updated"] += updated
                AUTO_REFRESH_TASK_STATUS["total_keys_removed"] += removed

                # 计算处理时间
                processing_time = time.time() - batch_start_time
                if AUTO_REFRESH_TASK_STATUS["run_count"] == 1:
                    AUTO_REFRESH_TASK_STATUS["average_processing_time"] = processing_time
                else:
                    # 使用移动平均
                    current_avg = AUTO_REFRESH_TASK_STATUS["average_processing_time"]
                    AUTO_REFRESH_TASK_STATUS["average_processing_time"] = (current_avg * 0.8) + (processing_time * 0.2)

                # 记录本次批处理统计
                AUTO_REFRESH_TASK_STATUS["last_batch_stats"] = {
                    "processed": len(all_keys),
                    "updated": updated,
                    "removed": removed,
                    "errors": error_count,
                    "processing_time": processing_time,
                    "timestamp": time.time()
                }

                # 使统计缓存失效
                invalidate_stats_cache()

                logger.info(f"自动刷新完成: 已更新 {updated} 个密钥, 移除 {removed} 个无效密钥, 错误 {error_count} 个, 耗时 {processing_time:.2f} 秒")

            except asyncio.CancelledError:
                logger.info("自动刷新任务被取消")
                AUTO_REFRESH_TASK_STATUS["running"] = False
                break
            except Exception as e:
                AUTO_REFRESH_TASK_STATUS["error_count"] += 1
                AUTO_REFRESH_TASK_STATUS["last_error"] = str(e)
                logger.error(f"自动刷新密钥任务出错: {str(e)}")

                # 如果错误次数过多，增加等待时间
                error_delay = min(300, 60 * AUTO_REFRESH_TASK_STATUS["error_count"])  # 最多等待5分钟
                logger.info(f"由于错误，将等待 {error_delay} 秒后重试")
                await asyncio.sleep(error_delay)

    except Exception as e:
        logger.error(f"自动刷新任务异常退出: {str(e)}")
        AUTO_REFRESH_TASK_STATUS["last_error"] = str(e)
    finally:
        AUTO_REFRESH_TASK_STATUS["running"] = False
        _auto_refresh_task = None

async def validate_and_update_key(key: str, db) -> tuple:
    """验证并更新单个密钥的辅助函数"""
    try:
        valid, balance = await validate_key_async(key)
        if valid and float(balance) > 0:
            await db.update_key_balance(key, balance)
            return ("updated", True)
        else:
            await db.delete_key(key)
            return ("removed", True)
    except Exception as e:
        logger.error(f"验证密钥 {key[:8]}*** 失败: {str(e)}")
        return ("error", False)

# 监控自动刷新任务的状态
async def monitor_auto_refresh_task():
    """监控自动刷新任务，如果任务停止则重新启动"""
    global _auto_refresh_task, _monitor_task

    logger.info("启动自动刷新任务监控器")

    while True:
        try:
            await asyncio.sleep(300)  # 每5分钟检查一次

            # 检查自动刷新功能是否启用
            if AUTO_REFRESH_INTERVAL <= 0:
                continue

            current_time = time.time()
            last_run_time = AUTO_REFRESH_TASK_STATUS["last_run_time"]

            # 检查任务是否需要重启
            should_restart = False

            # 如果任务没有运行
            if not AUTO_REFRESH_TASK_STATUS["running"]:
                should_restart = True
                reason = "任务未运行"
            # 如果任务实例不存在或已完成
            elif _auto_refresh_task is None or _auto_refresh_task.done():
                should_restart = True
                reason = "任务实例已完成"
            # 如果上次运行时间超过了2倍的刷新间隔
            elif last_run_time > 0 and current_time - last_run_time > AUTO_REFRESH_INTERVAL * 2:
                should_restart = True
                reason = f"任务超时未运行 ({current_time - last_run_time:.0f}秒)"

            if should_restart:
                logger.warning(f"检测到自动刷新任务需要重启: {reason}")
                await start_auto_refresh_task()

        except Exception as e:
            logger.error(f"自动刷新任务监控器出错: {str(e)}")
            await asyncio.sleep(60)  # 出错后等待1分钟

async def start_auto_refresh_task():
    """安全地启动自动刷新任务"""
    global _auto_refresh_task

    async with _task_lock:
        # 如果已有任务在运行，先取消
        if _auto_refresh_task and not _auto_refresh_task.done():
            _auto_refresh_task.cancel()
            try:
                await _auto_refresh_task
            except asyncio.CancelledError:
                pass

        # 重置状态
        AUTO_REFRESH_TASK_STATUS["running"] = False
        AUTO_REFRESH_TASK_STATUS["error_count"] = 0
        AUTO_REFRESH_TASK_STATUS["last_error"] = None

        # 启动新任务
        _auto_refresh_task = asyncio.create_task(auto_refresh_keys_task())
        logger.info("自动刷新任务已启动")

# 旧的事件处理器已被 lifespan 处理器替代

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
    """Create a new session for the user."""
    session_id = generate_session_id()
    db = await AsyncDBPool.get_instance()
    await db.create_session(session_id, username)
    return session_id


async def validate_session(session_id: str = Cookie(None)) -> bool:
    """Validate a user session."""
    if not session_id:
        return False
    
    db = await AsyncDBPool.get_instance()
    
    # Clean up old sessions
    await db.cleanup_old_sessions()
    
    # Check if session exists
    session = await db.get_session(session_id)
    return bool(session)


async def require_auth(session_id: str = Cookie(None)):
    """Require authentication to access a resource."""
    if not session_id:
        raise HTTPException(status_code=401, detail="未授权访问，请先登录")
    
    db = await AsyncDBPool.get_instance()
    
    # Clean up old sessions
    try:
        await db.cleanup_old_sessions()
    except Exception as e:
        print(f"清理过期会话时出错: {str(e)}")
        # 继续执行，不要因为清理失败而阻止用户访问
    
    # Check if session exists
    try:
        session = await db.get_session(session_id)
        if not session:
            raise HTTPException(status_code=401, detail="会话已过期，请重新登录")
    except Exception as e:
        print(f"检查会话时出错: {str(e)}")
        raise HTTPException(status_code=401, detail="验证会话时出错，请重新登录")
    
    return True


@app.get("/")
async def root(session_id: str = Cookie(None)):
    # 检查用户是否已登录
    if session_id:
        try:
            is_valid = await validate_session(session_id)
            if is_valid:
                # 用户已登录，重定向到 admin 页面
                return RedirectResponse(url="/admin")
        except Exception as e:
            print(f"检查会话时出错: {str(e)}")
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
async def login(login_data: LoginRequest):
    if login_data.username == ADMIN_USERNAME and login_data.password == ADMIN_PASSWORD:
        session_id = await create_session(login_data.username)
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
        db = await AsyncDBPool.get_instance()
        await db.delete_session(session_id)
    
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
async def import_keys(key_data: APIKeyImport, authorized: bool = Depends(require_auth)):
    keys_text = key_data.keys
    keys = [k.strip() for k in keys_text.splitlines() if k.strip()]
    if not keys:
        raise HTTPException(status_code=400, detail="未提供有效的api-key")
    
    # Get database instance
    db = await AsyncDBPool.get_instance()
    
    # Check for duplicate keys
    duplicate_keys = []
    all_keys = await db.get_key_list()
    for key in keys:
        if key in all_keys:
            duplicate_keys.append(key)
    
    # Prepare tasks for validation
    tasks = []
    for key in keys:
        if key in duplicate_keys:
            # Skip validation for duplicate keys - create a coroutine that returns the result
            async def create_duplicate_result(captured_key):
                async def duplicate_result():
                    return ("duplicate", captured_key)
                return duplicate_result()
            tasks.append(create_duplicate_result(key))
        else:
            tasks.append(validate_key_async(key))
    
    results = await asyncio.gather(*tasks)
    
    # Process results
    imported_count = 0
    duplicate_count = len(duplicate_keys)
    invalid_count = 0

    for idx, result in enumerate(results):
        current_key = keys[idx]
        if isinstance(result, tuple) and result[0] == "duplicate":
            continue
        else:
            valid, balance = result
            if valid and float(balance) > 0:
                await db.insert_api_key(current_key, balance)
                imported_count += 1
            else:
                invalid_count += 1
    
    # Invalidate stats cache
    invalidate_stats_cache()
    
    return JSONResponse(
        {
            "message": f"导入成功 {imported_count} 个，有重复 {duplicate_count} 个，无效 {invalid_count} 个"
        }
    )


@app.post("/refresh")
async def refresh_keys(authorized: bool = Depends(require_auth)):
    # Get all keys
    db = await AsyncDBPool.get_instance()
    all_keys = await db.get_key_list()

    # Create tasks for parallel validation
    tasks = [validate_key_async(key) for key in all_keys]
    results = await asyncio.gather(*tasks)

    # Update database with results
    removed = 0
    for key, (valid, balance) in zip(all_keys, results):
        if valid and float(balance) > 0:
            await db.update_key_balance(key, balance)
        else:
            await db.delete_key(key)
            removed += 1

    # Invalidate the stats cache
    invalidate_stats_cache()

    return JSONResponse(
        {"message": f"刷新完成，共移除 {removed} 个余额用尽或无效的key"}
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, background_tasks: BackgroundTasks):
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
    db = await AsyncDBPool.get_instance()
    keys = await db.get_best_keys(limit=10)
    
    if not keys:
        raise HTTPException(status_code=500, detail="没有可用的API密钥")
    
    # 随机选择前10个余额较高的密钥中的一个
    selected_key = random.choice(keys)
    
    # 更新使用计数
    background_tasks.add_task(db.increment_key_usage, selected_key)
    
    # 解析请求体
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="无效的JSON请求体")
    
    # 获取模型名称，确保它不为空
    model = body.get("model", "")
    if not model:
        model = "gpt-4o"  # 设置默认模型名称
    
    # 记录请求时间
    call_time_stamp = time.time()
    
    # 检查是否是流式请求
    is_stream = body.get("stream", False)
    
    # 准备转发请求的头部
    headers = {
        "Authorization": f"Bearer {selected_key}",
        "Content-Type": "application/json",
    }
    
    if is_stream:
        # Process streaming response
        async def stream_with_logging():
            """Wrap streaming to add background task for logging at the end"""
            completion_tokens = 0
            prompt_tokens = 0
            total_tokens = 0
            
            async for chunk in stream_response(selected_key, "/v1/chat/completions", body, headers):
                yield chunk
                
                # Try to extract token info from the chunk
                try:
                    chunk_str = chunk.decode('utf-8')
                    if chunk_str.startswith('data: ') and 'usage' in chunk_str:
                        data = json.loads(chunk_str[6:])
                        usage = data.get('usage', {})
                        if usage:
                            prompt_tokens = usage.get('prompt_tokens', 0)
                            completion_tokens = usage.get('completion_tokens', 0)
                            total_tokens = usage.get('total_tokens', 0)
                except:
                    pass
            
            # Once the stream is finished, log the completion
            background_tasks.add_task(
                db.log_completion,
                selected_key,
                model,
                call_time_stamp,
                prompt_tokens,
                completion_tokens,
                total_tokens
            )
        
        return StreamingResponse(
            stream_with_logging(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        # Process non-streaming response
        try:
            status, response_data = await make_api_request(
                "/v1/chat/completions", 
                selected_key, 
                method="POST", 
                data=body,
                timeout=120
            )
            
            if status != 200:
                raise HTTPException(status_code=status, detail=response_data.get("error", "API request failed"))
            
            # 计算 token 使用量
            usage = response_data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
            # 记录使用情况
            background_tasks.add_task(
                db.log_completion,
                selected_key,
                model,
                call_time_stamp,
                prompt_tokens,
                completion_tokens,
                total_tokens
            )
            
            return JSONResponse(response_data)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"请求处理失败: {str(e)}")


@app.post("/v1/embeddings")
async def embeddings(request: Request, background_tasks: BackgroundTasks):
    # Check API_KEY if configured
    if API_KEY is not None:
        request_api_key = request.headers.get("Authorization")
        if request_api_key != f"Bearer {API_KEY}":
            raise HTTPException(status_code=403, detail="无效的API_KEY")
    
    # Get a key from the pool
    db = await AsyncDBPool.get_instance()
    keys = await db.get_key_list()
    
    if not keys:
        raise HTTPException(status_code=500, detail="没有可用的api-key")
    
    # Choose a random key
    selected = random.choice(keys)
    
    # Update usage count
    background_tasks.add_task(db.increment_key_usage, selected)
    
    # Forward the request
    forward_headers = dict(request.headers)
    forward_headers["Authorization"] = f"Bearer {selected}"
    
    try:
        from utils import get_session
        session = await get_session()
        async with session.post(
            f"{BASE_URL}/v1/embeddings",
            headers=forward_headers,
            data=await request.body(),
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            data = await resp.json()
            return JSONResponse(content=data, status_code=resp.status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"请求转发失败: {str(e)}")


@app.get("/v1/models")
async def list_models(request: Request):
    # Get a key from the pool
    db = await AsyncDBPool.get_instance()
    keys = await db.get_key_list()
    
    if not keys:
        raise HTTPException(status_code=500, detail="没有可用的api-key")
    
    # Choose a random key
    selected = random.choice(keys)
    
    # Forward the request
    forward_headers = dict(request.headers)
    forward_headers["Authorization"] = f"Bearer {selected}"
    
    try:
        from utils import get_session
        session = await get_session()
        async with session.get(
            f"{BASE_URL}/v1/models", headers=forward_headers, timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            data = await resp.json()
            return JSONResponse(content=data, status_code=resp.status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"请求转发失败: {str(e)}")


@app.get("/api/stats/overview")
async def stats_overview(authorized: bool = Depends(require_auth)):
    """Get system statistics."""
    db = await AsyncDBPool.get_instance()
    stats_data = await db.get_stats()
    
    return JSONResponse(StatsResponse(**stats_data).model_dump())


@app.get("/export_keys")
async def export_keys(authorized: bool = Depends(require_auth)):
    """Export all API keys as a text file."""
    db = await AsyncDBPool.get_instance()
    keys = await db.get_key_list()
    
    keys_text = "\n".join(keys)
    headers = {"Content-Disposition": "attachment; filename=keys.txt"}
    
    return Response(content=keys_text, media_type="text/plain", headers=headers)


@app.get("/logs")
async def get_logs(
    page: int = 1, 
    model: str = None, 
    authorized: bool = Depends(require_auth)
):
    """Get paginated logs with optional model filter."""
    db = await AsyncDBPool.get_instance()
    logs, total = await db.get_logs(page=page, page_size=10, model=model)
    
    # Format logs for response
    formatted_logs = []
    for log in logs:
        formatted_logs.append(LogEntry(
            api_key=log['used_key'],
            model=log['model'] or "未知",
            call_time=log['call_time'],
            input_tokens=log['input_tokens'],
            output_tokens=log['output_tokens'],
            total_tokens=log['total_tokens']
        ))
    
    return JSONResponse(LogsResponse(
        logs=formatted_logs,
        total=total,
        page=page,
        page_size=10
    ).model_dump())


@app.post("/clear_logs")
async def clear_logs(authorized: bool = Depends(require_auth)):
    """Clear all logs from the database."""
    db = await AsyncDBPool.get_instance()
    
    try:
        await db.clear_logs()
        
        # Invalidate stats cache
        invalidate_stats_cache()
        
        return JSONResponse({"message": "日志已清空"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空日志失败: {str(e)}")


@app.get("/api/keys")
async def get_keys(
    page: int = 1, 
    sort_field: str = "add_time", 
    sort_order: str = "desc", 
    authorized: bool = Depends(require_auth)
):
    """Get paginated API keys with sorting options."""
    allowed_fields = ["add_time", "balance", "usage_count"]
    allowed_orders = ["asc", "desc"]

    if sort_field not in allowed_fields:
        sort_field = "add_time"
    if sort_order not in allowed_orders:
        sort_order = "desc"

    page_size = 10
    offset = (page - 1) * page_size

    db = await AsyncDBPool.get_instance()
    
    # Custom query for sorted, paginated keys
    query = f"SELECT key, add_time, balance, usage_count FROM api_keys ORDER BY {sort_field} {sort_order} LIMIT ? OFFSET ?"
    rows = await db.execute(query, (page_size, offset), fetch_all=True)
    
    # Get total count
    count_row = await db.execute("SELECT COUNT(*) as count FROM api_keys", fetch_one=True)
    total = count_row['count'] if count_row else 0
    
    # Format keys
    key_list = []
    for row in rows:
        key_list.append(APIKeyInfo(
            key=row['key'],
            add_time=row['add_time'],
            balance=row['balance'],
            usage_count=row['usage_count']
        ))

    return JSONResponse(APIKeyResponse(
        keys=key_list,
        total=total,
        page=page,
        page_size=page_size
    ).model_dump())

@app.post("/api/refresh_key")
async def refresh_single_key(key_data: APIKeyRefresh, authorized: bool = Depends(require_auth)):
    """Refresh a single API key's balance."""
    key = key_data.key
    
    # Validate key
    valid, balance = await validate_key_async(key)
    
    db = await AsyncDBPool.get_instance()
    
    if valid and float(balance) > 0:
        # Update key balance
        await db.update_key_balance(key, balance)
        
        # Invalidate stats cache
        invalidate_stats_cache()
        
        return JSONResponse({"message": "密钥刷新成功", "balance": balance})
    else:
        # Delete invalid key
        await db.delete_key(key)
        
        # Invalidate stats cache
        invalidate_stats_cache()
        
        raise HTTPException(status_code=400, detail="密钥无效或余额为零，已从系统中移除")

@app.get("/api/key_info")
async def get_key_info(key: str, authorized: bool = Depends(require_auth)):
    """Get information about a specific API key."""
    db = await AsyncDBPool.get_instance()
    
    # Custom query to get key info
    row = await db.execute(
        "SELECT key, balance, usage_count, add_time FROM api_keys WHERE key = ?", 
        (key,),
        fetch_one=True
    )
    
    if not row:
        raise HTTPException(status_code=404, detail="密钥不存在")
    
    return JSONResponse({
        "key": row['key'],
        "balance": row['balance'],
        "usage_count": row['usage_count'],
        "add_time": row['add_time']
    })

# Add a general exception handler for uncaught exceptions
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    # 记录详细的错误信息
    error_msg = f"路径: {request.url.path}, 方法: {request.method}, 错误: {str(exc)}"
    logger.error(f"未处理的异常: {error_msg}")
    
    # 对于API端点，返回JSON错误
    if request.url.path.startswith("/v1/") or request.headers.get("accept") == "application/json":
        return JSONResponse(
            status_code=500,
            content={"detail": "服务器内部错误"}
        )
    # 对于网页，返回500错误页面
    return FileResponse("static/500.html", status_code=500)


@app.get("/keys")
async def keys_page(authorized: bool = Depends(require_auth)):
    """Render the keys management page."""
    response = FileResponse("static/keys.html")
    # 添加缓存控制头
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/stats")
async def stats_page(authorized: bool = Depends(require_auth)):
    """Render the statistics page."""
    response = FileResponse("static/stats.html")
    # 添加缓存控制头
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/api/stats/daily")
async def get_daily_stats(authorized: bool = Depends(require_auth)):
    """Get daily statistics by hour."""
    db = await AsyncDBPool.get_instance()
    
    # 获取今天的开始时间戳（当地时间的0点）
    today = time.time()
    today_start = today - (today % 86400)
    
    # 准备数据结构
    hours = list(range(24))
    calls = [0] * 24
    input_tokens = [0] * 24
    output_tokens = [0] * 24
    
    # 获取今日按小时统计的调用数据
    query = """
        SELECT 
            strftime('%H', datetime(call_time, 'unixepoch', 'localtime')) as hour,
            COUNT(*) as call_count,
            SUM(input_tokens) as input_sum,
            SUM(output_tokens) as output_sum
        FROM logs
        WHERE call_time >= ?
        GROUP BY hour
        ORDER BY hour
    """
    
    rows = await db.execute(query, (today_start,), fetch_all=True)
    
    # 填充数据
    for row in rows:
        hour = int(row['hour'])
        if 0 <= hour < 24:
            calls[hour] = row['call_count']
            input_tokens[hour] = row['input_sum']
            output_tokens[hour] = row['output_sum']
    
    # 获取今日模型使用分布
    model_query = """
        SELECT 
            model,
            SUM(total_tokens) as token_sum
        FROM logs
        WHERE call_time >= ?
        GROUP BY model
        ORDER BY token_sum DESC
        LIMIT 10
    """
    
    model_rows = await db.execute(model_query, (today_start,), fetch_all=True)
    model_labels = []
    model_tokens = []
    
    for row in model_rows:
        if row['model'] and row['token_sum'] > 0:
            model_labels.append(row['model'])
            model_tokens.append(row['token_sum'])
    
    return JSONResponse({
        "labels": [str(h) for h in hours],
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model_labels": model_labels,
        "model_tokens": model_tokens
    })


@app.get("/api/stats/monthly")
async def get_monthly_stats(authorized: bool = Depends(require_auth)):
    """Get monthly statistics by day."""
    db = await AsyncDBPool.get_instance()
    
    # 获取当前月的开始时间戳
    now = time.localtime()
    month_start = time.mktime((now.tm_year, now.tm_mon, 1, 0, 0, 0, 0, 0, 0))
    
    # 获取当月天数
    if now.tm_mon == 12:
        next_month = time.mktime((now.tm_year + 1, 1, 1, 0, 0, 0, 0, 0, 0))
    else:
        next_month = time.mktime((now.tm_year, now.tm_mon + 1, 1, 0, 0, 0, 0, 0, 0))
    
    days_in_month = int((next_month - month_start) / 86400)
    
    # 准备数据结构
    days = list(range(1, days_in_month + 1))
    calls = [0] * days_in_month
    input_tokens = [0] * days_in_month
    output_tokens = [0] * days_in_month
    
    # 获取本月按天统计的调用数据
    query = """
        SELECT 
            strftime('%d', datetime(call_time, 'unixepoch', 'localtime')) as day,
            COUNT(*) as call_count,
            SUM(input_tokens) as input_sum,
            SUM(output_tokens) as output_sum
        FROM logs
        WHERE call_time >= ?
        GROUP BY day
        ORDER BY day
    """
    
    rows = await db.execute(query, (month_start,), fetch_all=True)
    
    # 填充数据
    for row in rows:
        day = int(row['day'])
        if 1 <= day <= days_in_month:
            calls[day-1] = row['call_count']
            input_tokens[day-1] = row['input_sum']
            output_tokens[day-1] = row['output_sum']
    
    # 获取本月模型使用分布
    model_query = """
        SELECT 
            model,
            SUM(total_tokens) as token_sum
        FROM logs
        WHERE call_time >= ?
        GROUP BY model
        ORDER BY token_sum DESC
        LIMIT 10
    """
    
    model_rows = await db.execute(model_query, (month_start,), fetch_all=True)
    model_labels = []
    model_tokens = []
    
    for row in model_rows:
        if row['model'] and row['token_sum'] > 0:
            model_labels.append(row['model'])
            model_tokens.append(row['token_sum'])
    
    return JSONResponse({
        "labels": [str(d) for d in days],
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model_labels": model_labels,
        "model_tokens": model_tokens
    })


@app.get("/health")
async def health_check():
    """健康检查端点，用于监控系统状态"""
    try:
        # 检查数据库连接
        db = await AsyncDBPool.get_instance()
        await db.execute("SELECT 1", fetch_one=True)
        
        # 检查是否有可用的API密钥
        keys = await db.get_key_list()
        key_status = "available" if keys else "unavailable"
        
        # 检查自动刷新任务状态
        auto_refresh_status = {
            "enabled": AUTO_REFRESH_INTERVAL > 0,
            "running": AUTO_REFRESH_TASK_STATUS["running"],
            "last_run_time": AUTO_REFRESH_TASK_STATUS["last_run_time"],
            "run_count": AUTO_REFRESH_TASK_STATUS["run_count"],
            "error_count": AUTO_REFRESH_TASK_STATUS["error_count"],
            "last_error": AUTO_REFRESH_TASK_STATUS["last_error"],
            "performance": {
                "total_keys_processed": AUTO_REFRESH_TASK_STATUS["total_keys_processed"],
                "total_keys_updated": AUTO_REFRESH_TASK_STATUS["total_keys_updated"],
                "total_keys_removed": AUTO_REFRESH_TASK_STATUS["total_keys_removed"],
                "average_processing_time": round(AUTO_REFRESH_TASK_STATUS["average_processing_time"], 2)
            },
            "last_batch": AUTO_REFRESH_TASK_STATUS["last_batch_stats"]
        }
        
        # 计算上次运行时间距现在的时间差
        if auto_refresh_status["last_run_time"] > 0:
            auto_refresh_status["time_since_last_run"] = time.time() - auto_refresh_status["last_run_time"]
            auto_refresh_status["next_run_in"] = max(0, AUTO_REFRESH_INTERVAL - auto_refresh_status["time_since_last_run"])
        else:
            auto_refresh_status["time_since_last_run"] = None
            auto_refresh_status["next_run_in"] = None
        
        # 计算成功率
        if auto_refresh_status["performance"]["total_keys_processed"] > 0:
            success_count = auto_refresh_status["performance"]["total_keys_updated"] + auto_refresh_status["performance"]["total_keys_removed"]
            auto_refresh_status["performance"]["success_rate"] = round(
                (success_count / auto_refresh_status["performance"]["total_keys_processed"]) * 100, 2
            )
        else:
            auto_refresh_status["performance"]["success_rate"] = 0
        
        return JSONResponse({
            "status": "healthy",
            "database": "connected",
            "api_keys": key_status,
            "api_keys_count": len(keys),
            "auto_refresh": auto_refresh_status,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return JSONResponse({
            "status": "unhealthy",
            "error": str(e),
            "auto_refresh": {
                "enabled": AUTO_REFRESH_INTERVAL > 0,
                "running": AUTO_REFRESH_TASK_STATUS["running"],
                "error_count": AUTO_REFRESH_TASK_STATUS["error_count"],
                "last_error": AUTO_REFRESH_TASK_STATUS["last_error"]
            },
            "timestamp": time.time()
        }, status_code=500)


@app.post("/admin/restart_auto_refresh")
async def restart_auto_refresh(authorized: bool = Depends(require_auth)):
    """手动重启自动刷新任务"""
    try:
        if AUTO_REFRESH_INTERVAL <= 0:
            return JSONResponse({
                "message": "自动刷新功能已禁用",
                "success": False
            })

        # 使用安全的重启函数
        await start_auto_refresh_task()

        logger.info("手动重启自动刷新任务")

        return JSONResponse({
            "message": "自动刷新任务已重启",
            "success": True
        })
    except Exception as e:
        logger.error(f"重启自动刷新任务失败: {str(e)}")
        return JSONResponse({
            "message": f"重启失败: {str(e)}",
            "success": False
        }, status_code=500)


async def db_maintenance_task():
    """定期数据库维护任务"""
    logger.info("启动数据库维护任务")
    
    while True:
        try:
            # 每24小时执行一次维护
            await asyncio.sleep(24 * 60 * 60)
            
            logger.info("开始执行数据库维护...")
            
            # 获取数据库实例
            db = await AsyncDBPool.get_instance()
            
            # 清理过期会话
            await db.cleanup_old_sessions()
            
            # 执行VACUUM优化数据库
            try:
                async with aiosqlite.connect(DB_PATH) as conn:
                    await conn.execute("VACUUM")
                    await conn.commit()
                logger.info("数据库VACUUM完成")
            except Exception as e:
                logger.error(f"数据库VACUUM失败: {str(e)}")
            
            logger.info("数据库维护完成")
            
        except Exception as e:
            logger.error(f"数据库维护任务出错: {str(e)}")
            await asyncio.sleep(60 * 60)  # 出错后等待一小时再试


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7898)

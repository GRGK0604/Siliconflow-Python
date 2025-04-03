from fastapi import FastAPI, Request, HTTPException, Depends, Response, Cookie, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from config import API_KEY, ADMIN_USERNAME, ADMIN_PASSWORD
import json
import random
import time
import asyncio
import aiohttp
import uuid
from uvicorn.config import LOGGING_CONFIG
from typing import Optional, List, Dict, Any, Tuple

# Import our new modules
from db import AsyncDBPool
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

app = FastAPI(
    title="SiliconFlow API",
    description="A proxy server for the Silicon Flow API with key management",
    version="2.0.0"
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

# Initialize the database on startup
@app.on_event("startup")
async def startup_event():
    # Initialize the database
    db = await AsyncDBPool.get_instance()
    await db.initialize()

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
            # Skip validation for duplicate keys
            tasks.append(asyncio.sleep(0, result=("duplicate", key)))
        else:
            tasks.append(validate_key_async(key))
    
    results = await asyncio.gather(*tasks)
    
    # Process results
    imported_count = 0
    duplicate_count = len(duplicate_keys)
    invalid_count = 0
    
    for idx, result in enumerate(results):
        if isinstance(result, tuple) and result[0] == "duplicate":
            continue
        else:
            valid, balance = result
            if valid and float(balance) > 0:
                await db.insert_api_key(keys[idx], balance)
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
    body = await request.json()
    
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
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{BASE_URL}/v1/models", headers=forward_headers, timeout=30
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
    
    return JSONResponse(StatsResponse(**stats_data).dict())


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
    ).dict())


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
    ).dict())

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
    # Log the error here if needed
    print(f"Unhandled exception: {str(exc)}")
    
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7898)

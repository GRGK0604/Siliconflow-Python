from fastapi import FastAPI, Request, HTTPException, Depends, Response, Cookie
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from config import API_KEY, ADMIN_USERNAME, ADMIN_PASSWORD
import sqlite3
import random
import time
import asyncio
import aiohttp
import json
import secrets
from contextlib import contextmanager
from uvicorn.config import LOGGING_CONFIG

LOGGING_CONFIG["formatters"]["default"]["fmt"] = (
    "%(asctime)s - %(levelprefix)s %(message)s"
)


app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"))

# SQLite DB initialization
conn = sqlite3.connect("pool.db", check_same_thread=False)

# 创建一个上下文管理器来处理数据库连接和游标
@contextmanager
def get_cursor():
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception as e:
        conn.rollback()
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
        usage_count INTEGER
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

BASE_URL = "https://api.siliconflow.cn"  # adjust if needed


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
def create_session(username: str) -> str:
    session_id = secrets.token_hex(16)
    with get_cursor() as cursor:
        cursor.execute(
            "INSERT INTO sessions (session_id, username, created_at) VALUES (?, ?, ?)",
            (session_id, username, time.time())
        )
    return session_id


def validate_session(session_id: str = Cookie(None)) -> bool:
    if not session_id:
        return False
    
    # 使用单独的游标清理旧会话
    with get_cursor() as cursor:
        # Clean up old sessions (older than 24 hours)
        cursor.execute(
            "DELETE FROM sessions WHERE created_at < ?",
            (time.time() - 86400,)  # 24 hours in seconds
        )
    
    # 使用另一个游标检查会话是否存在
    with get_cursor() as cursor:
        cursor.execute("SELECT username FROM sessions WHERE session_id = ?", (session_id,))
        result = cursor.fetchone()
        return bool(result)


def require_auth(session_id: str = Cookie(None)):
    """验证用户是否已登录，如果未登录则抛出401异常"""
    if not session_id:
        raise HTTPException(status_code=401, detail="未授权访问，请先登录")
    
    # 使用单独的游标清理过期会话
    try:
        with get_cursor() as cursor:
            # 清理过期会话
            cursor.execute(
                "DELETE FROM sessions WHERE created_at < ?",
                (time.time() - 86400,)  # 24 hours in seconds
            )
    except Exception as e:
        print(f"清理过期会话时出错: {str(e)}")
        # 继续执行，不要因为清理失败而阻止用户访问
    
    # 使用另一个游标检查当前会话
    try:
        with get_cursor() as cursor:
            # 查询当前会话
            cursor.execute("SELECT username FROM sessions WHERE session_id = ?", (session_id,))
            result = cursor.fetchone()
            
            if not result:
                # 如果会话不存在，抛出401异常
                raise HTTPException(status_code=401, detail="会话已过期，请重新登录")
    except sqlite3.Error as e:
        print(f"检查会话时出错: {str(e)}")
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
    with get_cursor() as cursor:
        cursor.execute(
            "INSERT OR IGNORE INTO api_keys (key, add_time, balance, usage_count) VALUES (?, ?, ?, ?)",
            (api_key, time.time(), balance, 0),
        )


def log_completion(
    used_key: str,
    model: str,
    call_time: float,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
):
    with get_cursor() as cursor:
        cursor.execute(
            "INSERT INTO logs (used_key, model, call_time, input_tokens, output_tokens, total_tokens) VALUES (?, ?, ?, ?, ?, ?)",
            (used_key, model, call_time, input_tokens, output_tokens, total_tokens),
        )


@app.get("/")
async def root(session_id: str = Cookie(None)):
    # 检查用户是否已登录
    if session_id:
        try:
            with get_cursor() as cursor:
                cursor.execute("SELECT username FROM sessions WHERE session_id = ?", (session_id,))
                result = cursor.fetchone()
                if result:
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
async def login(request: Request):
    data = await request.json()
    username = data.get("username")
    password = data.get("password")
    
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session_id = create_session(username)
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
        with get_cursor() as cursor:
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    
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
    with get_cursor() as cursor:
        for key in keys:
            cursor.execute("SELECT key FROM api_keys WHERE key = ?", (key,))
            if cursor.fetchone():
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
    with get_cursor() as cursor:
        cursor.execute("SELECT key FROM api_keys")
        all_keys = [row[0] for row in cursor.fetchall()]

    # Create tasks for parallel validation
    tasks = [validate_key_async(key) for key in all_keys]
    results = await asyncio.gather(*tasks)

    removed = 0
    for key, (valid, balance) in zip(all_keys, results):
        if valid and float(balance) > 0:
            with get_cursor() as cursor:
                cursor.execute(
                    "UPDATE api_keys SET balance = ? WHERE key = ?", (balance, key)
                )
        else:
            with get_cursor() as cursor:
                cursor.execute("DELETE FROM api_keys WHERE key = ?", (key,))
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
            async with session.post(
                f"{BASE_URL}/v1/chat/completions",
                headers=headers,
                json=req_json,
                timeout=300,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
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
                                    except:
                                        pass
                        except Exception as e:
                            print(f"Error processing line: {e}")
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
                
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'error': '请求超时'})}\n\n".encode('utf-8')
        except Exception as e:
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
    with get_cursor() as cursor:
        cursor.execute("SELECT key FROM api_keys")
        keys = [row[0] for row in cursor.fetchall()]
    
    if not keys:
        raise HTTPException(status_code=500, detail="没有可用的API密钥")
    
    # 随机选择一个密钥
    selected_key = random.choice(keys)
    
    # 更新使用计数
    with get_cursor() as cursor:
        cursor.execute(
            "UPDATE api_keys SET usage_count = usage_count + 1 WHERE key = ?", 
            (selected_key,)
        )
    
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
                async with session.post(
                    "https://api.siliconflow.cn/v1/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=120,
                ) as response:
                    response_data = await response.json()
                    
                    # 计算 token 使用量
                    usage = response_data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    
                    # 记录使用情况
                    log_completion(
                        selected_key,
                        model,  # 使用请求中的模型名称
                        call_time_stamp,
                        prompt_tokens,
                        completion_tokens,
                        prompt_tokens + completion_tokens,
                    )
                    
                    return JSONResponse(response_data)
        except Exception as e:
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
    
    with get_cursor() as cursor:
        cursor.execute("SELECT key FROM api_keys")
        keys = [row[0] for row in cursor.fetchall()]
    
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
    with get_cursor() as cursor:
        cursor.execute("SELECT key FROM api_keys")
        keys = [row[0] for row in cursor.fetchall()]
    
    if not keys:
        raise HTTPException(status_code=500, detail="没有可用的api-key")
    
    selected = random.choice(keys)
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


@app.get("/stats")
async def stats(authorized: bool = Depends(require_auth)):
    with get_cursor() as cursor:
        cursor.execute("SELECT COUNT(*), COALESCE(SUM(balance), 0) FROM api_keys")
        count, total_balance = cursor.fetchone()
    
    return JSONResponse({"key_count": count, "total_balance": total_balance})


@app.get("/export_keys")
async def export_keys(authorized: bool = Depends(require_auth)):
    with get_cursor() as cursor:
        cursor.execute("SELECT key FROM api_keys")
        all_keys = cursor.fetchall()
    
    keys = "\n".join(row[0] for row in all_keys)
    headers = {"Content-Disposition": "attachment; filename=keys.txt"}
    return Response(content=keys, media_type="text/plain", headers=headers)


@app.get("/logs")
async def get_logs(page: int = 1, authorized: bool = Depends(require_auth)):
    page_size = 10
    offset = (page - 1) * page_size
    
    try:
        with get_cursor() as cursor:
            # 获取总记录数
            cursor.execute("SELECT COUNT(*) FROM logs")
            total = cursor.fetchone()[0]
            
            # 获取分页数据
            cursor.execute(
                """
                SELECT used_key, model, call_time, input_tokens, output_tokens, total_tokens 
                FROM logs 
                ORDER BY call_time DESC 
                LIMIT ? OFFSET ?
                """,
                (page_size, offset)
            )
            logs = cursor.fetchall()
        
        # 格式化日志数据
        formatted_logs = []
        for log in logs:
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
        with get_cursor() as cursor:
            cursor.execute("DELETE FROM logs")
        
        with get_cursor() as cursor:
            cursor.execute("VACUUM")
        
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

    with get_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM api_keys")
        total = cursor.fetchone()[0]

        cursor.execute(
            f"SELECT key, add_time, balance, usage_count FROM api_keys ORDER BY {sort_field} {sort_order} LIMIT ? OFFSET ?",
            (page_size, offset),
        )
        keys = cursor.fetchall()

    # Format keys as list of dicts
    key_list = [
        {
            "key": row[0],
            "add_time": row[1],
            "balance": row[2],
            "usage_count": row[3],
        }
        for row in keys
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
        with get_cursor() as cursor:
            cursor.execute(
                "UPDATE api_keys SET balance = ? WHERE key = ?", 
                (balance, key)
            )
        return JSONResponse({"message": "密钥刷新成功", "balance": balance})
    else:
        with get_cursor() as cursor:
            cursor.execute("DELETE FROM api_keys WHERE key = ?", (key,))
        raise HTTPException(status_code=400, detail="密钥无效或余额为零，已从系统中移除")

@app.post("/api/delete_key")
async def delete_key(request: Request, authorized: bool = Depends(require_auth)):
    data = await request.json()
    key = data.get("key")

    if not key:
        raise HTTPException(status_code=400, detail="未提供API密钥")

    try:
        with get_cursor() as cursor:
            cursor.execute("DELETE FROM api_keys WHERE key = ?", (key,))
        return JSONResponse({"message": "密钥已成功删除"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除密钥失败: {str(e)}")

@app.get("/api/key_info")
async def get_key_info(key: str, authorized: bool = Depends(require_auth)):
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT key, balance, usage_count, add_time FROM api_keys WHERE key = ?", 
            (key,)
        )
        result = cursor.fetchone()
    
    if not result:
        raise HTTPException(status_code=404, detail="密钥不存在")
    
    return JSONResponse({
        "key": result[0],
        "balance": result[1],
        "usage_count": result[2],
        "add_time": result[3]
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
    response = FileResponse("static/keys.html")
    # 添加缓存控制头
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7898)

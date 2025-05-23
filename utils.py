import aiohttp
import secrets
import time
import asyncio
from typing import Tuple, Dict, Any, Optional
from functools import lru_cache
import logging

# Base URL for Silicon Flow API
BASE_URL = "https://api.siliconflow.cn"

# Cache for API key validation results
API_KEY_CACHE = {}
API_KEY_CACHE_TTL = 60  # seconds
CACHE_MAX_SIZE = 1000  # 最大缓存条目数

# Cache for stats
STATS_CACHE = {}
STATS_CACHE_TIME = 0
STATS_CACHE_TTL = 60  # seconds

# 全局连接器，用于复用连接池
_connector = None
_session = None

logger = logging.getLogger("siliconflow.utils")

async def get_session():
    """获取全局aiohttp会话，使用连接池"""
    global _connector, _session
    
    if _session is None or _session.closed:
        # 创建连接器，配置连接池
        _connector = aiohttp.TCPConnector(
            limit=100,  # 总连接池大小
            limit_per_host=30,  # 每个主机的连接数
            ttl_dns_cache=300,  # DNS缓存时间
            use_dns_cache=True,
            keepalive_timeout=60,  # 保持连接时间
            enable_cleanup_closed=True
        )
        
        # 创建客户端会话
        timeout = aiohttp.ClientTimeout(
            total=30,      # 总超时时间
            connect=10,    # 连接超时
            sock_read=20   # 读取超时
        )
        
        _session = aiohttp.ClientSession(
            connector=_connector,
            timeout=timeout,
            headers={
                'User-Agent': 'SiliconFlow-Proxy/2.0',
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate'
            }
        )
    
    return _session

async def cleanup_cache():
    """清理过期的缓存条目"""
    current_time = time.time()
    expired_keys = []
    
    for cache_key, (timestamp, _) in API_KEY_CACHE.items():
        if current_time - timestamp > API_KEY_CACHE_TTL:
            expired_keys.append(cache_key)
    
    for key in expired_keys:
        del API_KEY_CACHE[key]
    
    # 如果缓存太大，删除最旧的条目
    if len(API_KEY_CACHE) > CACHE_MAX_SIZE:
        # 按时间戳排序，删除最旧的条目
        sorted_items = sorted(API_KEY_CACHE.items(), key=lambda x: x[1][0])
        items_to_remove = len(sorted_items) - CACHE_MAX_SIZE + 100  # 多删除一些，避免频繁清理
        
        for i in range(items_to_remove):
            del API_KEY_CACHE[sorted_items[i][0]]
        
        logger.info(f"缓存清理完成，删除了 {len(expired_keys)} 个过期条目和 {items_to_remove} 个旧条目")

async def validate_key_async(api_key: str, max_retries: int = 3) -> Tuple[bool, Any]:
    """
    Validate an API key against the Silicon Flow API with retry mechanism.
    Returns (is_valid, balance_or_error_message) tuple.
    """
    # Check if we have a non-expired cache entry
    current_time = time.time()
    cache_key = f"key_{api_key}"
    
    if cache_key in API_KEY_CACHE:
        timestamp, result = API_KEY_CACHE[cache_key]
        if current_time - timestamp < API_KEY_CACHE_TTL:
            return result
    
    # 定期清理缓存
    if len(API_KEY_CACHE) % 50 == 0:  # 每50次调用清理一次
        await cleanup_cache()
    
    # No valid cache entry, make the API call with retry
    headers = {"Authorization": f"Bearer {api_key}"}
    
    for attempt in range(max_retries):
        try:
            session = await get_session()
            async with session.get(
                f"{BASE_URL}/v1/user/info", 
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    balance = data.get("data", {}).get("totalBalance", 0)
                    result = (True, balance)
                elif r.status == 401:
                    # 认证失败，不重试
                    result = (False, "API密钥无效")
                elif r.status == 429:
                    # 速率限制，等待后重试
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + 1  # 指数退避
                        logger.warning(f"API限制，等待 {wait_time} 秒后重试...")
                        await asyncio.sleep(wait_time)
                        continue
                    result = (False, "API速率限制")
                else:
                    data = await r.json()
                    result = (False, data.get("message", f"API返回错误: {r.status}"))
                
                break  # 成功获得响应，退出重试循环
                
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                logger.warning(f"验证密钥超时，重试 {attempt + 1}/{max_retries}")
                await asyncio.sleep(1)
                continue
            result = (False, "请求超时")
        except aiohttp.ClientError as e:
            if attempt < max_retries - 1:
                logger.warning(f"网络错误，重试 {attempt + 1}/{max_retries}: {str(e)}")
                await asyncio.sleep(1)
                continue
            result = (False, f"网络错误: {str(e)}")
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"未知错误，重试 {attempt + 1}/{max_retries}: {str(e)}")
                await asyncio.sleep(1)
                continue
            result = (False, f"验证失败: {str(e)}")
    
    # Cache the result
    API_KEY_CACHE[cache_key] = (current_time, result)
    
    return result

def generate_session_id() -> str:
    """Generate a random session ID."""
    return secrets.token_hex(16)

def invalidate_stats_cache() -> None:
    """Invalidate the stats cache."""
    global STATS_CACHE_TIME
    STATS_CACHE_TIME = 0

async def make_api_request(
    endpoint: str, 
    api_key: str, 
    method: str = "GET", 
    data: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30
) -> Tuple[int, Dict[str, Any]]:
    """
    Make a request to the Silicon Flow API.
    Returns (status_code, response_data) tuple.
    """
    url = f"{BASE_URL}{endpoint}"
    
    # Setup headers
    request_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Add additional headers if provided
    if headers:
        request_headers.update(headers)
    
    try:
        async with aiohttp.ClientSession() as session:
            if method.upper() == "GET":
                async with session.get(url, headers=request_headers, timeout=timeout) as resp:
                    status = resp.status
                    data = await resp.json()
            elif method.upper() == "POST":
                async with session.post(
                    url, headers=request_headers, json=data, timeout=timeout
                ) as resp:
                    status = resp.status
                    data = await resp.json()
            else:
                return 400, {"error": f"Unsupported method: {method}"}
                
            return status, data
    except Exception as e:
        return 500, {"error": f"API request failed: {str(e)}"}

# Function to add streaming support for proxy requests
async def stream_response(api_key: str, endpoint: str, request_data: Dict[str, Any], headers: Dict[str, str]):
    """
    Stream a response from the Silicon Flow API.
    Yields chunks of data as they arrive.
    """
    url = f"{BASE_URL}{endpoint}"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                url, headers=headers, json=request_data, timeout=300
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    try:
                        error_json = f"data: {error_text}\n\n"
                    except:
                        error_json = f"data: {{\"error\": \"API返回错误: HTTP {resp.status}\"}}\n\n"
                    yield error_json.encode('utf-8')
                    return
                
                # Stream the response line by line
                async for line in resp.content:
                    if line:
                        yield line
        except aiohttp.ClientError as e:
            error_json = f"data: {{\"error\": \"连接API失败: {str(e)}\"}}\n\n"
            yield error_json.encode('utf-8')
            yield b"data: [DONE]\n\n"
        except asyncio.TimeoutError:
            error_json = f"data: {{\"error\": \"请求超时\"}}\n\n"
            yield error_json.encode('utf-8')
            yield b"data: [DONE]\n\n"
        except Exception as e:
            error_json = f"data: {{\"error\": \"处理请求时发生错误: {str(e)}\"}}\n\n"
            yield error_json.encode('utf-8')
            yield b"data: [DONE]\n\n" 
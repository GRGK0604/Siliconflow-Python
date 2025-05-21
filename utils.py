import aiohttp
import secrets
import time
import asyncio
from typing import Tuple, Dict, Any, Optional
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

# Base URL for Silicon Flow API
BASE_URL = "https://api.siliconflow.cn"

# Cache for API key validation results
API_KEY_CACHE = {}
API_KEY_CACHE_TTL = 60  # seconds

# Cache for stats
STATS_CACHE = {}
STATS_CACHE_TIME = 0
STATS_CACHE_TTL = 60  # seconds

# Setup logger
logger = logging.getLogger("siliconflow")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def validate_key_async(api_key: str) -> Tuple[Optional[bool], Optional[float], Optional[str]]:
    """
    Validate an API key against the Silicon Flow API.
    Returns (is_valid, balance, error_message) tuple:
    - is_valid: True if status code is 200 and totalBalance > 0, False otherwise
    - balance: Balance if status code is 200, None otherwise
    - error_message: Error details if validation failed or key is invalid
    """
    logger.info(f"开始验证API密钥 {api_key[:4]}****")
    
    # Check if we have a non-expired cache entry
    current_time = time.time()
    cache_key = f"key_{api_key}"
    
    if cache_key in API_KEY_CACHE:
        timestamp, result = API_KEY_CACHE[cache_key]
        if current_time - timestamp < API_KEY_CACHE_TTL:
            logger.debug(f"缓存命中 for key {api_key[:4]}****: {result}")
            return result
    
    # No valid cache entry, make the API call
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with aiohttp.ClientSession() as session:
            logger.debug(f"发送请求到 {BASE_URL}/v1/user/info")
            async with session.get(
                f"{BASE_URL}/v1/user/info", headers=headers, timeout=10
            ) as r:
                response_text = await r.text()
                logger.debug(f"API 响应状态码: {r.status}, 内容: {response_text}")
                if r.status == 200:
                    data = await r.json()
                    balance = data.get("data", {}).get("totalBalance", 0)
                    if balance > 0:
                        result = (True, float(balance), None)
                        logger.info(f"验证成功 for key {api_key[:4]}****: balance={balance}")
                    else:
                        result = (False, None, "余额为 0")
                        logger.warning(f"验证失败 for key {api_key[:4]}****: 余额为 0")
                else:
                    error_msg = f"状态码 {r.status}: {response_text}"
                    result = (False, None, error_msg)
                    logger.warning(f"验证失败 for key {api_key[:4]}****: {error_msg}")
    except aiohttp.ClientError as e:
        result = (False, None, f"网络错误: {str(e)}")
        logger.error(f"网络错误 validating key {api_key[:4]}****: {str(e)}")
    except asyncio.TimeoutError:
        result = (False, None, "请求超时")
        logger.error(f"请求超时 validating key {api_key[:4]}****")
    except Exception as e:
        result = (False, None, f"未知错误: {str(e)}")
        logger.error(f"未知错误 validating key {api_key[:4]}****: {str(e)}")
    
    # Cache only successful results
    if result[0] is True:
        API_KEY_CACHE[cache_key] = (current_time, result)
        logger.debug(f"缓存结果 for key {api_key[:4]}****: {result}")
    
    return result

def generate_session_id() -> str:
    """Generate a random session ID."""
    return secrets.token_hex(16)

def invalidate_stats_cache() -> None:
    """Invalidate the stats cache."""
    global STATS_CACHE_TIME
    STATS_CACHE_TIME = 0
    logger.debug("Stats cache invalidated")

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
                logger.error(f"不支持的请求方法: {method}")
                return 400, {"error": f"不支持的请求方法: {method}"}
                
            logger.debug(f"API 请求 {endpoint} 返回状态码 {status}")
            return status, data
    except Exception as e:
        logger.error(f"API 请求 {endpoint} 失败: {str(e)}")
        return 500, {"error": f"API 请求失败: {str(e)}"}

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
                    logger.error(f"流式请求 {endpoint} 失败，状态码 {resp.status}: {error_text}")
                    yield f"data: {{'error': '{error_text}'}}\n\n".encode('utf-8')
                    return
                
                # Stream the response line by line
                async for line in resp.content:
                    if line:
                        yield line
                        logger.debug(f"流式传输块 from {endpoint}")
        except Exception as e:
            error_json = f"data: {{'error': '{str(e)}'}}\n\n"
            logger.error(f"流式请求 {endpoint} 失败: {str(e)}")
            yield error_json.encode('utf-8')
            yield b"data: [DONE]\n\n"
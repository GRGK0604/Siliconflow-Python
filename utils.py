import aiohttp
import secrets
import time
import asyncio
from typing import Tuple, Dict, Any, Optional
from functools import lru_cache

# Base URL for Silicon Flow API
BASE_URL = "https://api.siliconflow.cn"

# Cache for API key validation results
API_KEY_CACHE = {}
API_KEY_CACHE_TTL = 60  # seconds

# Cache for stats
STATS_CACHE = {}
STATS_CACHE_TIME = 0
STATS_CACHE_TTL = 60  # seconds

# Use LRU cache for API key validation results to reduce API calls
async def validate_key_async(api_key: str) -> Tuple[bool, Any]:
    """
    Validate an API key against the Silicon Flow API.
    Returns (is_valid, balance_or_error_message) tuple.
    """
    # Check if we have a non-expired cache entry
    current_time = time.time()
    cache_key = f"key_{api_key}"
    
    if cache_key in API_KEY_CACHE:
        timestamp, result = API_KEY_CACHE[cache_key]
        if current_time - timestamp < API_KEY_CACHE_TTL:
            return result
    
    # No valid cache entry, make the API call
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{BASE_URL}/v1/user/info", headers=headers, timeout=10
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    balance = data.get("data", {}).get("totalBalance", 0)
                    result = (True, balance)
                else:
                    data = await r.json()
                    result = (False, data.get("message", "验证失败"))
    except Exception as e:
        result = (False, f"请求失败: {str(e)}")
    
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
import aiosqlite
import asyncio
from typing import Optional, List, Dict, Any, Tuple
import time
import os

# Path to SQLite database file
DB_PATH = os.path.join("data", "pool.db")

class AsyncDBPool:
    _instance = None
    _pool = None
    _lock = asyncio.Lock()
    
    @classmethod
    async def get_instance(cls):
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    await cls._instance.initialize()
        return cls._instance
    
    async def initialize(self):
        """Initialize the database and create tables if they don't exist."""
        # 确保数据目录存在
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        
        async with aiosqlite.connect(DB_PATH) as db:
            # Enable WAL mode for better concurrency
            await db.execute("PRAGMA journal_mode=WAL")
            
            # Set synchronous mode for better performance while maintaining safety
            await db.execute("PRAGMA synchronous=NORMAL")
            
            # Create tables if they don't exist
            await db.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                key TEXT PRIMARY KEY,
                add_time REAL,
                balance REAL,
                usage_count INTEGER,
                enabled INTEGER DEFAULT 1
            )
            """)

            # Create logs table for recording completion calls
            await db.execute("""
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
            await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                username TEXT,
                created_at REAL
            )
            """)
            
            # Create indexes for frequently queried columns
            await db.execute("CREATE INDEX IF NOT EXISTS idx_logs_call_time ON logs(call_time)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_balance ON api_keys(balance)")
            
            await db.commit()
    
    async def execute(self, query: str, params: tuple = (), fetch_one: bool = False, fetch_all: bool = False) -> Any:
        """Execute a SQL query and optionally fetch results."""
        max_retries = 3
        retry_count = 0
        retry_delay = 0.5  # 初始延迟0.5秒
        
        while retry_count < max_retries:
            try:
                async with aiosqlite.connect(DB_PATH, timeout=10) as db:
                    db.row_factory = aiosqlite.Row
                    async with db.execute(query, params) as cursor:
                        if fetch_one:
                            return await cursor.fetchone()
                        elif fetch_all:
                            return await cursor.fetchall()
                        else:
                            await db.commit()
                            return cursor.lastrowid
            except aiosqlite.OperationalError as e:
                # 数据库锁定或其他操作错误，尝试重试
                retry_count += 1
                if retry_count >= max_retries:
                    raise
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
            except Exception as e:
                # 其他错误直接抛出
                raise
    
    async def get_all_keys(self) -> List[Dict[str, Any]]:
        """Get all API keys from the database."""
        rows = await self.execute("SELECT key, add_time, balance, usage_count FROM api_keys", fetch_all=True)
        return [dict(row) for row in rows] if rows else []

    async def get_key_list(self) -> List[str]:
        """Get list of all API key strings."""
        rows = await self.execute("SELECT key FROM api_keys", fetch_all=True)
        return [row['key'] for row in rows] if rows else []
    
    async def get_best_keys(self, limit: int = 10) -> List[str]:
        """Get the best API keys based on balance and usage count."""
        rows = await self.execute(
            "SELECT key FROM api_keys ORDER BY (balance / (usage_count + 1)) DESC LIMIT ?", 
            (limit,), 
            fetch_all=True
        )
        return [row['key'] for row in rows] if rows else []
    
    async def insert_api_key(self, key: str, balance: float) -> None:
        """Insert a new API key if it doesn't exist."""
        await self.execute(
            "INSERT OR IGNORE INTO api_keys (key, add_time, balance, usage_count) VALUES (?, ?, ?, ?)",
            (key, time.time(), balance, 0)
        )
    
    async def update_key_balance(self, key: str, balance: float) -> None:
        """Update the balance of an API key."""
        await self.execute(
            "UPDATE api_keys SET balance = ? WHERE key = ?", 
            (balance, key)
        )
    
    async def increment_key_usage(self, key: str) -> None:
        """Increment the usage count of an API key."""
        await self.execute(
            "UPDATE api_keys SET usage_count = usage_count + 1 WHERE key = ?", 
            (key,)
        )
    
    async def delete_key(self, key: str) -> None:
        """Delete an API key from the database."""
        await self.execute("DELETE FROM api_keys WHERE key = ?", (key,))
    
    async def log_completion(self, used_key: str, model: str, call_time: float, 
                           input_tokens: int, output_tokens: int, total_tokens: int) -> None:
        """Log a completion call."""
        await self.execute(
            "INSERT INTO logs (used_key, model, call_time, input_tokens, output_tokens, total_tokens) VALUES (?, ?, ?, ?, ?, ?)",
            (used_key, model, call_time, input_tokens, output_tokens, total_tokens)
        )
    
    async def create_session(self, session_id: str, username: str) -> None:
        """Create a new user session."""
        await self.execute(
            "INSERT INTO sessions (session_id, username, created_at) VALUES (?, ?, ?)",
            (session_id, username, time.time())
        )
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID."""
        row = await self.execute(
            "SELECT username, created_at FROM sessions WHERE session_id = ?", 
            (session_id,), 
            fetch_one=True
        )
        return dict(row) if row else None
    
    async def delete_session(self, session_id: str) -> None:
        """Delete a session by ID."""
        await self.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    
    async def cleanup_old_sessions(self, max_age: int = 86400) -> None:
        """Clean up sessions older than max_age seconds."""
        await self.execute(
            "DELETE FROM sessions WHERE created_at < ?",
            (time.time() - max_age,)
        )
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get system stats."""
        key_count = await self.execute(
            "SELECT COUNT(*) as count FROM api_keys", 
            fetch_one=True
        )
        
        total_balance = await self.execute(
            "SELECT COALESCE(SUM(balance), 0) as sum FROM api_keys", 
            fetch_one=True
        )
        
        total_calls = await self.execute(
            "SELECT COUNT(*) as count FROM logs", 
            fetch_one=True
        )
        
        total_tokens = await self.execute(
            "SELECT COALESCE(SUM(total_tokens), 0) as sum FROM logs", 
            fetch_one=True
        )
        
        return {
            "key_count": key_count['count'] if key_count else 0,
            "total_balance": total_balance['sum'] if total_balance else 0,
            "total_calls": total_calls['count'] if total_calls else 0,
            "total_tokens": total_tokens['sum'] if total_tokens else 0
        }
    
    async def get_logs(self, page: int = 1, page_size: int = 10, model: str = None) -> Tuple[List[Dict[str, Any]], int]:
        """Get paginated logs with optional model filter."""
        offset = (page - 1) * page_size
        params = []
        
        # Build query
        count_query = "SELECT COUNT(*) as count FROM logs"
        data_query = """
            SELECT used_key, model, call_time, input_tokens, output_tokens, total_tokens 
            FROM logs
        """
        
        # Add model filter if provided
        if model:
            where_clause = " WHERE model LIKE ?"
            count_query += where_clause
            data_query += where_clause
            params.append(f"%{model}%")
        
        # Add pagination
        data_query += " ORDER BY call_time DESC LIMIT ? OFFSET ?"
        params_with_pagination = params.copy()
        params_with_pagination.extend([page_size, offset])
        
        # Get total count
        total_row = await self.execute(count_query, tuple(params), fetch_one=True)
        total = total_row['count'] if total_row else 0
        
        # Get data
        rows = await self.execute(data_query, tuple(params_with_pagination), fetch_all=True)
        logs = [dict(row) for row in rows] if rows else []
        
        return logs, total
    
    async def clear_logs(self) -> None:
        """Clear all logs from the database."""
        await self.execute("DELETE FROM logs")
        # 执行VACUUM优化数据库空间
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute("VACUUM")
                await db.commit()
        except Exception as e:
            # 如果VACUUM失败，记录错误但不阻止清理日志的操作完成
            print(f"执行VACUUM时出错: {str(e)}")
            pass 
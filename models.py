from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import time

class LoginRequest(BaseModel):
    username: str
    password: str

class APIKeyImport(BaseModel):
    keys: str = Field(..., description="API keys separated by newlines")

class APIKeyInfo(BaseModel):
    key: str
    add_time: float
    balance: float
    usage_count: int
    enabled: bool = True

class APIKeyResponse(BaseModel):
    keys: List[APIKeyInfo]
    total: int
    page: int
    page_size: int

class APIKeyRefresh(BaseModel):
    key: str

class LogEntry(BaseModel):
    api_key: str
    model: str = "unknown"
    call_time: float
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

class LogsResponse(BaseModel):
    logs: List[LogEntry]
    total: int
    page: int
    page_size: int

class StatsResponse(BaseModel):
    key_count: int = 0
    total_balance: float = 0
    total_calls: int = 0
    total_tokens: int = 0

class MessageResponse(BaseModel):
    message: str

class ErrorResponse(BaseModel):
    detail: str

# ChatGPT API Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    stream: Optional[bool] = False

class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: TokenUsage 
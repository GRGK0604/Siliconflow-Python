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

# Anthropic Messages API Models
class AnthropicMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stream: Optional[bool] = False
    stop_sequences: Optional[List[str]] = None
    system: Optional[str] = None

# Rerank API Models
class RerankDocument(BaseModel):
    text: str

class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[Union[str, RerankDocument]]
    top_n: Optional[int] = None
    return_documents: Optional[bool] = True

class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[Dict[str, Any]] = None

class RerankResponse(BaseModel):
    id: str
    results: List[RerankResult]
    meta: Optional[Dict[str, Any]] = None

# Image Generation API Models
class ImageGenerationRequest(BaseModel):
    model: str
    prompt: str
    negative_prompt: Optional[str] = None
    image_size: Optional[str] = "1024x1024"
    batch_size: Optional[int] = 1
    num_inference_steps: Optional[int] = 20
    guidance_scale: Optional[float] = 7.5
    seed: Optional[int] = None

class ImageData(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None

class ImageGenerationResponse(BaseModel):
    created: int = Field(default_factory=lambda: int(time.time()))
    data: List[ImageData]

# Audio API Models
class AudioTranscriptionRequest(BaseModel):
    model: str
    file: str  # This will be handled as multipart/form-data
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[str] = "json"
    temperature: Optional[float] = 0

class AudioTranscriptionResponse(BaseModel):
    text: str

class TextToSpeechRequest(BaseModel):
    model: str
    input: str
    voice: str
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0

# Video Generation API Models
class VideoGenerationRequest(BaseModel):
    model: str
    prompt: str
    image_url: Optional[str] = None
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 6.0
    num_frames: Optional[int] = 49
    fps: Optional[int] = 8
    seed: Optional[int] = None

class VideoGenerationResponse(BaseModel):
    id: str
    status: str
    created_at: int = Field(default_factory=lambda: int(time.time()))

class VideoStatusResponse(BaseModel):
    id: str
    status: str
    video_url: Optional[str] = None
    created_at: int
    completed_at: Optional[int] = None
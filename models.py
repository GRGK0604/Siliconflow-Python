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

# Anthropic Chat Models
class AnthropicMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class AnthropicChatRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stream: Optional[bool] = False
    system: Optional[str] = None

# Embeddings Models
class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: TokenUsage

# Reranking Models
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
    results: List[RerankResult]
    model: str
    usage: Optional[Dict[str, Any]] = None

# Image Generation Models
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
    url: str
    seed: Optional[int] = None

class ImageResponse(BaseModel):
    images: List[ImageData]
    timings: Optional[Dict[str, float]] = None

# Audio Models
class AudioUploadRequest(BaseModel):
    audio_name: str

class TTSRequest(BaseModel):
    model: str
    input: str
    voice: str
    audio_name: Optional[str] = None
    language: Optional[str] = "zh"
    speed: Optional[float] = 1.0

class STTRequest(BaseModel):
    model: str
    audio_url: str
    language: Optional[str] = "zh"

class AudioListResponse(BaseModel):
    audios: List[Dict[str, Any]]

# Video Generation Models
class VideoGenerationRequest(BaseModel):
    model: str
    prompt: str
    image_url: Optional[str] = None
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 6.0
    seed: Optional[int] = None

class VideoResponse(BaseModel):
    video_id: str
    status: str

class VideoStatusRequest(BaseModel):
    video_id: str

class VideoStatusResponse(BaseModel):
    video_id: str
    status: str
    video_url: Optional[str] = None
    error: Optional[str] = None

# Batch Processing Models
class BatchRequest(BaseModel):
    input_file_id: str
    endpoint: str
    completion_window: str = "24h"
    metadata: Optional[Dict[str, str]] = None

class BatchResponse(BaseModel):
    id: str
    object: str = "batch"
    endpoint: str
    errors: Optional[Dict[str, Any]] = None
    input_file_id: str
    completion_window: str
    status: str
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    created_at: int
    in_progress_at: Optional[int] = None
    expires_at: Optional[int] = None
    finalizing_at: Optional[int] = None
    completed_at: Optional[int] = None
    failed_at: Optional[int] = None
    expired_at: Optional[int] = None
    cancelling_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    request_counts: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, str]] = None

class FileUploadResponse(BaseModel):
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str

class FileListResponse(BaseModel):
    object: str = "list"
    data: List[FileUploadResponse]

# Platform Models
class UserModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str

class UserModelsResponse(BaseModel):
    object: str = "list"
    data: List[UserModelInfo]

class UserAccountInfo(BaseModel):
    email: str
    total_balance: float
    used_balance: float
    available_balance: float 
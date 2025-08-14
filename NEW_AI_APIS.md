# 新增AI接口文档

本次更新为SiliconFlow Python代理服务器添加了多个AI相关的接口，使其支持更多的AI功能。

## 📝 新增接口列表

### 1. Anthropic Messages API
- **路径**: `/v1/messages`
- **方法**: POST
- **功能**: 支持Anthropic Claude模型的对话接口
- **支持流式**: ✅

**请求示例**:
```json
{
  "model": "claude-3-sonnet-20240229",
  "max_tokens": 1000,
  "messages": [
    {
      "role": "user", 
      "content": "Hello, how are you?"
    }
  ],
  "temperature": 0.7,
  "stream": false
}
```

### 2. Rerank API
- **路径**: `/v1/rerank`
- **方法**: POST
- **功能**: 文档重排序，用于搜索和推荐系统

**请求示例**:
```json
{
  "model": "BAAI/bge-reranker-v2-m3",
  "query": "机器学习",
  "documents": [
    "人工智能是计算机科学的一个分支",
    "机器学习是人工智能的核心技术",
    "深度学习是机器学习的一个子领域"
  ],
  "top_n": 3
}
```

### 3. Image Generation API
- **路径**: `/v1/images/generations`
- **方法**: POST
- **功能**: AI图像生成

**请求示例**:
```json
{
  "model": "black-forest-labs/FLUX.1-schnell",
  "prompt": "A beautiful sunset over the ocean",
  "image_size": "1024x1024",
  "batch_size": 1,
  "num_inference_steps": 20,
  "guidance_scale": 7.5
}
```

### 4. Audio Transcription API
- **路径**: `/v1/audio/transcriptions`
- **方法**: POST
- **功能**: 语音转文本
- **内容类型**: multipart/form-data

**请求示例**:
```bash
curl -X POST "http://localhost:7898/v1/audio/transcriptions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1"
```

### 5. Text-to-Speech API
- **路径**: `/v1/audio/speech`
- **方法**: POST
- **功能**: 文本转语音
- **返回**: 音频文件 (MP3)

**请求示例**:
```json
{
  "model": "tts-1",
  "input": "Hello, this is a test of text to speech.",
  "voice": "alloy",
  "response_format": "mp3",
  "speed": 1.0
}
```

### 6. Video Generation API
- **路径**: `/v1/videos/generations`
- **方法**: POST
- **功能**: AI视频生成

**请求示例**:
```json
{
  "model": "cogvideox-5b",
  "prompt": "A cat playing with a ball of yarn",
  "num_inference_steps": 50,
  "guidance_scale": 6.0,
  "num_frames": 49,
  "fps": 8
}
```

### 7. Video Status API
- **路径**: `/v1/videos/{video_id}`
- **方法**: GET
- **功能**: 查询视频生成状态

**请求示例**:
```bash
curl -X GET "http://localhost:7898/v1/videos/video_123" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## 🔧 技术特性

### 认证
所有新接口都支持与现有接口相同的认证机制：
- 如果配置了`API_KEY`，需要在请求头中提供`Authorization: Bearer YOUR_API_KEY`
- 如果未配置`API_KEY`，则无需认证

### 负载均衡
- 自动从可用的API密钥池中选择密钥
- 支持密钥使用统计和余额管理
- 自动移除无效或余额不足的密钥

### 日志记录
- 所有API调用都会记录到数据库
- 包含使用的密钥、模型、时间戳和token使用量
- 支持统计分析和监控

### 错误处理
- 统一的错误响应格式
- 详细的错误信息和状态码
- 自动重试机制（在utils层实现）

## 🧪 测试

你可以使用任何HTTP客户端来测试新接口，例如：

```bash
# 测试Anthropic Messages API
curl -X POST "http://localhost:7898/v1/messages" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# 测试图像生成API
curl -X POST "http://localhost:7898/v1/images/generations" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "black-forest-labs/FLUX.1-schnell",
    "prompt": "A beautiful sunset over the ocean"
  }'
```

## 📊 监控

新接口的使用情况可以通过以下方式监控：

1. **Web管理界面**: 访问 `http://localhost:7898/admin`
2. **统计API**: `GET /api/stats/overview`
3. **日志API**: `GET /logs`
4. **健康检查**: `GET /health`

## ⚠️ 注意事项

1. **超时设置**: 不同接口有不同的超时时间
   - 图像生成: 180秒
   - 视频生成: 300秒
   - 语音处理: 120秒
   - 其他: 30-60秒

2. **文件上传**: 语音转文本接口需要上传音频文件，请确保文件格式正确

3. **流式响应**: Anthropic Messages API支持流式响应，与OpenAI Chat Completions类似

4. **资源消耗**: 视频和图像生成接口消耗较多计算资源，可能需要更长时间

## 🔄 兼容性

新增接口完全兼容SiliconFlow官方API规范，可以直接替换官方API端点使用。

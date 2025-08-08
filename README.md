# SiliconFlow API Proxy

一个功能完整的 SiliconFlow API 代理服务器，支持密钥管理、负载均衡、使用统计和所有 SiliconFlow API 端点。

## 功能特性

- 🔑 **API 密钥池管理** - 支持多个 API 密钥的导入、验证和自动刷新
- ⚖️ **智能负载均衡** - 自动选择余额最高的密钥进行请求
- 📊 **详细统计分析** - 实时监控 API 使用情况和 token 消耗
- 🔄 **自动密钥刷新** - 定期验证密钥有效性，自动移除无效密钥
- 🌐 **完整 API 支持** - 支持所有 SiliconFlow API 端点
- 📝 **请求日志记录** - 详细记录每次 API 调用的信息
- 🛡️ **安全认证** - 支持管理员登录和 API 密钥验证
- 🚀 **高性能** - 基于 FastAPI 和异步处理

## 支持的 API 端点

### 文本系列
- `POST /v1/chat/completions` - OpenAI 格式的对话完成
- `POST /v1/messages` - Anthropic Claude 对话完成
- `POST /v1/embeddings` - 文本嵌入生成
- `POST /v1/rerank` - 文本重排序

### 图像系列
- `POST /v1/image/generations` - 图像生成

### 语音系列
- `POST /v1/audio/speech` - 文本转语音
- `POST /v1/audio/transcriptions` - 语音转文本
- `POST /v1/audio/reference` - 上传参考音频
- `GET /v1/audio/reference` - 获取参考音频列表
- `POST /v1/audio/reference/delete` - 删除参考音频

### 视频系列
- `POST /v1/video/generations` - 视频生成
- `POST /v1/video/status` - 获取视频生成状态

### 批量处理
- `POST /v1/files` - 上传文件
- `GET /v1/files` - 获取文件列表
- `POST /v1/batches` - 创建批处理任务
- `GET /v1/batches` - 获取批处理任务列表
- `GET /v1/batches/{batch_id}` - 获取批处理任务详情
- `POST /v1/batches/{batch_id}/cancel` - 取消批处理任务

### 平台系列
- `GET /v1/models` - 获取可用模型列表
- `GET /v1/user/models` - 获取用户模型列表
- `GET /v1/user/info` - 获取用户账户信息

## 快速开始

### 环境要求
- Python 3.8+
- 依赖包见 `requirements.txt`

### 安装运行

1. 克隆项目
```bash
git clone <repository-url>
cd siliconflow-python
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置环境变量（可选）
```bash
# 复制配置文件
cp config.py.example config.py

# 编辑配置文件
# API_KEY = "your-api-key"  # 可选，用于保护 API 访问
# ADMIN_USERNAME = "admin"
# ADMIN_PASSWORD = "password"
# AUTO_REFRESH_INTERVAL = 3600  # 自动刷新间隔（秒）
```

4. 启动服务
```bash
python main.py
```

服务将在 `http://localhost:7898` 启动。

### Docker 部署

```bash
# 构建镜像
docker build -t siliconflow-api .

# 运行容器
docker run -d -p 7898:7898 \
  -e ADMIN_USERNAME=admin \
  -e ADMIN_PASSWORD=your_password \
  -v ./data:/app/data \
  siliconflow-api
```

或使用 docker-compose：
```bash
docker-compose up -d
```

## 使用说明

### 管理界面

访问 `http://localhost:7898` 进入管理界面：

1. **登录** - 使用配置的管理员账号密码登录
2. **密钥管理** - 导入、查看、刷新 API 密钥
3. **统计分析** - 查看使用统计和图表
4. **日志查看** - 查看详细的 API 调用日志

### API 调用示例

#### 1. OpenAI 格式对话
```bash
curl -X POST "http://localhost:7898/v1/chat/completions" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

#### 2. Anthropic Claude 对话
```bash
curl -X POST "http://localhost:7898/v1/messages" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'
```

#### 3. 文本嵌入
```bash
curl -X POST "http://localhost:7898/v1/embeddings" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "text-embedding-ada-002",
    "input": "Hello world"
  }'
```

#### 4. 图像生成
```bash
curl -X POST "http://localhost:7898/v1/image/generations" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dall-e-3",
    "prompt": "A beautiful sunset over the ocean",
    "image_size": "1024x1024"
  }'
```

#### 5. 文本转语音
```bash
curl -X POST "http://localhost:7898/v1/audio/speech" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello, this is a test.",
    "voice": "alloy"
  }'
```

#### 6. 视频生成
```bash
curl -X POST "http://localhost:7898/v1/video/generations" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "video-generation-model",
    "prompt": "A cat playing with a ball of yarn"
  }'
```

### 流式响应

所有支持流式响应的端点都可以通过设置 `"stream": true` 来启用：

```bash
curl -X POST "http://localhost:7898/v1/chat/completions" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

## 配置选项

### config.py 配置文件

```python
# API 访问密钥（可选）
API_KEY = None  # 设置后客户端需要提供此密钥才能访问 API

# 管理员登录凭据
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# 自动刷新设置
AUTO_REFRESH_INTERVAL = 3600  # 秒，设置为 0 禁用自动刷新

# 数据库设置
DB_PATH = "data/api_keys.db"  # 数据库文件路径
```

### 环境变量

所有配置都可以通过环境变量覆盖：

```bash
export API_KEY="your-protection-key"
export ADMIN_USERNAME="admin"
export ADMIN_PASSWORD="secure-password"
export AUTO_REFRESH_INTERVAL=1800
```

## 健康检查

访问 `/health` 端点获取系统状态：

```bash
curl http://localhost:7898/health
```

返回信息包括：
- 系统状态
- 数据库连接状态
- API 密钥数量
- 自动刷新任务状态

## 监控和日志

### 统计数据

- **实时统计** - `/api/stats/overview`
- **每日统计** - `/api/stats/daily`
- **每月统计** - `/api/stats/monthly`

### 日志管理

- 查看日志：`GET /logs?page=1&model=gpt-4o`
- 清空日志：`POST /clear_logs`
- 导出密钥：`GET /export_keys`

## 性能优化

1. **连接池** - 使用 aiohttp 连接池复用连接
2. **缓存机制** - API 密钥验证结果缓存
3. **异步处理** - 全异步架构提高并发性能
4. **智能选择** - 优先选择余额高的密钥
5. **批量处理** - 支持并发密钥验证

## 故障排除

### 常见问题

1. **密钥无效**
   - 检查 SiliconFlow API 密钥是否正确
   - 确认密钥有足够余额

2. **连接超时**
   - 检查网络连接
   - 调整超时设置

3. **数据库错误**
   - 确保有写入权限
   - 检查磁盘空间

### 日志位置

- 应用日志：控制台输出
- 数据库：`data/api_keys.db`
- 静态文件：`static/` 目录

## 许可证

MIT License

## 界面截图

### 登录页面
![登录页](./doc/login.png)

### 管理页面
![管理页](./doc/admin.png)

### 密钥管理页面
![密钥页](./doc/keylists.png)

## 贡献

欢迎提交 Issue 和 Pull Request！
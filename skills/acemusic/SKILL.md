---
name: acemusic
description: 使用 ACE-Step API 生成音乐、编辑歌曲、remix 音乐。支持文本生成音乐、歌词生成、音频续写、音频重绘等功能。当用户提到生成音乐、创作歌曲、音乐制作、remix、续写音频时使用此技能。
allowed-tools: Read, Write, Bash
---

# ACE Music 音乐生成技能

使用 ACE-Step V1.5 API 进行音乐生成和编辑。

## 输出文件

生成完成后，脚本会自动保存结果到项目根目录的 `acemusic_output` 文件夹（与 `.claude` 同级）：

```
project_root/
├── .claude/
│   └── skills/acemusic/...
├── acemusic_output/          # 输出目录
│   ├── <job_id>.json         # 完整任务结果 (JSON)
│   ├── <job_id>_1.mp3        # 第一个音频文件
│   ├── <job_id>_2.mp3        # 第二个音频文件 (如有 batch_size > 1)
│   └── ...
└── ...
```

## 配置说明

脚本使用 `scripts/config.json` 管理默认配置。

### 配置优先级规则

**重要**: 配置遵循以下优先级（从高到低）：

1. **命令行参数** > **config.json 默认值**
2. 用户指定的参数会**临时覆盖**默认配置，但**不会修改** config.json
3. 只有使用 `config --set` 命令时才会**永久修改** config.json

**示例**：
```bash
# config.json 中 thinking=true

# 使用默认配置（thinking=true）
./scripts/acemusic.sh generate "流行音乐"

# 临时覆盖（本次 thinking=false，但 config.json 不变）
./scripts/acemusic.sh generate "流行音乐" --no-thinking

# 永久修改默认配置
./scripts/acemusic.sh config --set generation.thinking false
```

### API 连接流程

1. **加载配置**: 读取 `scripts/config.json`（不存在则使用内置默认值）
2. **健康检查**: 请求 `/health` 端点验证服务可用性
3. **连接失败处理**: 提示用户输入正确的 API 地址并保存到 config.json

### 默认配置文件 (`scripts/config.json`)

```json
{
  "api_url": "http://127.0.0.1:8001",
  "generation": {
    "thinking": true,
    "use_format": true,
    "use_cot_caption": true,
    "use_cot_language": true,
    "batch_size": 1,
    "audio_format": "mp3",
    "vocal_language": "en"
  }
}
```

### 配置项说明

| 配置项 | 默认值 | 描述 |
|--------|--------|------|
| `api_url` | `http://127.0.0.1:8001` | API 服务地址 |
| `generation.thinking` | `true` | 启用 5Hz LM 模型（高质量模式） |
| `generation.use_format` | `true` | 使用 LM 增强 caption/lyrics |
| `generation.use_cot_caption` | `true` | 使用 CoT 增强 caption |
| `generation.use_cot_language` | `true` | 使用 CoT 增强语言检测 |
| `generation.audio_format` | `mp3` | 输出格式 |
| `generation.vocal_language` | `en` | 演唱语言 |

## 脚本使用方法

### 配置管理（永久修改 config.json）

```bash
# 查看所有配置
./scripts/acemusic.sh config

# 列出所有配置项及当前值
./scripts/acemusic.sh config --list

# 获取单个配置值
./scripts/acemusic.sh config --get generation.thinking

# 永久修改配置值（会写入 config.json）
./scripts/acemusic.sh config --set generation.thinking false
./scripts/acemusic.sh config --set api_url http://192.168.1.100:8001

# 重置为默认配置
./scripts/acemusic.sh config --reset
```

### 生成音乐（命令行参数临时覆盖，不修改 config.json）

支持两种生成模式：

**Caption 模式** - 直接指定音乐风格描述
```bash
./scripts/acemusic.sh generate "流行音乐，吉他伴奏"
./scripts/acemusic.sh generate -c "抒情流行" -l "[Verse] 歌词内容"
```

**Simple 模式** - 用简单描述，LM 自动生成 caption 和 lyrics
```bash
./scripts/acemusic.sh generate -d "一首关于春天的欢快歌曲"
./scripts/acemusic.sh generate -d "A love song for February"
```

**其他选项**
```bash
# 临时禁用 thinking 模式（本次生效，不修改配置文件）
./scripts/acemusic.sh generate "电子舞曲" --no-thinking

# 临时禁用 format 模式
./scripts/acemusic.sh generate "古典钢琴" --no-format

# 临时指定其他参数
./scripts/acemusic.sh generate "爵士乐" --steps 16 --guidance 8.0

# 随机生成
./scripts/acemusic.sh random

# 查询任务状态（已完成的任务会自动下载音频）
./scripts/acemusic.sh status <job_id>

# 列出可用模型
./scripts/acemusic.sh models

# 检查 API 健康状态
./scripts/acemusic.sh health
```

### Shell 脚本 (Linux/macOS，需要 curl + jq)

```bash
# 配置管理
./scripts/acemusic.sh config --list
./scripts/acemusic.sh config --set generation.thinking false

# Caption 模式（完成后自动保存结果和下载音频）
./scripts/acemusic.sh generate "流行音乐，吉他伴奏"
./scripts/acemusic.sh generate -c "抒情流行" -l "[Verse] 歌词内容"

# Simple 模式（LM 自动生成 caption/lyrics）
./scripts/acemusic.sh generate -d "一首关于春天的欢快歌曲"

# 随机生成
./scripts/acemusic.sh random

# 其他命令
./scripts/acemusic.sh status <job_id>
./scripts/acemusic.sh models
./scripts/acemusic.sh health
```

## 脚本依赖

| 脚本 | 依赖 | 平台 |
|------|------|------|
| `acemusic.sh` | curl | Linux/macOS/Git Bash |

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/v1/music/generate` | POST | 创建音乐生成任务 |
| `/v1/music/random` | POST | 创建随机采样任务 |
| `/v1/jobs/{job_id}` | GET | 查询任务状态和结果 |
| `/v1/models` | GET | 列出可用模型 |
| `/v1/audio?path={path}` | GET | 下载生成的音频文件 |

## 主要参数

### 基础参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `caption` | string | - | 音乐描述文本（Caption 模式） |
| `description` | string | - | 简单描述，LM 自动生成 caption/lyrics（Simple 模式） |
| `lyrics` | string | "" | 歌词内容 |
| `thinking` | bool | true | 启用 5Hz LM 模型（高质量） |
| `use_format` | bool | true | 使用 LM 增强输入 |
| `model` | string | - | 指定模型名称 |

### 音乐属性

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `bpm` | int | - | 节奏速度 |
| `key_scale` | string | - | 调性 (如 "C Major") |
| `time_signature` | string | - | 拍号 (如 "4/4") |
| `vocal_language` | string | "en" | 演唱语言 (en/zh) |
| `audio_duration` | float | - | 音频时长（秒） |

### 任务类型

| 参数值 | 描述 |
|--------|------|
| `text2music` | 文本生成音乐（默认） |
| `continuation` | 音频续写 |
| `repainting` | 音频重绘 |

## 任务状态

| 状态 | 描述 |
|------|------|
| `queued` | 排队等待中 |
| `running` | 正在生成 |
| `succeeded` | 生成成功 |
| `failed` | 生成失败 |

## 响应示例

### 创建任务响应

```json
{
  "job_id": "abc123-def456",
  "status": "queued",
  "queue_position": 1
}
```

### 任务完成响应

```json
{
  "job_id": "abc123-def456",
  "status": "succeeded",
  "result": {
    "audio_paths": ["/v1/audio?path=..."],
    "bpm": 120,
    "keyscale": "C Major",
    "duration": 60.0,
    "dit_model": "acestep-v15-turbo"
  }
}
```

## 注意事项

1. **配置优先级**: 命令行参数 > config.json 默认值。用户指定的参数临时生效，不修改配置文件
2. **修改默认配置**: 只有 `config --set` 命令才会永久修改 config.json
3. **默认高质量模式**: `thinking=true`, `use_format=true`，可通过 `--no-thinking`/`--no-format` 临时禁用
4. **异步任务**: 所有生成任务都是异步的，需要轮询 `/v1/jobs/{job_id}` 获取结果
5. **自动下载**: 任务完成后会自动保存 JSON 结果并下载音频文件到 `acemusic_output/` 目录

## 参考资源
- Shell 脚本: [scripts/acemusic.sh](scripts/acemusic.sh) (Linux/macOS/Git Bash)
- 默认配置: [scripts/config.json](scripts/config.json)
- 输出目录: 项目根目录下的 `acemusic_output/`

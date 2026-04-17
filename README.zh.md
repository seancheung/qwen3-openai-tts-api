# Qwen3-TTS OpenAI-TTS API

[English](./README.md) · **中文**

一个 [OpenAI TTS](https://platform.openai.com/docs/api-reference/audio/createSpeech) 兼容的 HTTP 服务，对 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)（阿里千问多语言 TTS 系列，支持 10 种语言，24 kHz 输出，12 Hz tokenizer）进行封装，将官方三种模型变体（Base / CustomVoice / VoiceDesign）统一到一个 FastAPI 服务后面。

## 特性

- **OpenAI TTS 兼容**：`POST /v1/audio/speech`，请求体格式与 OpenAI SDK 一致（对应 Base 模型的零样本音色克隆）
- **三种模型变体**（通过 `QWEN3_MODEL` 切换）：
  - `-Base` → 基于 `voice.wav` + `voice.txt` 的音色克隆（`/v1/audio/speech`）
  - `-CustomVoice` → 9 个官方预置音色（`/v1/audio/custom`）；1.7B 版本支持 `instruct` 情感/风格控制
  - `-VoiceDesign` → 用自然语言描述创造全新音色（`/v1/audio/design`）
- **10 种语言**：中文、英语、日语、韩语、德语、法语、俄语、葡萄牙语、西班牙语、意大利语
- **2 个镜像**：`cuda`（含 flash-attn 2，尽力安装）与 `cpu`
- **模型运行时下载**：不打包进镜像，挂载 HuggingFace 缓存即可复用
- **多种输出格式**：`mp3`、`opus`、`aac`、`flac`、`wav`、`pcm`

## 可用镜像

| 镜像 | 设备 |
|---|---|
| `ghcr.io/seancheung/qwen3-openai-tts-api:cuda-latest` | CUDA 12.4 + flash-attn 2（尽力安装） |
| `ghcr.io/seancheung/qwen3-openai-tts-api:latest`      | CPU |

镜像仅构建 `linux/amd64`。

## 快速开始

### 1. 选择模型变体

`QWEN3_MODEL` 决定容器里哪个端点可用：

| `QWEN3_MODEL` | 生效端点 | 是否需要 `voices/` |
|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base`（默认） | `POST /v1/audio/speech` | **是** — 需要 wav + txt 对 |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | `POST /v1/audio/speech` | **是** |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | `POST /v1/audio/custom` | 否 — 使用 9 个预置音色 |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | `POST /v1/audio/custom` | 否（不支持 `instruct`） |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `POST /v1/audio/design` | 否 — 通过 `instruct` 描述音色 |

与加载模型不匹配的端点会返回 HTTP 501。切换模型变体需要重启容器。

### 2. 准备音色目录（仅 Base 模式）

```
voices/
├── alice.wav     # 参考音频，单声道，16 kHz 以上，推荐 3-20 秒
├── alice.txt     # UTF-8 纯文本，内容为 alice.wav 中说出的原文
├── bob.wav
└── bob.txt
```

**规则**：必须同时存在同名的 `.wav` 和 `.txt` 才会被识别为有效音色；文件名（不含后缀）即音色 id；多余或缺对的文件会被忽略；文本必须为非空 UTF-8。

### 3. 运行容器

GPU 版本（推荐——1.7B 在 bf16 + flash-attn 下约需 10 GB 显存，不带 flash-attn 约需 14 GB）：

```bash
docker run --rm -p 8000:8000 --gpus all \
  -v $PWD/voices:/voices:ro \
  -v $PWD/hf_cache:/root/.cache/huggingface \
  ghcr.io/seancheung/qwen3-openai-tts-api:cuda-latest
```

CPU 版本（实际上只能跑小模型，而且非常慢）：

```bash
docker run --rm -p 8000:8000 \
  -v $PWD/voices:/voices:ro \
  -v $PWD/hf_cache:/root/.cache/huggingface \
  -e QWEN3_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  ghcr.io/seancheung/qwen3-openai-tts-api:latest
```

首次启动会从 HuggingFace 下载模型权重（1.7B 约 3.5 GB、0.6B 约 1.3 GB）。挂载 `/root/.cache/huggingface` 可让权重在容器重启后复用。

> **GPU 要求**：宿主机需安装 NVIDIA 驱动与 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。Windows 需 Docker Desktop + WSL2 + NVIDIA Windows 驱动。

### 4. docker-compose

参考 [`docker/docker-compose.example.yml`](./docker/docker-compose.example.yml)。

## API 用法

服务默认监听 `8000` 端口。

### GET `/v1/audio/voices`

列出所有可用音色。返回结构随当前加载的模型而不同。

```bash
curl -s http://localhost:8000/v1/audio/voices | jq
```

**Base**：列出 `voices/` 中发现的 wav+txt 对，含 `preview_url` 和 `prompt_text`：

```json
{
  "object": "list",
  "data": [
    {
      "id": "alice",
      "preview_url": "http://localhost:8000/v1/audio/voices/preview?id=alice",
      "prompt_text": "你好，这是一段参考音频。"
    }
  ]
}
```

**CustomVoice**：列出 9 个预置音色 id（不含 `preview_url` / `prompt_text`）：

```json
{
  "object": "list",
  "data": [
    { "id": "aiden" },
    { "id": "dylan" },
    { "id": "eric" },
    { "id": "ono_anna" },
    { "id": "ryan" },
    { "id": "serena" },
    { "id": "sohee" },
    { "id": "uncle_fu" },
    { "id": "vivian" }
  ]
}
```

**VoiceDesign**：返回空列表（无音色目录）：

```json
{ "object": "list", "data": [] }
```

### GET `/v1/audio/voices/preview?id={id}`

返回参考音频原文件（`audio/wav`）。**只在 Base 模式下可用**，其他模式返回 404。

### POST `/v1/audio/speech`（需 `-Base` 模型）

OpenAI TTS 兼容端点——零样本音色克隆。音色的 `.wav` 与 `.txt` 会一并作为 prompt 传给 Qwen3-TTS（ICL 模式）。

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-tts",
    "input": "你好世界，这是一段测试语音。",
    "voice": "alice",
    "response_format": "mp3"
  }' \
  -o out.mp3
```

请求字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `model` | string | 接受但忽略（为了与 OpenAI SDK 兼容） |
| `input` | string | 要合成的文本，最长 `MAX_INPUT_CHARS`（默认 8000） |
| `voice` | string | 音色 id，必须匹配 `/v1/audio/voices` 中的某一项 |
| `response_format` | string | `mp3`（默认） / `opus` / `aac` / `flac` / `wav` / `pcm` |
| `speed` | float | 为 OpenAI SDK 兼容保留，**实际忽略**——Qwen3-TTS 无语速控制 |
| `language` | string | `Auto`（默认） / `Chinese` / `English` / `Japanese` / `Korean` / `German` / `French` / `Russian` / `Portuguese` / `Spanish` / `Italian` |
| `x_vector_only_mode` | bool | `true` 时仅用 speaker embedding 做克隆、忽略 `.txt`；更快但质量稍低。默认 `false`。 |
| `max_new_tokens` | int | 可选生成长度上限（默认 `2048`） |
| `top_k` / `top_p` / `temperature` / `repetition_penalty` | — | 可选 HF 采样覆盖 |

输出音频为单声道 24 kHz；`pcm` 为裸 s16le 数据。

### POST `/v1/audio/custom`（需 `-CustomVoice` 模型）

使用预置音色，可选用自然语言 `instruct` 控制情感/风格。

```bash
curl -s http://localhost:8000/v1/audio/custom \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    "voice": "Vivian",
    "language": "Chinese",
    "instruct": "用特别愤怒的语气说",
    "response_format": "mp3"
  }' \
  -o out_custom.mp3
```

预置音色（运行时 `/v1/audio/voices` 返回的是权威列表）：

| 音色 | 母语 | 描述 |
|---|---|---|
| `Vivian` | 中文 | 明亮少女 |
| `Serena` | 中文 | 温暖轻柔少女 |
| `Uncle_Fu` | 中文 | 沉稳男声，低沉醇厚 |
| `Dylan` | 中文（北京） | 北京少年，音色清澈 |
| `Eric` | 中文（四川） | 成都青年，略带沙哑 |
| `Ryan` | 英语 | 动感男声，节奏强 |
| `Aiden` | 英语 | 阳光美音男声 |
| `Ono_Anna` | 日语 | 俏皮日系少女 |
| `Sohee` | 韩语 | 温暖韩系女声 |

> 0.6B CustomVoice 不支持 `instruct`，字段会被忽略。

### POST `/v1/audio/design`（需 `-VoiceDesign` 模型）

用自然语言描述生成全新音色，无需参考音频。

```bash
curl -s http://localhost:8000/v1/audio/design \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
    "instruct": "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显。",
    "language": "Chinese",
    "response_format": "mp3"
  }' \
  -o out_design.mp3
```

请求字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `input` | string | 要合成的文本 |
| `instruct` | string | **必填**。音色/风格的自然语言描述 |
| `response_format` | string | 同 `/speech` |
| `language` / `max_new_tokens` / 采样参数 | — | 语义同 `/speech` |

### 使用 OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-noop")

with client.audio.speech.with_streaming_response.create(
    model="qwen3-tts",
    voice="alice",
    input="你好世界",
    response_format="mp3",
) as resp:
    resp.stream_to_file("out.mp3")
```

扩展字段（`language`、`x_vector_only_mode`、`max_new_tokens`、采样参数）可通过 `extra_body={...}` 传入。注意 OpenAI SDK 只会调 `/v1/audio/speech`；用 CustomVoice / VoiceDesign 模型时请改用普通 HTTP 客户端。

### GET `/healthz`

返回加载的模型名、运行模式（`base` / `custom_voice` / `voice_design`）、设备、采样率与状态，用于健康检查。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `QWEN3_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | HuggingFace 仓库 id 或本地路径，决定当前激活哪个端点 |
| `QWEN3_DEVICE` | `auto` | `auto` 按 CUDA > CPU 优先级。也可强制 `cuda` / `cpu` |
| `QWEN3_CUDA_INDEX` | `0` | `cuda` / `auto` 时选择 `cuda:N` |
| `QWEN3_CACHE_DIR` | — | 加载模型前写入 `HF_HOME` |
| `QWEN3_DTYPE` | `auto` | `auto` → CUDA 用 `bfloat16`、CPU 用 `float32`。或强制 `bfloat16` / `float16` / `float32` |
| `QWEN3_ATTN_IMPLEMENTATION` | `auto` | `auto` → CUDA 用 `flash_attention_2`、CPU 用 `sdpa`；flash-attn 缺失会自动降级到 `sdpa`。也可强制 `flash_attention_2` / `sdpa` / `eager` |
| `QWEN3_DEFAULT_LANGUAGE` | `Auto` | 请求 `language` 字段的默认值 |
| `QWEN3_MAX_NEW_TOKENS` | `2048` | 默认生成长度上限 |
| `QWEN3_TOP_K` | `50` | 默认 top-k |
| `QWEN3_TOP_P` | `1.0` | 默认 top-p |
| `QWEN3_TEMPERATURE` | `0.9` | 默认采样温度 |
| `QWEN3_REPETITION_PENALTY` | `1.05` | 默认重复惩罚 |
| `QWEN3_CLONE_PROMPT_CACHE_SIZE` | `32` | `voice_clone_prompt` 缓存上限（仅 Base 模式）。设为 `0` 关闭缓存 |
| `QWEN3_VOICES_DIR` | `/voices` | 音色目录（仅 Base 模式使用） |
| `MAX_INPUT_CHARS` | `8000` | `input` 字段上限 |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | |
| `HOST` | `0.0.0.0` | |
| `PORT` | `8000` | |
| `LOG_LEVEL` | `info` | |

## 本地构建镜像

构建前需先初始化 submodule（workflow 已处理）。

```bash
git submodule update --init --recursive

# CUDA 镜像
docker buildx build -f docker/Dockerfile.cuda \
  -t qwen3-openai-tts-api:cuda .

# CPU 镜像
docker buildx build -f docker/Dockerfile.cpu \
  -t qwen3-openai-tts-api:cpu .
```

## 局限 / 注意事项

- **`speed` 字段是 no-op**：Qwen3-TTS 无原生语速控制，保留该字段只为让 OpenAI Python SDK 的默认请求体（总会带 `speed=1.0`）不被 422。若需调速请对返回音频做后处理。
- **一个容器只能跑一种模式**：`QWEN3_MODEL` 决定三个端点中哪一个生效，其他两个返回 501。如果同时需要克隆和预置音色，请起两个容器。
- **不做 OpenAI 固定音色名映射**（`alloy`、`echo`、`fable` 等）：Base 模式下在 `voices/` 放同名 `.wav` + `.txt` 即可；CustomVoice 模式下使用 9 个预置音色 id 之一。
- **并发**：单个 Qwen3-TTS 实例非线程安全，服务内部用 asyncio Lock 串行化。并发请求依赖横向扩容（多容器 + 负载均衡）。
- **长文本**：超过 `MAX_INPUT_CHARS`（默认 8000）返回 413。
- **不支持 HTTP 层流式返回**：生成完成后一次性返回。（Qwen3-TTS 本身支持 ~97 ms 的端到端流式，服务层暂未暴露。）
- **CUDA 镜像中的 flash-attn 为尽力安装**：若预编译 wheel 无法匹配当前 CUDA/torch/python 组合，构建仍会成功，运行时自动降级到 PyTorch 内置 SDPA kernel（略慢、显存更高）。可用 `QWEN3_ATTN_IMPLEMENTATION` 强制指定 kernel。
- **CPU 推理很慢**，尤其是 1.7B 模型；建议在 CPU 上使用 0.6B 变体。
- **无内置鉴权**：如需 token 访问控制，请在反向代理层（Nginx、Cloudflare 等）做。

## 目录结构

```
.
├── Qwen3-TTS/                  # 只读 submodule，不修改
├── app/                        # FastAPI 应用
│   ├── server.py
│   ├── engine.py               # 模型加载 + 推理
│   ├── voices.py               # 音色扫描
│   ├── audio.py                # 多格式编码
│   ├── config.py
│   └── schemas.py
├── docker/
│   ├── Dockerfile.cuda
│   ├── Dockerfile.cpu
│   ├── requirements.api.txt
│   ├── entrypoint.sh
│   └── docker-compose.example.yml
├── .github/workflows/
│   └── build-images.yml        # cuda + cpu 矩阵构建
└── README.md
```

## 致谢

基于 [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)（Apache 2.0）。

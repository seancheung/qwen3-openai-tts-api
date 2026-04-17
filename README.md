# Qwen3-TTS OpenAI-TTS API

**English** · [中文](./README.zh.md)

An [OpenAI TTS](https://platform.openai.com/docs/api-reference/audio/createSpeech)-compatible HTTP service wrapping [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — Alibaba's multilingual TTS family (10 languages, 24 kHz output, 12 Hz tokenizer) — exposing all three official model variants (Base / CustomVoice / VoiceDesign) behind a single FastAPI.

## Features

- **OpenAI TTS compatible** — `POST /v1/audio/speech` with the same request shape as the OpenAI SDK (for the Base model's zero-shot voice cloning)
- **Three model variants via `QWEN3_MODEL`**:
  - `-Base` → voice cloning from a `voice.wav` + `voice.txt` pair (`/v1/audio/speech`)
  - `-CustomVoice` → 9 preset speakers (`/v1/audio/custom`), optional `instruct` emotion/style control on the 1.7B checkpoint
  - `-VoiceDesign` → build a voice from a natural-language description (`/v1/audio/design`)
- **10 languages** — Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
- **2 images** — `cuda` (with optional flash-attn 2) and `cpu`
- **Model weights downloaded at runtime** — nothing heavy baked into the image; HuggingFace cache is mounted for reuse
- **Multiple output formats** — `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`

## Available images

| Image | Device |
|---|---|
| `ghcr.io/seancheung/qwen3-openai-tts-api:cuda-latest` | CUDA 12.4 + flash-attn 2 (best-effort) |
| `ghcr.io/seancheung/qwen3-openai-tts-api:latest`      | CPU |

Images are built for `linux/amd64`.

## Quick start

### 1. Pick a model variant

`QWEN3_MODEL` decides which endpoint is available in the container:

| `QWEN3_MODEL` | Active endpoint | Needs `voices/`? |
|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` *(default)* | `POST /v1/audio/speech` | **Yes** — wav + txt pair(s) |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | `POST /v1/audio/speech` | **Yes** |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | `POST /v1/audio/custom` | No — 9 preset speakers |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | `POST /v1/audio/custom` | No (no `instruct` support) |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `POST /v1/audio/design` | No — voice described via `instruct` |

Endpoints that don't match the loaded model return HTTP 501. Switching between variants needs a container restart.

### 2. Prepare the voices directory (Base only)

```
voices/
├── alice.wav     # reference audio, mono, 16 kHz+, ~3-20 s recommended
├── alice.txt     # UTF-8 text: the exact transcript of alice.wav
├── bob.wav
└── bob.txt
```

**Rules**: a voice is valid only when both files with the same stem exist; the stem is the voice id; unpaired or extra files are ignored. Text files must be non-empty UTF-8.

### 3. Run the container

GPU (recommended — 1.7B needs ~10 GB VRAM with bfloat16 + flash-attn, ~14 GB without):

```bash
docker run --rm -p 8000:8000 --gpus all \
  -v $PWD/voices:/voices:ro \
  -v $PWD/cache:/root/.cache \
  ghcr.io/seancheung/qwen3-openai-tts-api:cuda-latest
```

CPU (only small models are realistic; inference is slow):

```bash
docker run --rm -p 8000:8000 \
  -v $PWD/voices:/voices:ro \
  -v $PWD/cache:/root/.cache \
  -e QWEN3_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  ghcr.io/seancheung/qwen3-openai-tts-api:latest
```

Model weights (1.7B ≈ 3.5 GB, 0.6B ≈ 1.3 GB) are pulled from HuggingFace on first start. Mounting `/root/.cache` persists them across container restarts.

> **GPU prerequisites**: NVIDIA driver + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on Linux. On Windows use Docker Desktop + WSL2 + NVIDIA Windows driver; no host CUDA toolkit required.

### 4. docker-compose

See [`docker/docker-compose.example.yml`](./docker/docker-compose.example.yml).

## API usage

The service listens on port `8000` by default.

### GET `/v1/audio/voices`

List available voices. The response shape depends on the loaded model.

```bash
curl -s http://localhost:8000/v1/audio/voices | jq
```

**Base** — lists wav+txt pairs discovered in `voices/`, each with a `preview_url` and `prompt_text`:

```json
{
  "object": "list",
  "data": [
    {
      "id": "alice",
      "preview_url": "http://localhost:8000/v1/audio/voices/preview?id=alice",
      "prompt_text": "Hello, this is a reference audio sample."
    }
  ]
}
```

**CustomVoice** — lists the 9 preset speaker ids (`preview_url` / `prompt_text` are absent):

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

**VoiceDesign** — empty list (no voice catalog):

```json
{ "object": "list", "data": [] }
```

### GET `/v1/audio/voices/preview?id={id}`

Returns the raw reference wav (`audio/wav`). Only available in Base mode; returns 404 otherwise.

### POST `/v1/audio/speech` *(requires `-Base` model)*

OpenAI TTS-compatible endpoint — zero-shot voice cloning. The voice's `.wav` and `.txt` are both fed to Qwen3-TTS (ICL mode).

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-tts",
    "input": "Hello world, this is a test.",
    "voice": "alice",
    "response_format": "mp3"
  }' \
  -o out.mp3
```

Request fields:

| Field | Type | Description |
|---|---|---|
| `model` | string | Accepted but ignored (for OpenAI SDK compatibility) |
| `input` | string | Text to synthesize, up to `MAX_INPUT_CHARS` (default 8000) |
| `voice` | string | Voice id — must match an entry from `/v1/audio/voices` |
| `response_format` | string | `mp3` (default) / `opus` / `aac` / `flac` / `wav` / `pcm` |
| `speed` | float | Accepted for OpenAI SDK compatibility but **ignored** — Qwen3-TTS has no speed control |
| `language` | string | `Auto` (default) / `Chinese` / `English` / `Japanese` / `Korean` / `German` / `French` / `Russian` / `Portuguese` / `Spanish` / `Italian` |
| `x_vector_only_mode` | bool | If `true`, clone using speaker embedding only — ignores `.txt`. Faster, lower quality. Default `false`. |
| `max_new_tokens` | int | Optional generation cap (default `2048`) |
| `top_k` / `top_p` / `temperature` / `repetition_penalty` | — | Optional HF-style sampling overrides |

Output audio is mono 24 kHz; `pcm` is raw s16le.

### POST `/v1/audio/custom` *(requires `-CustomVoice` model)*

Use a preset speaker, optionally shaped by a natural-language `instruct`.

```bash
curl -s http://localhost:8000/v1/audio/custom \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "She said she would be here by noon.",
    "voice": "Ryan",
    "language": "English",
    "instruct": "Speak happily.",
    "response_format": "mp3"
  }' \
  -o out_custom.mp3
```

Preset speakers (check `/v1/audio/voices` at runtime for the authoritative list):

| Speaker | Native language | Description |
|---|---|---|
| `Vivian` | Chinese | Bright young female |
| `Serena` | Chinese | Warm, gentle young female |
| `Uncle_Fu` | Chinese | Seasoned male, low mellow timbre |
| `Dylan` | Chinese (Beijing) | Youthful Beijing male |
| `Eric` | Chinese (Sichuan) | Lively Chengdu male |
| `Ryan` | English | Dynamic male, strong rhythm |
| `Aiden` | English | Sunny American male |
| `Ono_Anna` | Japanese | Playful Japanese female |
| `Sohee` | Korean | Warm Korean female |

> The `instruct` field is **ignored** by the 0.6B CustomVoice checkpoint.

### POST `/v1/audio/design` *(requires `-VoiceDesign` model)*

Build a voice from a natural-language description, no reference audio needed.

```bash
curl -s http://localhost:8000/v1/audio/design \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "It is in the top drawer... wait, it is empty?",
    "instruct": "Speak in an incredulous tone, with a hint of panic.",
    "language": "English",
    "response_format": "mp3"
  }' \
  -o out_design.mp3
```

Request fields:

| Field | Type | Description |
|---|---|---|
| `input` | string | Text to synthesize |
| `instruct` | string | **Required.** Voice/style description |
| `response_format` | string | Same as `/speech` |
| `language` / `max_new_tokens` / sampling params | — | Same semantics as `/speech` |

### Using the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-noop")

with client.audio.speech.with_streaming_response.create(
    model="qwen3-tts",
    voice="alice",
    input="Hello world",
    response_format="mp3",
) as resp:
    resp.stream_to_file("out.mp3")
```

Extensions (`language`, `x_vector_only_mode`, `max_new_tokens`, sampling params) can be passed via `extra_body={...}`. Note that the OpenAI SDK only targets `/v1/audio/speech` — for CustomVoice / VoiceDesign models you need a plain HTTP client.

### GET `/healthz`

Returns the loaded model name, detected mode (`base` / `custom_voice` / `voice_design`), device, sample rate, and status.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `QWEN3_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | HuggingFace repo id or local path. Picks which endpoint is active. |
| `QWEN3_DEVICE` | `auto` | `auto` → CUDA if available else CPU. Or force `cuda` / `cpu`. |
| `QWEN3_CUDA_INDEX` | `0` | Selects `cuda:N` when device is `cuda` or `auto` |
| `QWEN3_CACHE_DIR` | — | Sets `HF_HOME` before model load |
| `QWEN3_DTYPE` | `auto` | `auto` → `bfloat16` on CUDA, `float32` on CPU. Or `bfloat16` / `float16` / `float32`. |
| `QWEN3_ATTN_IMPLEMENTATION` | `auto` | `auto` → `flash_attention_2` on CUDA, `sdpa` on CPU. Falls back to `sdpa` if flash-attn import fails. Or force `flash_attention_2` / `sdpa` / `eager`. |
| `QWEN3_DEFAULT_LANGUAGE` | `Auto` | Default value of the `language` request field |
| `QWEN3_MAX_NEW_TOKENS` | `2048` | Default generation cap |
| `QWEN3_TOP_K` | `50` | Default top-k sampling |
| `QWEN3_TOP_P` | `1.0` | Default top-p sampling |
| `QWEN3_TEMPERATURE` | `0.9` | Default sampling temperature |
| `QWEN3_REPETITION_PENALTY` | `1.05` | Default repetition penalty |
| `QWEN3_CLONE_PROMPT_CACHE_SIZE` | `32` | Max cached `voice_clone_prompt` entries (Base mode only). `0` disables caching. |
| `QWEN3_VOICES_DIR` | `/voices` | Voices directory (Base mode only) |
| `MAX_INPUT_CHARS` | `8000` | Upper bound for the `input` field |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | |
| `HOST` | `0.0.0.0` | |
| `PORT` | `8000` | |
| `LOG_LEVEL` | `info` | |

## Building images locally

Initialize the submodule first (the workflow does this automatically).

```bash
git submodule update --init --recursive

# CUDA image
docker buildx build -f docker/Dockerfile.cuda \
  -t qwen3-openai-tts-api:cuda .

# CPU image
docker buildx build -f docker/Dockerfile.cpu \
  -t qwen3-openai-tts-api:cpu .
```

## Caveats

- **`speed` is a no-op.** Qwen3-TTS has no native speed control, but the field is kept so that OpenAI's Python SDK default request body (which always sends `speed=1.0`) does not 422. If you need tempo control, post-process the returned audio.
- **One mode per container.** `QWEN3_MODEL` determines which of the three endpoints works; the other two return 501. Run a second container if you need both cloning and preset voices.
- **No built-in OpenAI voice names** (`alloy`, `echo`, `fable`, …). For Base, drop `alloy.wav` + `alloy.txt` into `voices/`; for CustomVoice, use one of the 9 preset speaker ids.
- **Concurrency**: a single Qwen3-TTS instance is not thread-safe; the service serializes inference with an asyncio Lock. Scale out by running more containers behind a load balancer.
- **Long text**: requests whose `input` exceeds `MAX_INPUT_CHARS` (default 8000) return 413.
- **Streaming is not supported** on the HTTP layer — the endpoint returns the complete audio once generation finishes. (Qwen3-TTS supports ~97 ms end-to-end streaming internally; exposing it here is future work.)
- **flash-attn is best-effort in the CUDA image.** If the prebuilt wheel cannot be installed at image build time, the engine falls back to PyTorch's built-in SDPA kernel at runtime (slightly slower, higher VRAM). Force a specific kernel via `QWEN3_ATTN_IMPLEMENTATION`.
- **CPU inference is slow**, especially for the 1.7B models. Use 0.6B variants or CUDA whenever possible.
- **No built-in auth** — deploy behind a reverse proxy (Nginx, Cloudflare, etc.) if you need token-based access control.

## Project layout

```
.
├── Qwen3-TTS/                  # read-only submodule, never modified
├── app/                        # FastAPI application
│   ├── server.py
│   ├── engine.py               # model loading + inference
│   ├── voices.py               # voices directory scanner
│   ├── audio.py                # multi-format encoder
│   ├── config.py
│   └── schemas.py
├── docker/
│   ├── Dockerfile.cuda
│   ├── Dockerfile.cpu
│   ├── requirements.api.txt
│   ├── entrypoint.sh
│   └── docker-compose.example.yml
├── .github/workflows/
│   └── build-images.yml        # cuda + cpu matrix build
└── README.md
```

## Acknowledgements

Built on top of [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache 2.0).

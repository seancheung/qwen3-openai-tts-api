from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response

from .audio import CONTENT_TYPES, encode
from .config import get_settings
from .engine import TTSEngine
from .schemas import (
    CustomVoiceRequest,
    DesignRequest,
    HealthResponse,
    SpeechRequest,
    VoiceInfo,
    VoiceList,
)
from .voices import VoiceCatalog

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    app.state.settings = settings
    app.state.catalog = VoiceCatalog(settings.voices_path)
    app.state.engine = None
    try:
        app.state.engine = TTSEngine(settings)
    except Exception:
        log.exception("failed to load Qwen3-TTS model")
        raise
    yield


app = FastAPI(
    title="Qwen3-TTS OpenAI-TTS API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse)
async def healthz(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    engine: TTSEngine | None = request.app.state.engine
    if engine is None:
        return HealthResponse(status="loading", model=settings.qwen3_model)
    return HealthResponse(
        status="ok",
        model=settings.qwen3_model,
        mode=engine.mode,
        device=engine.device,
        sample_rate=engine.sample_rate,
    )


@app.get("/v1/audio/voices", response_model=VoiceList)
async def list_voices(request: Request) -> VoiceList:
    engine: TTSEngine = request.app.state.engine
    catalog: VoiceCatalog = request.app.state.catalog
    base = str(request.base_url).rstrip("/")

    if engine.mode == "base":
        voices = catalog.scan()
        data = [
            VoiceInfo(
                id=v.id,
                preview_url=f"{base}/v1/audio/voices/preview?id={quote(v.id, safe='')}",
                prompt_text=v.prompt_text,
            )
            for v in voices.values()
        ]
        return VoiceList(data=data)

    if engine.mode == "custom_voice":
        speakers = engine.supported_speakers or []
        data = [VoiceInfo(id=spk) for spk in speakers]
        return VoiceList(data=data)

    # voice_design: no voice catalog — the voice is produced from an instruct string.
    return VoiceList(data=[])


@app.get("/v1/audio/voices/preview")
async def preview_voice(id: str, request: Request):
    engine: TTSEngine = request.app.state.engine
    if engine.mode != "base":
        raise HTTPException(
            status_code=404,
            detail=f"voice previews are only available in 'base' mode (current mode: '{engine.mode}')",
        )
    catalog: VoiceCatalog = request.app.state.catalog
    voice = catalog.get(id)
    if voice is None:
        raise HTTPException(status_code=404, detail=f"voice '{id}' not found")
    return FileResponse(
        path=str(voice.wav_path), media_type="audio/wav", filename=f"{id}.wav"
    )


def _validate_text(raw: str, max_chars: int) -> str:
    text = (raw or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="input is empty")
    if len(text) > max_chars:
        raise HTTPException(
            status_code=413, detail=f"input exceeds {max_chars} chars",
        )
    return text


def _validate_format(fmt: str) -> None:
    if fmt not in CONTENT_TYPES:
        raise HTTPException(
            status_code=422, detail=f"unsupported response_format: {fmt}",
        )


def _require_mode(engine: TTSEngine, expected: str) -> None:
    if engine.mode != expected:
        raise HTTPException(
            status_code=501,
            detail=(
                f"this endpoint requires a '{expected}' model, "
                f"but QWEN3_MODEL is currently loaded as '{engine.mode}'"
            ),
        )


def _encode_response(samples, sample_rate: int, fmt: str) -> Response:
    try:
        audio_bytes, content_type = encode(samples, sample_rate, fmt)
    except Exception as e:
        log.exception("encoding failed")
        raise HTTPException(status_code=500, detail=f"encoding failed: {e}") from e
    return Response(content=audio_bytes, media_type=content_type)


@app.post("/v1/audio/speech")
async def create_speech(body: SpeechRequest, request: Request):
    settings = request.app.state.settings
    engine: TTSEngine = request.app.state.engine
    catalog: VoiceCatalog = request.app.state.catalog

    _require_mode(engine, "base")
    text = _validate_text(body.input, settings.max_input_chars)
    _validate_format(body.response_format)

    voice = catalog.get(body.voice)
    if voice is None:
        raise HTTPException(status_code=404, detail=f"voice '{body.voice}' not found")

    try:
        samples = await engine.synthesize_clone(
            text,
            wav_path=str(voice.wav_path),
            mtime=voice.mtime,
            ref_text=voice.prompt_text,
            language=body.language,
            x_vector_only_mode=body.x_vector_only_mode,
            max_new_tokens=body.max_new_tokens,
            top_k=body.top_k,
            top_p=body.top_p,
            temperature=body.temperature,
            repetition_penalty=body.repetition_penalty,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception("inference failed")
        raise HTTPException(status_code=500, detail=f"inference failed: {e}") from e

    return _encode_response(samples, engine.sample_rate, body.response_format)


@app.post("/v1/audio/custom")
async def create_custom(body: CustomVoiceRequest, request: Request):
    settings = request.app.state.settings
    engine: TTSEngine = request.app.state.engine

    _require_mode(engine, "custom_voice")
    text = _validate_text(body.input, settings.max_input_chars)
    _validate_format(body.response_format)

    speakers = engine.supported_speakers or []
    # Validate case-insensitively; Qwen3's validator also lowercases.
    if speakers and body.voice.lower() not in {s.lower() for s in speakers}:
        raise HTTPException(
            status_code=404,
            detail=f"voice '{body.voice}' not found; supported: {speakers}",
        )

    try:
        samples = await engine.synthesize_custom(
            text,
            speaker=body.voice,
            language=body.language,
            instruct=body.instruct,
            max_new_tokens=body.max_new_tokens,
            top_k=body.top_k,
            top_p=body.top_p,
            temperature=body.temperature,
            repetition_penalty=body.repetition_penalty,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception("inference failed")
        raise HTTPException(status_code=500, detail=f"inference failed: {e}") from e

    return _encode_response(samples, engine.sample_rate, body.response_format)


@app.post("/v1/audio/design")
async def create_design(body: DesignRequest, request: Request):
    settings = request.app.state.settings
    engine: TTSEngine = request.app.state.engine

    _require_mode(engine, "voice_design")
    text = _validate_text(body.input, settings.max_input_chars)
    _validate_format(body.response_format)

    instruct = (body.instruct or "").strip()
    if not instruct:
        raise HTTPException(status_code=422, detail="instruct is empty")

    try:
        samples = await engine.synthesize_design(
            text,
            instruct=instruct,
            language=body.language,
            max_new_tokens=body.max_new_tokens,
            top_k=body.top_k,
            top_p=body.top_p,
            temperature=body.temperature,
            repetition_penalty=body.repetition_penalty,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception("inference failed")
        raise HTTPException(status_code=500, detail=f"inference failed: {e}") from e

    return _encode_response(samples, engine.sample_rate, body.response_format)

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


ResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


class _GenParams(BaseModel):
    """Shared optional generation-parameter overrides for all three endpoints."""

    language: Optional[str] = Field(
        default=None,
        description="Synthesis language (e.g. 'Chinese', 'English', 'Auto'). Defaults to QWEN3_DEFAULT_LANGUAGE.",
    )
    max_new_tokens: Optional[int] = Field(default=None, ge=1, le=8192)
    top_k: Optional[int] = Field(default=None, ge=0, le=500)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    repetition_penalty: Optional[float] = Field(default=None, ge=0.5, le=2.0)


class SpeechRequest(_GenParams):
    """OpenAI-compatible `/v1/audio/speech` request — Base model voice cloning."""

    model: Optional[str] = Field(default=None, description="Accepted for OpenAI compatibility; ignored.")
    input: str = Field(..., description="Text to synthesize.")
    voice: str = Field(..., description="Voice id matching a file pair in the voices directory.")
    response_format: ResponseFormat = Field(default="mp3")
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Accepted for OpenAI compatibility; ignored (Qwen3-TTS has no speed control).",
    )
    x_vector_only_mode: bool = Field(
        default=False,
        description="If true, use speaker-embedding-only cloning (ignores the .txt transcript).",
    )


class CustomVoiceRequest(_GenParams):
    """`/v1/audio/custom` request — CustomVoice model with preset speakers."""

    input: str = Field(..., description="Text to synthesize.")
    voice: str = Field(..., description="Preset speaker id (see `/v1/audio/voices`).")
    response_format: ResponseFormat = Field(default="mp3")
    instruct: Optional[str] = Field(
        default=None,
        description="Optional natural-language emotion/style instruction. Ignored on 0.6B CustomVoice.",
    )


class DesignRequest(_GenParams):
    """`/v1/audio/design` request — VoiceDesign model, describe voice in natural language."""

    input: str = Field(..., description="Text to synthesize.")
    instruct: str = Field(
        ...,
        description="Required voice description, e.g. 'young woman, warm, soft'.",
    )
    response_format: ResponseFormat = Field(default="mp3")


class VoiceInfo(BaseModel):
    id: str
    preview_url: Optional[str] = None
    prompt_text: Optional[str] = None


class VoiceList(BaseModel):
    object: Literal["list"] = "list"
    data: list[VoiceInfo]


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    model: str
    mode: Optional[str] = None
    device: Optional[str] = None
    sample_rate: Optional[int] = None

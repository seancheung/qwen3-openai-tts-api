from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False, extra="ignore")

    qwen3_model: str = Field(default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    qwen3_device: Literal["auto", "cuda", "cpu"] = Field(default="auto")
    qwen3_cuda_index: int = Field(default=0)
    qwen3_cache_dir: Optional[str] = Field(default=None)
    qwen3_dtype: Literal["auto", "bfloat16", "float16", "float32"] = Field(default="auto")
    qwen3_attn_implementation: Literal[
        "auto", "flash_attention_2", "sdpa", "eager"
    ] = Field(default="auto")

    qwen3_default_language: str = Field(default="Auto")
    qwen3_max_new_tokens: int = Field(default=2048, ge=1, le=8192)
    qwen3_top_k: int = Field(default=50, ge=0, le=500)
    qwen3_top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    qwen3_temperature: float = Field(default=0.9, ge=0.0, le=2.0)
    qwen3_repetition_penalty: float = Field(default=1.05, ge=0.5, le=2.0)

    qwen3_clone_prompt_cache_size: int = Field(default=32, ge=0, le=1024)
    qwen3_voices_dir: str = Field(default="/voices")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="info")
    max_input_chars: int = Field(default=8000)
    default_response_format: Literal[
        "mp3", "opus", "aac", "flac", "wav", "pcm"
    ] = Field(default="mp3")

    @property
    def voices_path(self) -> Path:
        return Path(self.qwen3_voices_dir)

    @property
    def resolved_device(self) -> str:
        import torch

        if self.qwen3_device == "auto":
            if torch.cuda.is_available():
                return f"cuda:{self.qwen3_cuda_index}"
            return "cpu"
        if self.qwen3_device == "cuda":
            return f"cuda:{self.qwen3_cuda_index}"
        return self.qwen3_device

    @property
    def resolved_dtype(self):
        import torch

        device = self.resolved_device
        if self.qwen3_dtype == "auto":
            return torch.bfloat16 if device.startswith("cuda") else torch.float32
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.qwen3_dtype]

    @property
    def resolved_attn(self) -> str:
        device = self.resolved_device
        if self.qwen3_attn_implementation == "auto":
            return "flash_attention_2" if device.startswith("cuda") else "sdpa"
        return self.qwen3_attn_implementation


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

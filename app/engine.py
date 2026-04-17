from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel

log = logging.getLogger(__name__)


# 12Hz tokenizer always decodes to 24 kHz. Kept as a constant because the
# underlying `speech_tokenizer.decode(...)` only returns the sample rate after
# the first call; we need a value for `/healthz` before any inference happens.
QWEN3_12HZ_SAMPLE_RATE = 24000


class TTSEngine:
    def __init__(self, settings):
        self.settings = settings

        device = settings.resolved_device
        dtype = settings.resolved_dtype
        attn = settings.resolved_attn

        if settings.qwen3_cache_dir:
            os.environ.setdefault("HF_HOME", settings.qwen3_cache_dir)

        log.info(
            "loading Qwen3-TTS model=%s device=%s dtype=%s attn=%s",
            settings.qwen3_model, device, dtype, attn,
        )

        try:
            self.model = Qwen3TTSModel.from_pretrained(
                settings.qwen3_model,
                device_map=device,
                dtype=dtype,
                attn_implementation=attn,
            )
        except ImportError as e:
            if attn == "flash_attention_2":
                log.warning(
                    "flash_attention_2 not available (%s); falling back to sdpa", e,
                )
                self.model = Qwen3TTSModel.from_pretrained(
                    settings.qwen3_model,
                    device_map=device,
                    dtype=dtype,
                    attn_implementation="sdpa",
                )
                attn = "sdpa"
            else:
                raise

        self.device = device
        self.attn_implementation = attn
        self.mode: str = self.model.model.tts_model_type
        self.sample_rate: int = QWEN3_12HZ_SAMPLE_RATE
        self.supported_speakers: Optional[List[str]] = self.model.get_supported_speakers()
        self.supported_languages: Optional[List[str]] = self.model.get_supported_languages()

        self._lock = asyncio.Lock()
        self._clone_prompt_cache: "OrderedDict[Tuple[str, float, str, bool], Any]" = OrderedDict()
        self._clone_prompt_cache_size = settings.qwen3_clone_prompt_cache_size

        log.info(
            "Qwen3-TTS ready: mode=%s speakers=%s",
            self.mode,
            self.supported_speakers,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _gen_kwargs(
        self,
        *,
        max_new_tokens: Optional[int],
        top_k: Optional[int],
        top_p: Optional[float],
        temperature: Optional[float],
        repetition_penalty: Optional[float],
    ) -> Dict[str, Any]:
        s = self.settings
        return dict(
            max_new_tokens=max_new_tokens if max_new_tokens is not None else s.qwen3_max_new_tokens,
            top_k=top_k if top_k is not None else s.qwen3_top_k,
            top_p=top_p if top_p is not None else s.qwen3_top_p,
            temperature=temperature if temperature is not None else s.qwen3_temperature,
            repetition_penalty=repetition_penalty if repetition_penalty is not None else s.qwen3_repetition_penalty,
        )

    def _resolve_language(self, language: Optional[str]) -> str:
        return language if language else self.settings.qwen3_default_language

    @staticmethod
    def _pick_first(wavs) -> np.ndarray:
        if not wavs:
            raise RuntimeError("Qwen3-TTS returned no audio")
        arr = wavs[0]
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        return np.ascontiguousarray(np.asarray(arr).astype(np.float32, copy=False))

    def _get_or_build_clone_prompt(
        self,
        *,
        wav_path: str,
        mtime: float,
        ref_text: str,
        x_vector_only_mode: bool,
    ):
        key = (wav_path, mtime, ref_text if not x_vector_only_mode else "", bool(x_vector_only_mode))
        cached = self._clone_prompt_cache.get(key)
        if cached is not None:
            self._clone_prompt_cache.move_to_end(key)
            return cached

        prompt = self.model.create_voice_clone_prompt(
            ref_audio=wav_path,
            ref_text=ref_text if not x_vector_only_mode else None,
            x_vector_only_mode=x_vector_only_mode,
        )

        self._clone_prompt_cache[key] = prompt
        if self._clone_prompt_cache_size > 0:
            while len(self._clone_prompt_cache) > self._clone_prompt_cache_size:
                self._clone_prompt_cache.popitem(last=False)
        return prompt

    # ------------------------------------------------------------------
    # inference entrypoints
    # ------------------------------------------------------------------
    async def synthesize_clone(
        self,
        text: str,
        *,
        wav_path: str,
        mtime: float,
        ref_text: str,
        language: Optional[str] = None,
        x_vector_only_mode: bool = False,
        max_new_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> np.ndarray:
        if self.mode != "base":
            raise RuntimeError(f"synthesize_clone requires Base model, current mode is '{self.mode}'")

        gen = self._gen_kwargs(
            max_new_tokens=max_new_tokens, top_k=top_k, top_p=top_p,
            temperature=temperature, repetition_penalty=repetition_penalty,
        )
        lang = self._resolve_language(language)

        async with self._lock:
            prompt = await asyncio.to_thread(
                self._get_or_build_clone_prompt,
                wav_path=wav_path,
                mtime=mtime,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
            )
            wavs, _ = await asyncio.to_thread(
                self.model.generate_voice_clone,
                text=text,
                language=lang,
                voice_clone_prompt=prompt,
                **gen,
            )
        return self._pick_first(wavs)

    async def synthesize_custom(
        self,
        text: str,
        *,
        speaker: str,
        language: Optional[str] = None,
        instruct: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> np.ndarray:
        if self.mode != "custom_voice":
            raise RuntimeError(f"synthesize_custom requires CustomVoice model, current mode is '{self.mode}'")

        gen = self._gen_kwargs(
            max_new_tokens=max_new_tokens, top_k=top_k, top_p=top_p,
            temperature=temperature, repetition_penalty=repetition_penalty,
        )
        lang = self._resolve_language(language)

        async with self._lock:
            wavs, _ = await asyncio.to_thread(
                self.model.generate_custom_voice,
                text=text,
                speaker=speaker,
                language=lang,
                instruct=instruct or "",
                **gen,
            )
        return self._pick_first(wavs)

    async def synthesize_design(
        self,
        text: str,
        *,
        instruct: str,
        language: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> np.ndarray:
        if self.mode != "voice_design":
            raise RuntimeError(f"synthesize_design requires VoiceDesign model, current mode is '{self.mode}'")

        gen = self._gen_kwargs(
            max_new_tokens=max_new_tokens, top_k=top_k, top_p=top_p,
            temperature=temperature, repetition_penalty=repetition_penalty,
        )
        lang = self._resolve_language(language)

        async with self._lock:
            wavs, _ = await asyncio.to_thread(
                self.model.generate_voice_design,
                text=text,
                instruct=instruct,
                language=lang,
                **gen,
            )
        return self._pick_first(wavs)

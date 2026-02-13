#!/usr/bin/env python3
"""Resident local inference service for GhostType.

Exposes three routes:
- /dictate
- /ask
- /translate
"""

from __future__ import annotations

import argparse
from collections import deque
import gc
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
try:
    from mlx_lm import load, stream_generate
    import mlx_whisper
except ImportError:  # pragma: no cover - MLX requires Apple Silicon
    load = None  # type: ignore[assignment]
    stream_generate = None  # type: ignore[assignment]
    mlx_whisper = None  # type: ignore[assignment]
from pydantic import BaseModel, Field

from audio_io import WavFormatError, load_wav_pcm16_mono

try:
    from duckduckgo_search import DDGS
except Exception:  # pragma: no cover - optional runtime import
    DDGS = None

try:
    import webrtcvad
except Exception:  # pragma: no cover - optional runtime import
    webrtcvad = None

try:
    from webrtc_audio_processing import AudioProcessingModule as WebRTCAudioProcessingModule
except Exception:  # pragma: no cover - optional runtime import
    WebRTCAudioProcessingModule = None

try:
    from enhancement_engine import (
        EnhancementEngine,
        EnhancementV2Config,
        LimiterConfig as EnhancementLimiterConfig,
        TargetsConfig as EnhancementTargetsConfig,
        VADConfig as EnhancementVADConfig,
        probe_enhancement_plugins,
    )
except Exception:  # pragma: no cover - fallback when module is unavailable
    EnhancementEngine = None
    EnhancementV2Config = None
    EnhancementLimiterConfig = None
    EnhancementTargetsConfig = None
    EnhancementVADConfig = None

    def probe_enhancement_plugins() -> dict[str, bool]:
        return {
            "pyloudnorm": False,
            "webrtc_apm": bool(WebRTCAudioProcessingModule is not None),
            "rnnoise": False,
            "deepfilternet": False,
            "speexdsp": False,
            "ffmpeg": False,
        }


# 全局变量，用于保持大模型在内存中常驻
LLM_REPO = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
ASR_REPO = "mlx-community/whisper-small-mlx"

llm_model = None
llm_tokenizer = None


def initialize_models():
    """
    启动服务时的初始化函数。
    此函数会触发 MLX 自动从 Hugging Face 下载模型权重并加载到 Mac 的统一内存中。
    """
    global llm_model, llm_tokenizer

    print(f"⏳ 正在检查并下载 LLM 模型: {LLM_REPO} ...")
    # 这行代码会在首次运行时自动下载 4-bit 量化模型，之后直接从本地缓存读取
    llm_model, llm_tokenizer = load(LLM_REPO)
    print("✅ LLM 模型加载成功并驻留内存！")

    print(f"⏳ 正在检查并下载 ASR 模型: {ASR_REPO} ...")
    # MLX Whisper 默认在执行 transcribe 时才会下载模型。
    # 为了强制它在启动时就下载并缓存，我们传入一段 1 秒的纯静音空白音频做一次 Dummy 推理。
    dummy_audio = np.zeros(16000, dtype=np.float32)
    _ = mlx_whisper.transcribe(
        dummy_audio,
        path_or_hf_repo=ASR_REPO
    )
    print("✅ ASR 模型预热及加载成功！")


DEFAULT_ASR_MODEL = ASR_REPO
DEFAULT_LLM_MODEL = LLM_REPO


class DictateRequest(BaseModel):
    audio_path: str
    inference_audio_profile: str = "standard"
    ui_language: str = "English"
    output_language: str = "Auto"
    asr_model: str | None = None
    llm_model: str | None = None
    audio_enhancement_enabled: bool = True
    audio_enhancement_mode: str = "webrtc"
    low_volume_boost: str = "medium"
    noise_suppression_level: str = "moderate"
    anti_cutoff_pause_ms: int = Field(default=350, ge=200, le=1200)
    audio_debug_enabled: bool = False
    enhancement_version: str = "v2"
    enhancement_mode: str = "fast_dsp"
    ns_engine: str = "webrtc"
    loudness_strategy: str = "dynaudnorm"
    dynamics: str = "upward_comp"
    limiter: dict[str, Any] = Field(default_factory=dict)
    targets: dict[str, Any] = Field(default_factory=dict)
    vad: dict[str, Any] = Field(default_factory=dict)
    system_prompt: str | None = None
    max_tokens: int = 350


class AskRequest(BaseModel):
    audio_path: str
    inference_audio_profile: str = "standard"
    selected_text: str = ""
    ui_language: str = "English"
    output_language: str = "Auto"
    asr_model: str | None = None
    llm_model: str | None = None
    audio_enhancement_enabled: bool = True
    audio_enhancement_mode: str = "webrtc"
    low_volume_boost: str = "medium"
    noise_suppression_level: str = "moderate"
    anti_cutoff_pause_ms: int = Field(default=350, ge=200, le=1200)
    audio_debug_enabled: bool = False
    enhancement_version: str = "v2"
    enhancement_mode: str = "fast_dsp"
    ns_engine: str = "webrtc"
    loudness_strategy: str = "dynaudnorm"
    dynamics: str = "upward_comp"
    limiter: dict[str, Any] = Field(default_factory=dict)
    targets: dict[str, Any] = Field(default_factory=dict)
    vad: dict[str, Any] = Field(default_factory=dict)
    system_prompt: str | None = None
    web_search_enabled: bool = True
    max_search_results: int = Field(default=3, ge=1, le=8)
    max_tokens: int = 350


class TranslateRequest(BaseModel):
    audio_path: str
    inference_audio_profile: str = "standard"
    target_language: str = "Chinese"
    asr_model: str | None = None
    llm_model: str | None = None
    audio_enhancement_enabled: bool = True
    audio_enhancement_mode: str = "webrtc"
    low_volume_boost: str = "medium"
    noise_suppression_level: str = "moderate"
    anti_cutoff_pause_ms: int = Field(default=350, ge=200, le=1200)
    audio_debug_enabled: bool = False
    enhancement_version: str = "v2"
    enhancement_mode: str = "fast_dsp"
    ns_engine: str = "webrtc"
    loudness_strategy: str = "dynaudnorm"
    dynamics: str = "upward_comp"
    limiter: dict[str, Any] = Field(default_factory=dict)
    targets: dict[str, Any] = Field(default_factory=dict)
    vad: dict[str, Any] = Field(default_factory=dict)
    system_prompt: str | None = None
    max_tokens: int = 350


class ASRChunkRequest(BaseModel):
    audio_path: str
    inference_audio_profile: str = "standard"
    asr_model: str | None = None
    llm_model: str | None = None
    audio_enhancement_enabled: bool = True
    audio_enhancement_mode: str = "webrtc"
    low_volume_boost: str = "medium"
    noise_suppression_level: str = "moderate"
    anti_cutoff_pause_ms: int = Field(default=350, ge=200, le=1200)
    audio_debug_enabled: bool = False
    enhancement_version: str = "v2"
    enhancement_mode: str = "fast_dsp"
    ns_engine: str = "webrtc"
    loudness_strategy: str = "dynaudnorm"
    dynamics: str = "upward_comp"
    limiter: dict[str, Any] = Field(default_factory=dict)
    targets: dict[str, Any] = Field(default_factory=dict)
    vad: dict[str, Any] = Field(default_factory=dict)


class PreparedTranscriptRequest(BaseModel):
    mode: str = "dictate"
    raw_text: str
    selected_text: str = ""
    target_language: str = "Chinese"
    asr_model: str | None = None
    llm_model: str | None = None
    system_prompt: str | None = None
    web_search_enabled: bool = True
    max_search_results: int = Field(default=3, ge=1, le=8)
    max_tokens: int = 350
    timing_ms: dict[str, float] = Field(default_factory=dict)


class MemoryTimeoutRequest(BaseModel):
    idle_timeout_seconds: int | None = Field(default=300, ge=1)


class DictionaryUpdateRequest(BaseModel):
    terms: list[str] = Field(default_factory=list)


class InferenceResponse(BaseModel):
    mode: str
    raw_text: str
    output_text: str
    used_web_search: bool = False
    web_sources: list[dict[str, str]] = Field(default_factory=list)
    timing_ms: dict[str, float]


class ASRChunkResponse(BaseModel):
    text: str
    timing_ms: dict[str, float]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clamp_float(value: Any, fallback: float, lower: float, upper: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(fallback)
    return max(lower, min(upper, parsed))


def apply_background_scheduling() -> dict[str, str]:
    status: dict[str, str] = {}

    try:
        os.nice(10)
        status["nice"] = "set_to_10"
    except OSError as exc:
        status["nice"] = f"not_set ({exc})"

    if sys.platform == "darwin":
        pid = str(os.getpid())
        try:
            subprocess.run(
                ["/usr/bin/renice", "10", "-p", pid],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            status["renice"] = "applied"
        except FileNotFoundError:
            status["renice"] = "missing"

        try:
            subprocess.run(
                ["/usr/sbin/taskpolicy", "-b", "-p", pid],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            status["taskpolicy"] = "background_applied"
        except FileNotFoundError:
            status["taskpolicy"] = "missing"

    return status


@dataclass
class SearchDecision:
    need_search: bool
    query: str


@dataclass
class AudioEnhancementConfig:
    enabled: bool
    inference_audio_profile: str
    enhancement_version: str
    enhancement_mode: str
    mode: str
    ns_engine: str
    loudness_strategy: str
    dynamics: str
    low_volume_boost: str
    noise_suppression_level: str
    anti_cutoff_pause_ms: int
    audio_debug_enabled: bool
    limiter: dict[str, Any]
    targets: dict[str, Any]
    vad: dict[str, Any]


@dataclass
class AudioEnhancementResult:
    transcribe_paths: list[str]
    cleanup_paths: list[Path]
    stats: dict[str, Any]


@dataclass
class ASRProcessingResult:
    text: str
    timing_ms: dict[str, float]


class ASRRequestError(Exception):
    def __init__(
        self,
        *,
        error_code: str,
        human_message: str,
        status_code: int = 422,
        technical_message: str = "",
    ) -> None:
        super().__init__(human_message)
        self.error_code = error_code
        self.human_message = human_message
        self.status_code = status_code
        self.technical_message = technical_message

    def to_payload(self) -> dict[str, str]:
        payload = {
            "error_code": self.error_code,
            "human_message": self.human_message,
        }
        if self.technical_message:
            payload["technical_message"] = self.technical_message
        return payload


class ResidentModelRuntime:
    def __init__(
        self,
        state_dir: Path,
        asr_model: str = DEFAULT_ASR_MODEL,
        llm_model: str = DEFAULT_LLM_MODEL,
        idle_timeout_seconds: int | None = 300,
    ) -> None:
        self.state_dir = state_dir
        self.app_support_dir = self.state_dir.parent
        self.asr_model_id = asr_model
        self.llm_model_id = llm_model
        self.idle_timeout_seconds = idle_timeout_seconds

        self.dictionary_path = self.app_support_dir / "dictionary.json"
        self.style_profile_path = self.state_dir / "style_profile.json"

        self._asr_module: Any | None = None
        self._llm_module: Any | None = None
        self._llm_model: Any | None = None
        self._llm_tokenizer: Any | None = None

        self._lock = threading.RLock()
        self._generation_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        self._last_active = time.monotonic()
        self._style_learning_idle_grace_seconds = 20.0
        self._scheduling = apply_background_scheduling()
        self._enhancement_plugins = probe_enhancement_plugins()
        self._ffmpeg_path: str | None = None
        self._ffmpeg_source = "none"
        self._transcribe_accepts_ndarray: bool | None = None
        print(
            f"[enhancement-capabilities] {json.dumps(self._enhancement_plugins, ensure_ascii=False)}",
            flush=True,
        )
        self._refresh_ffmpeg_capability()
        print(
            f"[asr-capabilities] {json.dumps(self._asr_capability_payload(), ensure_ascii=False)}",
            flush=True,
        )

        self._ensure_state_files()

    def start(self) -> None:
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="ghosttype-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def stop(self) -> None:
        self._shutdown.set()
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=1.0)
        self.release_models()

    def set_idle_timeout(self, seconds: int | None) -> None:
        with self._lock:
            self.idle_timeout_seconds = seconds
            self._last_active = time.monotonic()

    def get_idle_timeout(self) -> int | None:
        with self._lock:
            return self.idle_timeout_seconds

    def health(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_ffmpeg_capability()
            return {
                "status": "ok",
                "asr_model": self.asr_model_id,
                "llm_model": self.llm_model_id,
                "llm_loaded": self._llm_model is not None,
                "idle_timeout_seconds": self.idle_timeout_seconds,
                "last_active_age_seconds": round(time.monotonic() - self._last_active, 2),
                "state_dir": str(self.state_dir),
                "scheduling": self._scheduling,
                "asr_capabilities": self._asr_capability_payload(),
            }

    def _asr_capability_payload(self) -> dict[str, Any]:
        return {
            "wav_direct_decode": True,
            "ffmpeg_available": bool(self._ffmpeg_path),
            "ffmpeg_source": self._ffmpeg_source,
            "ffmpeg_path": self._ffmpeg_path or "",
            "ndarray_transcribe_supported": (
                self._transcribe_accepts_ndarray if self._transcribe_accepts_ndarray is not None else "unknown"
            ),
        }

    def _resolve_ffmpeg_path(self) -> tuple[str | None, str]:
        bundled_path = str(os.environ.get("GHOSTTYPE_FFMPEG_PATH") or "").strip()
        if bundled_path:
            if Path(bundled_path).is_file() and os.access(bundled_path, os.X_OK):
                return bundled_path, "bundled_env"
            return None, "bundled_env_invalid"

        system_path = shutil.which("ffmpeg")
        if system_path:
            return system_path, "system_path"
        return None, "not_found"

    def _refresh_ffmpeg_capability(self) -> None:
        path, source = self._resolve_ffmpeg_path()
        self._ffmpeg_path = path
        self._ffmpeg_source = source

    def _touch(self) -> None:
        self._last_active = time.monotonic()

    def _watchdog_loop(self) -> None:
        while not self._shutdown.is_set():
            time.sleep(2.0)
            with self._lock:
                timeout = self.idle_timeout_seconds
                elapsed = time.monotonic() - self._last_active
                should_release = (
                    timeout is not None
                    and self._llm_model is not None
                    and elapsed >= timeout
                )
            if should_release:
                self.release_models()

    def _ensure_state_files(self) -> None:
        self.app_support_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        if not self.dictionary_path.exists():
            items: list[dict[str, str]] = []
            legacy_path = self.state_dir / "custom_dictionary.json"
            if legacy_path.exists():
                try:
                    legacy_payload = json.loads(legacy_path.read_text(encoding="utf-8"))
                    terms = legacy_payload.get("terms") if isinstance(legacy_payload, dict) else None
                    if isinstance(terms, list):
                        for term in terms:
                            text = str(term).strip()
                            if not text:
                                continue
                            items.append(
                                {
                                    "originalText": text,
                                    "correctedText": text,
                                }
                            )
                except Exception:
                    items = []
            self.dictionary_path.write_text(
                json.dumps({"items": items}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if not self.style_profile_path.exists():
            self.style_profile_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "updated_at": utc_now_iso(),
                        "rules": [],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    def _normalize_dictionary_items(self, payload: Any) -> list[dict[str, str]]:
        if not isinstance(payload, list):
            return []
        normalized: list[dict[str, str]] = []
        for raw_item in payload:
            if not isinstance(raw_item, dict):
                continue
            original = str(raw_item.get("originalText") or "").strip()
            corrected = str(raw_item.get("correctedText") or "").strip()
            if not original or not corrected:
                continue
            normalized.append(
                {
                    "originalText": original,
                    "correctedText": corrected,
                }
            )
        return normalized

    def get_dictionary_items(self) -> list[dict[str, str]]:
        try:
            payload = json.loads(self.dictionary_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                if isinstance(payload.get("items"), list):
                    return self._normalize_dictionary_items(payload.get("items"))
                if isinstance(payload.get("terms"), list):
                    # Backward-compatible fallback for legacy dictionary schema.
                    legacy_items = []
                    for term in payload.get("terms") or []:
                        text = str(term).strip()
                        if not text:
                            continue
                        legacy_items.append(
                            {
                                "originalText": text,
                                "correctedText": text,
                            }
                        )
                    return legacy_items
            return []
        except Exception:
            return []

    def get_dictionary_terms(self) -> list[str]:
        items = self.get_dictionary_items()
        terms: list[str] = []
        for item in items:
            value = str(item.get("correctedText") or "").strip()
            if value and value not in terms:
                terms.append(value)
        return terms

    def update_dictionary_terms(self, terms: list[str]) -> list[str]:
        normalized = []
        for term in terms:
            value = str(term).strip()
            if value and value not in normalized:
                normalized.append(value)

        items = [
            {
                "originalText": value,
                "correctedText": value,
            }
            for value in normalized
        ]
        self.dictionary_path.write_text(
            json.dumps({"items": items}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return normalized

    def get_style_profile(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.style_profile_path.read_text(encoding="utf-8"))
            payload.setdefault("rules", [])
            payload.setdefault("updated_at", utc_now_iso())
            payload.setdefault("version", 1)
            return payload
        except Exception:
            return {"version": 1, "updated_at": utc_now_iso(), "rules": []}

    def clear_style_profile(self) -> dict[str, Any]:
        payload = {"version": 1, "updated_at": utc_now_iso(), "rules": []}
        self.style_profile_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return payload

    def _style_rules_as_text(self) -> str:
        rules = self.get_style_profile().get("rules", [])
        if not rules:
            return ""
        lines = [str(rule).strip() for rule in rules if str(rule).strip()]
        return "；".join(lines)

    def _append_personalization_rules(self, prompt: str) -> str:
        dictionary_items = self.get_dictionary_items()
        style_profile = self._style_rules_as_text()

        if dictionary_items:
            prompt += (
                "\n\nAdditional rule: strictly apply the following proper-noun mapping dictionary while processing text: "
                f"{json.dumps({'items': dictionary_items}, ensure_ascii=False)}"
            )
        if style_profile:
            prompt += (
                "\n\nAdditional rule: follow these abstract writing-style traits when generating the final text: "
                f"{style_profile}"
            )
        return prompt

    def _resolve_system_prompt(self, override_prompt: str | None, fallback_prompt: str) -> str:
        if not override_prompt:
            return fallback_prompt
        cleaned = str(override_prompt).strip()
        return cleaned if cleaned else fallback_prompt

    def _dictate_system_prompt(self) -> str:
        return (
            "You are an extremely rigorous ASR Transcript Rewriting Specialist.\n\n"
            "Goal\n"
            "Transform a user's messy spoken-language ASR transcript into clear, natural written text while preserving the user's meaning exactly.\n\n"
            "Non-negotiable rule (highest priority)\n"
            "Be 100% faithful to the original meaning. Do NOT delete, omit, summarize, compress, or generalize any substantive detail, including facts, numbers, examples, reasoning steps, decisions, constraints, emotional signals, and preferences. Your job is to clean and rewrite, not to summarize.\n\n"
            "What you MAY remove (noise only)\n"
            "- Filler words and hesitation sounds (\"uh\", \"um\", \"like\", \"you know\", \"I mean\", etc.).\n"
            "- Stutters and false starts that carry no meaning.\n"
            "- Immediate repetitions that do not add emphasis or new information.\n"
            "- Broken half-sentences that are clearly abandoned and replaced.\n\n"
            "What you MUST preserve\n"
            "- All concrete details (names, dates, numbers, units, places, requirements, constraints).\n"
            "- The user's logic, causality, and sequencing.\n"
            "- Self-corrections and changes of mind: preserve both initial intent and correction when both are semantically meaningful, integrated coherently.\n"
            "- Planning content: action items, owners, dependencies, deadlines, priorities, and execution order.\n"
            "- Tone and emotional intent, expressed naturally in writing.\n"
            "- Subject continuity (\"I\", \"we\", \"he/she/they\"). Do not convert statements into commands.\n\n"
            "Rewriting rules\n"
            "1. Add accurate punctuation and split into readable paragraphs by topic/logic.\n"
            "2. Output fluent, natural declarative written sentences.\n"
            "3. If the subject is omitted but clearly implied as \"I\", restore \"I\".\n"
            "4. Use Markdown bullets (-) only when there are real parallel items, steps, options, or multiple tasks; keep full detail in each bullet.\n"
            "5. Output only the rewritten text. Do not add prefaces, labels, explanations, or commentary.\n"
            "6. If any fragment is truly unintelligible, keep its position and mark it as [inaudible]. Do not guess.\n\n"
            "Output format\n"
            "- Plain rewritten text only.\n"
            "- Use paragraphs and/or Markdown bullets as needed.\n"
            "- No headings unless clearly implied by the user's original words.\n\n"
            "Few-shot examples\n\n"
            "Example 1\n"
            "User: \"Tomorrow I'm gonna go to the supermarket to buy a banana and milk, actually never mind, just milk, and then after that I'll go to the gym.\"\n"
            "Assistant:\n"
            "- Tomorrow I plan to go to the supermarket to buy milk; at first I considered buying a banana as well, but I decided against it.\n"
            "- After that, I will go to the gym.\n\n"
            "Example 2\n"
            "User: \"Hey, I wanna email my boss to ask about that project... oh wait, not the project, the contract progress.\"\n"
            "Assistant:\n"
            "I want to email my boss to ask about the project; actually, I mean to ask about the contract's progress."
        )

    def _dictate_user_prompt(self, raw_text: str) -> str:
        return (
            "Rewrite the following ASR transcript into clean written text while preserving all substantive details:\n\n"
            f"{raw_text}"
        )

    def _ask_system_prompt(self) -> str:
        return (
            "You are a precise, efficient Q&A assistant in Ask mode.\n\n"
            "Inputs\n"
            "- Reference Text: a snippet the user highlighted from their screen.\n"
            "- Voice Question: a spoken question transcribed by ASR.\n\n"
            "Primary goal\n"
            "Answer the Voice Question as accurately as possible with the fewest necessary words.\n\n"
            "Rules (priority order)\n"
            "1) Use the Reference Text first.\n"
            "   - If the Reference Text contains enough information, answer using it.\n"
            "   - Quote or paraphrase only what is necessary.\n"
            "2) If the Reference Text is irrelevant or insufficient:\n"
            "   - Answer using concise general knowledge.\n"
            "   - Do not invent details that would require missing context.\n"
            "3) Extreme brevity:\n"
            "   - Give the shortest correct answer that fully resolves the question.\n"
            "   - Avoid background or reasoning unless explicitly requested.\n"
            "4) Output only the answer:\n"
            "   - No prefaces or meta phrases.\n"
            "   - Do not mention \"Reference Text\" or \"Voice Question\".\n"
            "5) Clarification handling:\n"
            "   - If ambiguity blocks correctness and the Reference Text cannot resolve it, ask one minimal clarifying question.\n"
            "   - Otherwise choose the most reasonable interpretation and answer directly.\n"
            "6) Formatting:\n"
            "   - Prefer one sentence or one short phrase.\n"
            "   - Use bullets only when multiple discrete items are strictly required."
        )

    def _ask_user_prompt(self, selected_text: str, question: str, search_text: str) -> str:
        return (
            "Reference Text:\n"
            f"{selected_text}\n\n"
            "Voice Question:\n"
            f"{question}\n\n"
            "Web Search Context:\n"
            f"{search_text}"
        )

    def _translate_system_prompt(self, target_language: str) -> str:
        return (
            "You are a high-accuracy machine translation engine.\n\n"
            "Your only task\n"
            f"Translate the user-provided text (the user's spoken content) into {target_language}.\n\n"
            "Rules (non-negotiable)\n"
            "1) Full normalization into the target language:\n"
            f"   - If the input mixes multiple languages, translate everything into fluent, natural {target_language}.\n"
            "   - Keep proper nouns and brand names unchanged only when they should remain unchanged.\n"
            "2) Meaning and terminology fidelity:\n"
            "   - Preserve the original meaning precisely.\n"
            "   - Translate technical and professional terms accurately and consistently.\n"
            "   - Keep numbers, units, dates, names, and identifiers correct.\n"
            "3) Tone and intent:\n"
            "   - Preserve tone, politeness level, and emotional intent where applicable.\n"
            "4) Output only the translation:\n"
            "   - Output must contain only translated text.\n"
            "   - Do not include source text, quotation marks, notes, labels, or prefatory phrases.\n"
            "5) Formatting preservation:\n"
            "   - Preserve paragraph breaks, list structure, and line breaks when possible.\n"
            "   - Do not add extra content.\n"
            "6) Ambiguity handling:\n"
            "   - Choose the most context-plausible translation.\n"
            "   - If critical meaning is truly unclear and cannot be inferred, use [unclear] rather than guessing."
        )

    def _translate_user_prompt(self, raw_text: str) -> str:
        return f"Text to translate:\n{raw_text}"

    def _timing_payload(
        self,
        t_asr: float,
        t_llm: float,
        extra_timings: dict[str, float] | None = None,
    ) -> dict[str, float]:
        payload: dict[str, float] = {
            "asr": round(t_asr, 2),
            "llm": round(t_llm, 2),
            "total": round(t_asr + t_llm, 2),
        }
        if extra_timings:
            for key, value in extra_timings.items():
                try:
                    payload[key] = round(float(value), 2)
                except Exception:
                    continue
        return payload

    def _build_inference_response(
        self,
        mode: str,
        raw_text: str,
        output_text: str,
        t_asr: float,
        t_llm: float,
        used_web_search: bool = False,
        web_sources: list[dict[str, str]] | None = None,
        extra_timings: dict[str, float] | None = None,
    ) -> InferenceResponse:
        return InferenceResponse(
            mode=mode,
            raw_text=raw_text,
            output_text=output_text or raw_text,
            used_web_search=used_web_search,
            web_sources=web_sources or [],
            timing_ms=self._timing_payload(t_asr, t_llm, extra_timings=extra_timings),
        )

    def _run_llm_sync(self, messages: list[dict[str, str]], max_tokens: int) -> tuple[str, float]:
        t1 = time.perf_counter()
        output = self._generate_chat(messages=messages, max_tokens=max_tokens)
        t_llm = (time.perf_counter() - t1) * 1000
        self._touch()
        return output, t_llm

    def _prepare_ask_search_context(self, req: AskRequest, question: str) -> tuple[bool, list[dict[str, str]], str]:
        used_search = False
        web_sources: list[dict[str, str]] = []
        if req.web_search_enabled:
            decision = self._decide_search(req.selected_text, question)
            if decision.need_search:
                web_sources = self._run_duckduckgo(decision.query, req.max_search_results)
                used_search = len(web_sources) > 0

        search_text = "\n".join(
            [
                f"- {source.get('title', '')}: {source.get('snippet', '')} ({source.get('url', '')})"
                for source in web_sources
            ]
        )
        if not search_text:
            search_text = "No web sources used."
        return used_search, web_sources, search_text

    def _stream_mode_response(
        self,
        mode: str,
        raw_text: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        t_asr: float,
        used_web_search: bool = False,
        web_sources: list[dict[str, str]] | None = None,
        extra_timings: dict[str, float] | None = None,
    ):
        t1 = time.perf_counter()
        output_parts: list[str] = []
        first_token_ms: float | None = None
        try:
            for token in self._stream_generate_chat(messages=messages, max_tokens=max_tokens):
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - t1) * 1000.0
                output_parts.append(token)
                yield self._sse_event({"type": "token", "token": token})
        finally:
            self._touch()
        t_llm = (time.perf_counter() - t1) * 1000

        output = "".join(output_parts).strip() or raw_text
        self._start_style_learning_task(output)
        response_timing = dict(extra_timings or {})
        if first_token_ms is not None:
            response_timing["llm_first_token"] = first_token_ms
        response = self._build_inference_response(
            mode=mode,
            raw_text=raw_text,
            output_text=output,
            t_asr=t_asr,
            t_llm=t_llm,
            used_web_search=used_web_search,
            web_sources=web_sources,
            extra_timings=response_timing,
        )
        yield self._sse_event({"type": "done", "meta": self._response_payload(response)})
        yield self._sse_event("[DONE]")

    def _clamp_max_tokens(self, value: int) -> int:
        return max(1, min(int(value), 350))

    def _make_asr_initial_prompt(self) -> str:
        terms = self.get_dictionary_terms()
        if not terms:
            return ""
        joined = ", ".join(terms)
        return (
            "Prefer these spellings for proper nouns and domain vocabulary: "
            f"{joined}."
        )

    def _ensure_asr_module(self) -> Any:
        if self._asr_module is None:
            self._asr_module = mlx_whisper
        return self._asr_module

    def _ensure_transcribe_ndarray_support(self, transcribe_func: Any) -> bool:
        if self._transcribe_accepts_ndarray is not None:
            return self._transcribe_accepts_ndarray
        try:
            signature = inspect.signature(transcribe_func)
            signature.bind(np.zeros(1, dtype=np.float32), path_or_hf_repo=self.asr_model_id)
            self._transcribe_accepts_ndarray = True
        except Exception:
            self._transcribe_accepts_ndarray = False
        return self._transcribe_accepts_ndarray

    @contextmanager
    def _ffmpeg_decode_environment(self, requires_ffmpeg: bool):
        if not requires_ffmpeg:
            yield
            return

        self._refresh_ffmpeg_capability()
        ffmpeg_path = self._ffmpeg_path
        if not ffmpeg_path:
            raise ASRRequestError(
                error_code="asr_decoder_unavailable",
                human_message=(
                    "音频需要 ffmpeg 解码，但当前未找到可用解码器。"
                    "请开启内置 ffmpeg 或安装系统 ffmpeg，或使用 16kHz 单声道 PCM16 WAV 重新录音。"
                ),
                technical_message="ffmpeg not available for path-based decode",
            )

        ffmpeg_dir = str(Path(ffmpeg_path).resolve().parent)
        original_path = str(os.environ.get("PATH") or "")
        has_dir = ffmpeg_dir in [entry for entry in original_path.split(os.pathsep) if entry]
        if has_dir:
            yield
            return

        os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{original_path}" if original_path else ffmpeg_dir
        try:
            yield
        finally:
            os.environ["PATH"] = original_path

    def _prepare_transcribe_input(self, audio_path: str, transcribe_func: Any) -> tuple[str | np.ndarray, bool]:
        self._refresh_ffmpeg_capability()
        path = Path(audio_path)
        is_wav = path.suffix.lower() == ".wav"

        if is_wav:
            try:
                waveform = load_wav_pcm16_mono(path)
            except WavFormatError as exc:
                if self._ffmpeg_path:
                    return audio_path, True
                raise ASRRequestError(
                    error_code="asr_wav_format_unsupported",
                    human_message=(
                        "音频格式不满足 16kHz 单声道 PCM16，且未找到可用 ffmpeg 解码器。"
                        "请开启内置 ffmpeg 选项或重新录音。"
                    ),
                    technical_message=str(exc),
                ) from exc
            except wave.Error as exc:
                if self._ffmpeg_path:
                    return audio_path, True
                raise ASRRequestError(
                    error_code="asr_wav_decode_failed",
                    human_message=(
                        "WAV 文件解析失败，且未找到可用 ffmpeg 解码器。"
                        "请开启内置 ffmpeg 选项或重新录音。"
                    ),
                    technical_message=str(exc),
                ) from exc
            except ValueError as exc:
                raise ASRRequestError(
                    error_code="asr_audio_invalid",
                    human_message="音频数据无效，请重新录音后重试。",
                    technical_message=str(exc),
                ) from exc

            if self._ensure_transcribe_ndarray_support(transcribe_func):
                return waveform, False

            if self._ffmpeg_path:
                return audio_path, True
            raise ASRRequestError(
                error_code="asr_ndarray_not_supported",
                human_message=(
                    "当前 mlx-whisper 版本不支持内存音频输入，且未找到可用 ffmpeg 解码器。"
                    "请开启内置 ffmpeg 或安装系统 ffmpeg。"
                ),
                technical_message="mlx_whisper.transcribe does not accept ndarray input",
            )

        if self._ffmpeg_path:
            return audio_path, True
        raise ASRRequestError(
            error_code="asr_decoder_unavailable",
            human_message=(
                "当前仅支持 16kHz 单声道 PCM16 WAV 直读；非 WAV 音频需要 ffmpeg 解码。"
                "请开启内置 ffmpeg 选项或安装系统 ffmpeg。"
            ),
            technical_message=f"unsupported input extension: {path.suffix.lower() or '<none>'}",
        )

    def _raise_transcribe_exception(self, exc: Exception) -> None:
        if isinstance(exc, ASRRequestError):
            raise exc
        if isinstance(exc, FileNotFoundError):
            raise ASRRequestError(
                error_code="asr_ffmpeg_not_found",
                human_message=(
                    "本地 ASR 需要 ffmpeg 解码当前音频，但未找到可执行文件。"
                    "请开启内置 ffmpeg 选项或安装系统 ffmpeg。"
                ),
                technical_message=str(exc),
            ) from exc
        if isinstance(exc, subprocess.CalledProcessError):
            raise ASRRequestError(
                error_code="asr_ffmpeg_decode_failed",
                human_message="ffmpeg 解码失败，请检查音频文件是否损坏，或重新录音后重试。",
                technical_message=str(exc),
            ) from exc
        if isinstance(exc, ValueError):
            raise ASRRequestError(
                error_code="asr_audio_invalid",
                human_message="音频数据无效，请重新录音后重试。",
                technical_message=str(exc),
            ) from exc
        raise exc

    def _ensure_llm_loaded(self) -> tuple[Any, Any]:
        global llm_model, llm_tokenizer

        if self._llm_model is None or self._llm_tokenizer is None:
            if (
                self.llm_model_id == LLM_REPO
                and llm_model is not None
                and llm_tokenizer is not None
            ):
                self._llm_model = llm_model
                self._llm_tokenizer = llm_tokenizer
            else:
                self._llm_model, self._llm_tokenizer = load(self.llm_model_id)
                if self.llm_model_id == LLM_REPO:
                    llm_model = self._llm_model
                    llm_tokenizer = self._llm_tokenizer
        return self._llm_model, self._llm_tokenizer

    def _apply_model_overrides(self, asr_model: str | None, llm_model: str | None) -> None:
        changed_llm = False
        if asr_model and asr_model != self.asr_model_id:
            self.asr_model_id = asr_model
        if llm_model and llm_model != self.llm_model_id:
            self.llm_model_id = llm_model
            changed_llm = True

        if changed_llm:
            self._llm_model = None
            self._llm_tokenizer = None
            gc.collect()
            try:
                import mlx.core as mx

                mx.metal.clear_cache()
            except Exception:
                pass

    def release_models(self) -> None:
        global llm_model, llm_tokenizer

        with self._lock:
            self._llm_model = None
            self._llm_tokenizer = None
            llm_model = None
            llm_tokenizer = None
            gc.collect()
            try:
                import mlx.core as mx

                mx.metal.clear_cache()
            except Exception:
                pass
            self._touch()

    def _audio_config_from_request(self, req: Any) -> AudioEnhancementConfig:
        raw_profile = str(getattr(req, "inference_audio_profile", "standard") or "standard").strip().lower()
        if raw_profile not in {"standard", "fast", "quality"}:
            raw_profile = "standard"

        raw_mode = str(getattr(req, "audio_enhancement_mode", "webrtc") or "webrtc").strip().lower()
        mode_aliases = {
            "webrtc": "webrtc",
            "web_rtc": "webrtc",
            "system_voice_processing": "system_voice_processing",
            "systemvoiceprocessing": "system_voice_processing",
            "system": "system_voice_processing",
            "off": "off",
            "none": "off",
            "disabled": "off",
        }
        mode = mode_aliases.get(raw_mode, "webrtc")

        low_boost = str(getattr(req, "low_volume_boost", "medium") or "medium").strip().lower()
        if low_boost not in {"low", "medium", "high"}:
            low_boost = "medium"

        raw_noise_level = str(getattr(req, "noise_suppression_level", "moderate") or "moderate").strip().lower()
        noise_aliases = {
            "low": "low",
            "moderate": "moderate",
            "high": "high",
            "veryhigh": "veryhigh",
            "very_high": "veryhigh",
            "very-high": "veryhigh",
        }
        noise_level = noise_aliases.get(raw_noise_level, "moderate")

        pause_ms = int(getattr(req, "anti_cutoff_pause_ms", 350) or 350)
        pause_ms = max(200, min(pause_ms, 1200))

        enabled = bool(getattr(req, "audio_enhancement_enabled", True))
        if mode == "off":
            enabled = False

        raw_enhancement_version = str(getattr(req, "enhancement_version", "v2") or "v2").strip().lower()
        enhancement_version = raw_enhancement_version if raw_enhancement_version in {"legacy", "v2"} else "v2"

        raw_enhancement_mode = str(getattr(req, "enhancement_mode", "fast_dsp") or "fast_dsp").strip().lower()
        enhancement_mode_aliases = {
            "fast_dsp": "fast_dsp",
            "fast": "fast_dsp",
            "high_quality": "high_quality",
            "highquality": "high_quality",
            "hq": "high_quality",
            "custom": "custom",
        }
        enhancement_mode = enhancement_mode_aliases.get(raw_enhancement_mode, "fast_dsp")

        raw_ns_engine = str(getattr(req, "ns_engine", "webrtc") or "webrtc").strip().lower()
        ns_engine_aliases = {
            "off": "off",
            "none": "off",
            "webrtc": "webrtc",
            "web_rtc": "webrtc",
            "rnnoise": "rnnoise",
            "deepfilternet": "deepfilternet",
            "deepfilter": "deepfilternet",
            "speex": "speex",
            "speexdsp": "speex",
        }
        ns_engine = ns_engine_aliases.get(raw_ns_engine, "webrtc")

        raw_loudness = str(getattr(req, "loudness_strategy", "dynaudnorm") or "dynaudnorm").strip().lower()
        loudness_aliases = {
            "lufs": "lufs",
            "lufs_normalize": "lufs",
            "dynaudnorm": "dynaudnorm",
            "dynamic": "dynaudnorm",
            "dynamic_normalize": "dynaudnorm",
            "rms": "rms",
            "legacy": "rms",
        }
        loudness_strategy = loudness_aliases.get(raw_loudness, "dynaudnorm")

        raw_dynamics = str(getattr(req, "dynamics", "upward_comp") or "upward_comp").strip().lower()
        dynamics_aliases = {
            "off": "off",
            "none": "off",
            "upward": "upward_comp",
            "upward_comp": "upward_comp",
            "upward_compressor": "upward_comp",
            "comp_limiter": "comp_limiter",
            "compressor_limiter": "comp_limiter",
        }
        dynamics = dynamics_aliases.get(raw_dynamics, "upward_comp")

        limiter_payload = getattr(req, "limiter", {}) or {}
        if not isinstance(limiter_payload, dict):
            limiter_payload = {}
        limiter = {
            "enabled": bool(limiter_payload.get("enabled", True)),
            "threshold": float(_clamp_float(limiter_payload.get("threshold"), 0.98, 0.6, 0.999)),
            "attack_ms": float(_clamp_float(limiter_payload.get("attack_ms"), 5.0, 0.1, 100.0)),
            "release_ms": float(_clamp_float(limiter_payload.get("release_ms"), 50.0, 1.0, 500.0)),
        }

        targets_payload = getattr(req, "targets", {}) or {}
        if not isinstance(targets_payload, dict):
            targets_payload = {}
        targets = {
            "lufs_target": float(_clamp_float(targets_payload.get("lufs_target"), -18.0, -32.0, -10.0)),
            "max_gain_db": float(_clamp_float(targets_payload.get("max_gain_db"), 18.0, 0.0, 36.0)),
        }

        vad_payload = getattr(req, "vad", {}) or {}
        if not isinstance(vad_payload, dict):
            vad_payload = {}
        vad_engine_raw = str(vad_payload.get("engine", "webrtcvad") or "webrtcvad").strip().lower()
        vad_engine_aliases = {
            "webrtcvad": "webrtcvad",
            "webrtc": "webrtcvad",
            "energy": "energy",
        }
        vad = {
            "engine": vad_engine_aliases.get(vad_engine_raw, "webrtcvad"),
            "aggressiveness": int(_clamp_float(vad_payload.get("aggressiveness"), 1, 0, 3)),
            "preroll_ms": int(_clamp_float(vad_payload.get("preroll_ms"), 100, 0, 500)),
            "hangover_ms": int(_clamp_float(vad_payload.get("hangover_ms"), pause_ms, 100, 1200)),
        }

        return AudioEnhancementConfig(
            enabled=enabled,
            inference_audio_profile=raw_profile,
            enhancement_version=enhancement_version,
            enhancement_mode=enhancement_mode,
            mode=mode,
            ns_engine=ns_engine,
            loudness_strategy=loudness_strategy,
            dynamics=dynamics,
            low_volume_boost=low_boost,
            noise_suppression_level=noise_level,
            anti_cutoff_pause_ms=pause_ms,
            audio_debug_enabled=bool(getattr(req, "audio_debug_enabled", False)),
            limiter=limiter,
            targets=targets,
            vad=vad,
        )

    def _transcribe_audio(
        self,
        audio_path: str,
        language: str = "auto",
        audio_config: AudioEnhancementConfig | None = None,
    ) -> ASRProcessingResult:
        asr_started_at = time.perf_counter()
        enhancement_result = AudioEnhancementResult(
            transcribe_paths=[audio_path],
            cleanup_paths=[],
            stats={"enabled": False},
        )
        if audio_config and audio_config.enabled:
            enhancement_result = self._prepare_audio_for_transcription(
                audio_path=audio_path,
                audio_config=audio_config,
            )
        asr_inference_started_at = time.perf_counter()

        try:
            chunks: list[str] = []
            first_packet_ms: float | None = None
            for index, path in enumerate(enhancement_result.transcribe_paths):
                text = self._transcribe_audio_single(path, language=language)
                if index == 0:
                    first_packet_ms = (time.perf_counter() - asr_inference_started_at) * 1000.0
                if text:
                    chunks.append(text)
            asr_elapsed_ms = (time.perf_counter() - asr_started_at) * 1000.0
            asr_inference_ms = (time.perf_counter() - asr_inference_started_at) * 1000.0
            timing: dict[str, float] = {
                "asr_request_send": 0.0,
                "asr_first_packet": first_packet_ms if first_packet_ms is not None else asr_inference_ms,
                "asr": asr_elapsed_ms,
                "asr_inference": asr_inference_ms,
            }
            stats = enhancement_result.stats
            if isinstance(stats, dict):
                for key in (
                    "audio_stop_to_vad_done",
                    "enhancement_chain",
                    "preprocess_total_ms",
                ):
                    value = stats.get(key)
                    if value is None:
                        continue
                    try:
                        timing[key] = float(value)
                    except Exception:
                        continue
                # Keep compatibility with existing "preprocess_total_ms" while exposing a shorter key.
                if "preprocess_total_ms" in stats and "preprocess" not in timing:
                    try:
                        timing["preprocess"] = float(stats["preprocess_total_ms"])
                    except Exception:
                        pass
            return ASRProcessingResult(
                text=" ".join(chunks).strip(),
                timing_ms=timing,
            )
        finally:
            files = [path for path in enhancement_result.cleanup_paths if path.is_file()]
            dirs = [path for path in enhancement_result.cleanup_paths if path.is_dir()]
            for path in files:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    continue
            for path in sorted(dirs, key=lambda item: len(str(item)), reverse=True):
                try:
                    path.rmdir()
                except Exception:
                    continue

    def _transcribe_audio_single(self, audio_path: str, language: str = "auto") -> str:
        module = self._ensure_asr_module()
        transcribe = module.transcribe
        lang = None if language.lower() == "auto" else language
        initial_prompt = self._make_asr_initial_prompt()
        transcribe_input, requires_ffmpeg = self._prepare_transcribe_input(audio_path, transcribe)

        attempts = [
            {
                "path_or_hf_repo": self.asr_model_id,
                "language": lang,
                "task": "transcribe",
                "initial_prompt": initial_prompt,
            },
            {
                "path_or_hf_repo": self.asr_model_id,
                "language": lang,
                "task": "transcribe",
            },
            {
                "path_or_hf_repo": self.asr_model_id,
                "language": lang,
            },
        ]

        result: dict[str, Any] | None = None
        for kwargs in attempts:
            clean_kwargs = {k: v for k, v in kwargs.items() if v not in (None, "")}
            try:
                with self._ffmpeg_decode_environment(requires_ffmpeg):
                    result = transcribe(transcribe_input, **clean_kwargs)
                break
            except TypeError:
                continue
            except (FileNotFoundError, subprocess.CalledProcessError, ValueError, ASRRequestError) as exc:
                self._raise_transcribe_exception(exc)

        if result is None:
            try:
                with self._ffmpeg_decode_environment(requires_ffmpeg):
                    result = transcribe(transcribe_input)
            except (FileNotFoundError, subprocess.CalledProcessError, ValueError, ASRRequestError) as exc:
                self._raise_transcribe_exception(exc)

        return str((result or {}).get("text") or "").strip()

    def _prepare_audio_for_transcription(
        self,
        audio_path: str,
        audio_config: AudioEnhancementConfig,
    ) -> AudioEnhancementResult:
        t0 = time.perf_counter()
        stage_started_at = t0
        stats: dict[str, Any] = {
            "enabled": audio_config.enabled,
            "inference_audio_profile": audio_config.inference_audio_profile,
            "mode": audio_config.mode,
            "enhancement_version": audio_config.enhancement_version,
            "enhancement_mode": audio_config.enhancement_mode,
            "ns_engine": audio_config.ns_engine,
            "loudness_strategy": audio_config.loudness_strategy,
            "dynamics": audio_config.dynamics,
            "low_volume_boost": audio_config.low_volume_boost,
            "noise_suppression_level": audio_config.noise_suppression_level,
            "anti_cutoff_pause_ms": audio_config.anti_cutoff_pause_ms,
            "webrtc_vad_available": bool(webrtcvad is not None),
            "webrtc_apm_available": bool(WebRTCAudioProcessingModule is not None),
        }

        if not audio_config.enabled or audio_config.mode == "off":
            return AudioEnhancementResult(
                transcribe_paths=[audio_path],
                cleanup_paths=[],
                stats=stats,
            )

        try:
            signal, load_stats = self._load_wav_as_mono_float32_16k(audio_path)
            stats.update(load_stats)
        except Exception as exc:
            stats["fallback_reason"] = f"audio_load_failed: {exc}"
            self._log_audio_enhancement_stats(stats, debug=audio_config.audio_debug_enabled)
            return AudioEnhancementResult(
                transcribe_paths=[audio_path],
                cleanup_paths=[],
                stats=stats,
            )

        if signal.size == 0:
            stats["fallback_reason"] = "empty_audio"
            self._log_audio_enhancement_stats(stats, debug=audio_config.audio_debug_enabled)
            return AudioEnhancementResult(
                transcribe_paths=[audio_path],
                cleanup_paths=[],
                stats=stats,
            )

        stats["input_duration_ms"] = round(1000.0 * signal.size / 16000.0, 2)
        stats["input_rms_dbfs"] = round(self._estimate_rms_dbfs(signal), 2)
        stats["input_peak_dbfs"] = round(self._estimate_peak_dbfs(signal), 2)

        enhancement_started_at = time.perf_counter()
        if audio_config.enhancement_version == "v2":
            enhanced_signal, v2_stats = self._apply_enhancement_v2(signal, audio_config)
            stats.update(v2_stats)
        else:
            if audio_config.mode == "webrtc":
                preamp_signal, gain_stats = self._apply_low_volume_boost(signal, audio_config)
                stats.update(gain_stats)
                stats["preamp_rms_dbfs"] = round(self._estimate_rms_dbfs(preamp_signal), 2)
                stats["preamp_peak_dbfs"] = round(self._estimate_peak_dbfs(preamp_signal), 2)

                hpf_signal, hpf_stats = self._apply_high_pass_filter(preamp_signal, cutoff_hz=80.0, sample_rate=16000)
                stats.update(hpf_stats)

                apm_signal, apm_stats = self._apply_webrtc_apm_if_available(hpf_signal, audio_config)
                stats.update(apm_stats)

                enhanced_signal, limiter_stats = self._apply_soft_limiter(apm_signal)
                stats.update(limiter_stats)
            else:
                # System voice-processing mode is handled by AVAudioEngine in Swift.
                # Keep a lightweight local boost+limiter pass for weak-input robustness.
                enhanced_signal, gain_stats = self._apply_low_volume_boost_and_limiter(signal, audio_config)
                stats.update(gain_stats)
                stats["apm_backend"] = "not_requested"
        stats["enhancement_chain"] = round((time.perf_counter() - enhancement_started_at) * 1000.0, 2)

        stats["enhanced_rms_dbfs"] = round(self._estimate_rms_dbfs(enhanced_signal), 2)
        stats["enhanced_peak_dbfs"] = round(self._estimate_peak_dbfs(enhanced_signal), 2)

        frame_size = 160  # 10ms at 16kHz
        frames = self._split_into_frames(enhanced_signal, frame_size=frame_size)
        requested_vad_engine = str(audio_config.vad.get("engine", "webrtcvad") or "webrtcvad").strip().lower()
        requested_vad_aggressiveness = int(
            _clamp_float(audio_config.vad.get("aggressiveness"), 1, 0, 3)
        )
        vad_flags, vad_stats = self._detect_speech_frames(
            frames,
            audio_config,
            vad_engine_override=requested_vad_engine,
            vad_aggressiveness=requested_vad_aggressiveness,
        )
        stats.update(vad_stats)

        hangover_ms = int(
            _clamp_float(
                audio_config.vad.get("hangover_ms"),
                audio_config.anti_cutoff_pause_ms,
                100,
                1200,
            )
        )
        preroll_ms = int(_clamp_float(audio_config.vad.get("preroll_ms"), 100, 0, 500))
        segments, segment_stats = self._segment_by_vad(
            frames=frames,
            vad_flags=vad_flags,
            anti_cutoff_pause_ms=hangover_ms,
            pre_roll_ms=preroll_ms,
            frame_size=frame_size,
        )
        stats.update(segment_stats)
        stats["audio_stop_to_vad_done"] = round((time.perf_counter() - stage_started_at) * 1000.0, 2)

        if not segments:
            stats["segmentation_fallback"] = True
            segments = [enhanced_signal]

        cleanup_paths: list[Path] = []
        transcribe_paths: list[str] = []
        temp_dir = Path(tempfile.mkdtemp(prefix="ghosttype-audio-"))

        for index, segment in enumerate(segments):
            clipped = np.clip(segment, -1.0, 1.0).astype(np.float32, copy=False)
            if clipped.size < frame_size:
                continue
            segment_path = temp_dir / f"segment-{index:03d}.wav"
            self._write_wav_mono_16k_int16(segment_path, clipped)
            cleanup_paths.append(segment_path)
            transcribe_paths.append(str(segment_path))

        if not transcribe_paths:
            stats["segment_write_fallback"] = True
            cleanup_paths = []
            transcribe_paths = [audio_path]
            try:
                temp_dir.rmdir()
            except Exception:
                pass
        else:
            cleanup_paths.append(temp_dir)

        stats["transcribe_segments"] = len(transcribe_paths)
        stats["preprocess_total_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        self._log_audio_enhancement_stats(stats, debug=audio_config.audio_debug_enabled)
        return AudioEnhancementResult(
            transcribe_paths=transcribe_paths,
            cleanup_paths=cleanup_paths,
            stats=stats,
        )

    def _build_v2_engine_config(self, audio_config: AudioEnhancementConfig) -> Any | None:
        if EnhancementV2Config is None:
            return None
        if EnhancementLimiterConfig is None or EnhancementTargetsConfig is None or EnhancementVADConfig is None:
            return None

        limiter = EnhancementLimiterConfig(
            enabled=bool(audio_config.limiter.get("enabled", True)),
            threshold=float(_clamp_float(audio_config.limiter.get("threshold"), 0.98, 0.6, 0.999)),
            attack_ms=float(_clamp_float(audio_config.limiter.get("attack_ms"), 5.0, 0.1, 100.0)),
            release_ms=float(_clamp_float(audio_config.limiter.get("release_ms"), 50.0, 1.0, 500.0)),
        )
        targets = EnhancementTargetsConfig(
            lufs_target=float(_clamp_float(audio_config.targets.get("lufs_target"), -18.0, -32.0, -10.0)),
            max_gain_db=float(_clamp_float(audio_config.targets.get("max_gain_db"), 18.0, 0.0, 36.0)),
        )
        vad = EnhancementVADConfig(
            engine=str(audio_config.vad.get("engine", "webrtcvad") or "webrtcvad"),
            aggressiveness=int(_clamp_float(audio_config.vad.get("aggressiveness"), 1, 0, 3)),
            preroll_ms=int(_clamp_float(audio_config.vad.get("preroll_ms"), 100, 0, 500)),
            hangover_ms=int(
                _clamp_float(
                    audio_config.vad.get("hangover_ms"),
                    audio_config.anti_cutoff_pause_ms,
                    100,
                    1200,
                )
            ),
        )
        return EnhancementV2Config(
            mode=audio_config.enhancement_mode,
            ns_engine=audio_config.ns_engine,
            noise_suppression_level=audio_config.noise_suppression_level,
            loudness_strategy=audio_config.loudness_strategy,
            dynamics=audio_config.dynamics,
            limiter=limiter,
            targets=targets,
            vad=vad,
            hpf_cutoff_hz=80.0,
        )

    def _speech_mask_from_vad_flags(
        self,
        vad_flags: list[bool],
        frame_size: int,
        total_samples: int,
    ) -> np.ndarray:
        if total_samples <= 0:
            return np.zeros(0, dtype=np.bool_)
        mask = np.zeros(total_samples, dtype=np.bool_)
        for index, is_speech in enumerate(vad_flags):
            if not is_speech:
                continue
            start = index * frame_size
            end = min(total_samples, start + frame_size)
            if start >= end:
                break
            mask[start:end] = True
        return mask

    def _apply_enhancement_v2(
        self,
        signal: np.ndarray,
        audio_config: AudioEnhancementConfig,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if EnhancementEngine is None:
            legacy_signal, legacy_stats = self._apply_low_volume_boost_and_limiter(signal, audio_config)
            legacy_stats["v2_backend"] = "legacy_fallback_module_missing"
            return legacy_signal, legacy_stats

        frame_size = 160
        probe_frames = self._split_into_frames(signal, frame_size=frame_size)
        probe_vad_engine = str(audio_config.vad.get("engine", "webrtcvad") or "webrtcvad")
        probe_vad_aggressiveness = int(_clamp_float(audio_config.vad.get("aggressiveness"), 1, 0, 3))
        probe_flags, _ = self._detect_speech_frames(
            probe_frames,
            audio_config,
            vad_engine_override=probe_vad_engine,
            vad_aggressiveness=probe_vad_aggressiveness,
        )
        speech_mask = self._speech_mask_from_vad_flags(
            probe_flags,
            frame_size=frame_size,
            total_samples=signal.size,
        )

        v2_config = self._build_v2_engine_config(audio_config)
        if v2_config is None:
            legacy_signal, legacy_stats = self._apply_low_volume_boost_and_limiter(signal, audio_config)
            legacy_stats["v2_backend"] = "legacy_fallback_config_missing"
            return legacy_signal, legacy_stats

        try:
            engine = EnhancementEngine(v2_config)
            result = engine.process(signal=signal, sample_rate=16000, speech_mask=speech_mask)
            stats = dict(result.stats)
            stats["v2_backend"] = "enhancement_engine_v2"
            return result.signal.astype(np.float32, copy=False), stats
        except Exception as exc:
            legacy_signal, legacy_stats = self._apply_low_volume_boost_and_limiter(signal, audio_config)
            legacy_stats["v2_backend"] = "legacy_fallback_runtime_error"
            legacy_stats["v2_error"] = str(exc)
            return legacy_signal, legacy_stats

    def _detect_speech_frames(
        self,
        frames: list[np.ndarray],
        audio_config: AudioEnhancementConfig,
        vad_engine_override: str | None = None,
        vad_aggressiveness: int | None = None,
    ) -> tuple[list[bool], dict[str, Any]]:
        if not frames:
            return [], {"vad_frame_count": 0, "vad_speech_ratio": 0.0, "vad_backend": "none"}

        aggressiveness_map = {
            "low": 0,
            "moderate": 1,
            "high": 2,
            "veryhigh": 3,
        }
        rms_gate_map = {
            "low": -49.0,
            "medium": -52.0,
            "high": -55.0,
        }

        flags: list[bool] = []
        backend = "energy_gate"
        requested_engine = (vad_engine_override or "webrtcvad").strip().lower()

        detector = None
        webrtc_aggressiveness = aggressiveness_map[audio_config.noise_suppression_level]
        if vad_aggressiveness is not None:
            webrtc_aggressiveness = int(_clamp_float(vad_aggressiveness, webrtc_aggressiveness, 0, 3))
        allow_webrtc = (
            requested_engine != "energy"
            and webrtcvad is not None
            and (
                audio_config.mode in {"webrtc", "system_voice_processing"}
                or audio_config.enhancement_version == "v2"
            )
        )
        if allow_webrtc:
            try:
                detector = webrtcvad.Vad(webrtc_aggressiveness)
                backend = "webrtcvad"
            except Exception:
                detector = None

        rms_gate = rms_gate_map.get(audio_config.low_volume_boost, -52.0)
        peak_gate = rms_gate + 8.0
        webrtc_errors = 0
        for frame in frames:
            speech = False
            if detector is not None:
                try:
                    frame_pcm16 = self._float_to_pcm16(frame).tobytes()
                    speech = bool(detector.is_speech(frame_pcm16, 16000))
                except Exception:
                    webrtc_errors += 1
                    detector = None
                    backend = "energy_gate"

            if detector is None:
                rms_db = self._estimate_rms_dbfs(frame)
                peak_db = self._estimate_peak_dbfs(frame)
                speech = rms_db >= rms_gate or peak_db >= peak_gate

            flags.append(speech)

        speech_count = sum(1 for item in flags if item)
        stats = {
            "vad_backend": backend,
            "vad_frame_count": len(flags),
            "vad_speech_frames": speech_count,
            "vad_speech_ratio": round(speech_count / max(1, len(flags)), 4),
            "vad_rms_gate_dbfs": round(rms_gate, 2),
            "vad_peak_gate_dbfs": round(peak_gate, 2),
            "vad_engine_requested": requested_engine,
            "vad_aggressiveness": webrtc_aggressiveness,
        }
        if webrtc_errors > 0:
            stats["vad_webrtc_errors"] = webrtc_errors
        return flags, stats

    def _segment_by_vad(
        self,
        frames: list[np.ndarray],
        vad_flags: list[bool],
        anti_cutoff_pause_ms: int,
        frame_size: int,
        pre_roll_ms: int = 100,
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        if not frames or not vad_flags:
            return [], {"segment_count": 0}

        enter_speech_frames = 3
        pre_roll_frames = max(0, min(int(round(pre_roll_ms / 10.0)), 50))
        base_exit_silence_frames = max(20, min(int(round(anti_cutoff_pause_ms / 10.0)), 120))
        adaptive_window_frames = 60  # 600ms
        adaptive_min_exit_frames = max(12, min(18, int(round(base_exit_silence_frames * 0.5))))
        max_segment_frames = int((30 * 16000) / frame_size)

        in_segment = False
        speech_run = 0
        silence_run = 0
        recent_flags = deque(maxlen=adaptive_window_frames)
        pre_roll = deque(maxlen=pre_roll_frames)
        current_frames: list[np.ndarray] = []
        segments: list[np.ndarray] = []

        pre_roll_hits = 0
        forced_split_count = 0
        ignored_short_segments = 0
        adaptive_fast_exit_hits = 0

        for frame, is_speech in zip(frames, vad_flags):
            recent_flags.append(is_speech)
            pre_roll.append(frame)

            if not in_segment:
                speech_run = speech_run + 1 if is_speech else 0
                if speech_run >= enter_speech_frames:
                    in_segment = True
                    if len(pre_roll) > enter_speech_frames:
                        pre_roll_hits += 1
                    current_frames = list(pre_roll)
                    silence_run = 0
                    speech_run = 0
                continue

            current_frames.append(frame)
            silence_run = 0 if is_speech else silence_run + 1

            if len(current_frames) >= max_segment_frames:
                segments.append(np.concatenate(current_frames, axis=0))
                current_frames = []
                forced_split_count += 1
                silence_run = 0
                continue

            effective_exit_frames = base_exit_silence_frames
            stable_recent_silence = (
                len(recent_flags) >= adaptive_window_frames and not any(recent_flags)
            )
            if stable_recent_silence:
                effective_exit_frames = adaptive_min_exit_frames
                adaptive_fast_exit_hits += 1

            if silence_run >= effective_exit_frames:
                keep_frames = max(1, len(current_frames) - max(0, silence_run - 2))
                segment = np.concatenate(current_frames[:keep_frames], axis=0)
                if segment.size >= frame_size * 3:
                    segments.append(segment)
                else:
                    ignored_short_segments += 1
                current_frames = []
                in_segment = False
                speech_run = 0
                silence_run = 0

        if current_frames:
            tail = np.concatenate(current_frames, axis=0)
            if tail.size >= frame_size * 3:
                segments.append(tail)
            else:
                ignored_short_segments += 1

        average_duration_ms = (
            round(sum(segment.size for segment in segments) * 1000.0 / (16000.0 * len(segments)), 2)
            if segments
            else 0.0
        )
        stats = {
            "segment_count": len(segments),
            "segment_avg_duration_ms": average_duration_ms,
            "segment_pre_roll_hits": pre_roll_hits,
            "segment_forced_splits": forced_split_count,
            "segment_ignored_short": ignored_short_segments,
            "segment_enter_frames": enter_speech_frames,
            "segment_exit_silence_frames": base_exit_silence_frames,
            "segment_exit_silence_frames_fast": adaptive_min_exit_frames,
            "segment_adaptive_fast_exit_hits": adaptive_fast_exit_hits,
        }
        return segments, stats

    def _apply_low_volume_boost(
        self,
        signal: np.ndarray,
        audio_config: AudioEnhancementConfig,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        max_gain_map = {"low": 8.0, "medium": 12.0, "high": 18.0}
        target_rms_map = {"low": -26.0, "medium": -24.0, "high": -22.0}

        max_gain_db = max_gain_map.get(audio_config.low_volume_boost, 12.0)
        target_rms_db = target_rms_map.get(audio_config.low_volume_boost, -24.0)
        input_rms_db = self._estimate_rms_dbfs(signal)
        input_peak_db = self._estimate_peak_dbfs(signal)
        if input_rms_db >= (target_rms_db - 1.0) and input_peak_db >= -8.0:
            return signal.astype(np.float32, copy=False), {
                "gain_skipped": True,
                "gain_skip_reason": "input_within_target",
                "gain_target_rms_dbfs": round(target_rms_db, 2),
                "gain_max_allowed_db": round(max_gain_db, 2),
                "gain_avg_db": 0.0,
                "gain_peak_db": 0.0,
            }

        processed = np.empty_like(signal, dtype=np.float32)
        frame_size = 160  # 10ms
        gain_values: list[float] = []
        current_gain_db = 0.0

        for start in range(0, signal.size, frame_size):
            end = min(start + frame_size, signal.size)
            frame = signal[start:end]
            frame_rms_db = self._estimate_rms_dbfs(frame)

            desired_gain_db = max(0.0, min(max_gain_db, target_rms_db - frame_rms_db))
            attack = 0.25
            release = 0.08
            smoothing = attack if desired_gain_db > current_gain_db else release
            current_gain_db += smoothing * (desired_gain_db - current_gain_db)
            gain_values.append(current_gain_db)

            gain_linear = 10.0 ** (current_gain_db / 20.0)
            processed[start:end] = frame * gain_linear

        avg_gain = float(np.mean(gain_values)) if gain_values else 0.0
        max_gain = float(np.max(gain_values)) if gain_values else 0.0
        stats = {
            "gain_target_rms_dbfs": round(target_rms_db, 2),
            "gain_max_allowed_db": round(max_gain_db, 2),
            "gain_avg_db": round(avg_gain, 2),
            "gain_peak_db": round(max_gain, 2),
        }
        return processed.astype(np.float32, copy=False), stats

    def _apply_soft_limiter(self, signal: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        tanh_denom = math.tanh(1.6)
        limiter_hits = int(np.count_nonzero(np.abs(signal) > 0.97))
        limited = np.tanh(signal * 1.6) / tanh_denom
        out = np.clip(limited, -1.0, 1.0).astype(np.float32, copy=False)
        return out, {"limiter_trigger_count": limiter_hits}

    def _apply_high_pass_filter(
        self,
        signal: np.ndarray,
        cutoff_hz: float,
        sample_rate: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if signal.size < 2 or cutoff_hz <= 0:
            return signal, {"hpf_enabled": False}

        dt = 1.0 / float(sample_rate)
        rc = 1.0 / (2.0 * math.pi * cutoff_hz)
        alpha = rc / (rc + dt)

        output = np.empty_like(signal, dtype=np.float32)
        output[0] = signal[0]
        for i in range(1, signal.size):
            output[i] = alpha * (output[i - 1] + signal[i] - signal[i - 1])

        stats = {
            "hpf_enabled": True,
            "hpf_cutoff_hz": round(cutoff_hz, 2),
            "hpf_alpha": round(alpha, 6),
        }
        return output, stats

    def _apply_webrtc_apm_if_available(
        self,
        signal: np.ndarray,
        audio_config: AudioEnhancementConfig,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if audio_config.mode != "webrtc":
            return signal, {"apm_backend": "not_requested"}
        if WebRTCAudioProcessingModule is None:
            return signal, {"apm_backend": "unavailable"}

        ns_level_map = {"low": 0, "moderate": 1, "high": 2, "veryhigh": 3}
        agc_target_map = {"low": 24, "medium": 20, "high": 16}
        agc_level_map = {"low": 42, "medium": 58, "high": 74}

        ns_level = ns_level_map.get(audio_config.noise_suppression_level, 1)
        agc_target = agc_target_map.get(audio_config.low_volume_boost, 20)
        agc_level = agc_level_map.get(audio_config.low_volume_boost, 58)

        frame_size = 160  # 10ms @ 16kHz
        if signal.size < frame_size:
            return signal, {
                "apm_backend": "webrtc_apm",
                "apm_frames_processed": 0,
                "apm_ns_level": ns_level,
                "apm_agc_target": agc_target,
                "apm_agc_level": agc_level,
            }

        try:
            apm = WebRTCAudioProcessingModule(
                aec_type=0,
                enable_ns=True,
                agc_type=1,
                enable_vad=False,
            )
            apm.set_stream_format(16000, 1, 16000, 1)
            apm.set_ns_level(ns_level)
            apm.set_agc_target(agc_target)
            apm.set_agc_level(agc_level)
        except Exception as exc:
            return signal, {
                "apm_backend": "error",
                "apm_error": f"init_failed: {exc}",
            }

        processed = np.empty_like(signal, dtype=np.float32)
        processed_frames = 0
        for start in range(0, signal.size, frame_size):
            end = min(start + frame_size, signal.size)
            frame = signal[start:end]
            if frame.size < frame_size:
                frame_in = np.pad(frame, (0, frame_size - frame.size))
            else:
                frame_in = frame

            try:
                payload = self._float_to_pcm16(frame_in).tobytes()
                out_payload = apm.process_stream(payload)
                if isinstance(out_payload, str):
                    out_payload = out_payload.encode("latin1")
                out_frame = np.frombuffer(out_payload, dtype=np.int16).astype(np.float32) / 32768.0
                if out_frame.size < frame_size:
                    out_frame = np.pad(out_frame, (0, frame_size - out_frame.size))
            except Exception as exc:
                return signal, {
                    "apm_backend": "error",
                    "apm_error": f"process_failed: {exc}",
                }

            processed[start:end] = out_frame[: end - start]
            processed_frames += 1

        stats = {
            "apm_backend": "webrtc_apm",
            "apm_frames_processed": processed_frames,
            "apm_ns_level": ns_level,
            "apm_agc_target": agc_target,
            "apm_agc_level": agc_level,
        }
        return processed.astype(np.float32, copy=False), stats

    def _apply_low_volume_boost_and_limiter(
        self,
        signal: np.ndarray,
        audio_config: AudioEnhancementConfig,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        boosted, gain_stats = self._apply_low_volume_boost(signal, audio_config)
        limited, limiter_stats = self._apply_soft_limiter(boosted)
        stats = dict(gain_stats)
        stats.update(limiter_stats)
        return limited, stats

    def _load_wav_as_mono_float32_16k(self, audio_path: str) -> tuple[np.ndarray, dict[str, Any]]:
        t0 = time.perf_counter()
        with wave.open(audio_path, "rb") as wf:
            channels = int(wf.getnchannels())
            sample_rate = int(wf.getframerate())
            original_sample_rate = sample_rate
            sample_width = int(wf.getsampwidth())
            frame_count = int(wf.getnframes())
            raw = wf.readframes(frame_count)
        read_ms = (time.perf_counter() - t0) * 1000.0

        if sample_width == 2:
            data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif sample_width == 4:
            data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        elif sample_width == 3:
            pcm = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            signed = (
                pcm[:, 0].astype(np.int32)
                | (pcm[:, 1].astype(np.int32) << 8)
                | (pcm[:, 2].astype(np.int32) << 16)
            )
            signed = np.where((signed & 0x800000) != 0, signed - 0x1000000, signed)
            data = signed.astype(np.float32) / 8388608.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sample_width}")

        if channels > 1:
            usable = (data.size // channels) * channels
            data = data[:usable].reshape(-1, channels).mean(axis=1)

        resample_ms = 0.0
        if sample_rate != 16000:
            t1 = time.perf_counter()
            data = self._resample_linear(data, src_rate=sample_rate, dst_rate=16000)
            resample_ms = (time.perf_counter() - t1) * 1000.0
            sample_rate = 16000

        stats = {
            "input_sample_rate": original_sample_rate,
            "output_sample_rate": sample_rate,
            "input_channels": channels,
            "input_sample_width": sample_width,
            "input_frames": frame_count,
            "audio_load_ms": round(read_ms, 2),
            "audio_resample_ms": round(resample_ms, 2),
        }
        return np.clip(data.astype(np.float32, copy=False), -1.0, 1.0), stats

    def _write_wav_mono_16k_int16(self, path: Path, signal: np.ndarray) -> None:
        pcm16 = self._float_to_pcm16(signal)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm16.tobytes())

    def _split_into_frames(self, signal: np.ndarray, frame_size: int) -> list[np.ndarray]:
        if signal.size == 0:
            return []
        frames: list[np.ndarray] = []
        for start in range(0, signal.size, frame_size):
            end = min(start + frame_size, signal.size)
            frame = signal[start:end]
            if frame.size < frame_size:
                frame = np.pad(frame, (0, frame_size - frame.size))
            frames.append(frame.astype(np.float32, copy=False))
        return frames

    def _float_to_pcm16(self, signal: np.ndarray) -> np.ndarray:
        clipped = np.clip(signal, -1.0, 1.0)
        return np.round(clipped * 32767.0).astype(np.int16, copy=False)

    def _resample_linear(self, signal: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if signal.size == 0 or src_rate == dst_rate:
            return signal.astype(np.float32, copy=False)
        src_len = signal.size
        dst_len = max(1, int(round(src_len * float(dst_rate) / float(src_rate))))
        src_x = np.linspace(0.0, 1.0, num=src_len, endpoint=False)
        dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
        out = np.interp(dst_x, src_x, signal.astype(np.float64))
        return out.astype(np.float32, copy=False)

    def _estimate_rms_dbfs(self, signal: np.ndarray) -> float:
        if signal.size == 0:
            return -120.0
        rms = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))
        return 20.0 * math.log10(max(rms, 1e-7))

    def _estimate_peak_dbfs(self, signal: np.ndarray) -> float:
        if signal.size == 0:
            return -120.0
        peak = float(np.max(np.abs(signal)))
        return 20.0 * math.log10(max(peak, 1e-7))

    def _log_audio_enhancement_stats(self, stats: dict[str, Any], debug: bool) -> None:
        summary = {
            "profile": stats.get("inference_audio_profile", "standard"),
            "mode": stats.get("mode", "unknown"),
            "enhancement_version": stats.get("enhancement_version", "legacy"),
            "enhancement_mode": stats.get("enhancement_mode", "legacy"),
            "enabled": stats.get("enabled", False),
            "apm_backend": stats.get("apm_backend", "n/a"),
            "ns_backend": stats.get("ns_backend", "n/a"),
            "hpf": bool(stats.get("hpf_enabled", False)),
            "vad_backend": stats.get("vad_backend", "n/a"),
            "segments": stats.get("segment_count", stats.get("transcribe_segments", 1)),
            "speech_ratio": stats.get("vad_speech_ratio", 0.0),
            "speech_lufs": stats.get("speech_lufs", "n/a"),
            "applied_gain_db": stats.get("applied_gain_db", stats.get("gain_avg_db", 0.0)),
            "noise_estimate_db": stats.get("noise_estimate_db", "n/a"),
            "gain_avg_db": stats.get("gain_avg_db", 0.0),
            "limiter_hits": stats.get("limiter_trigger_count", 0),
            "limiter_reduction_db": stats.get("limiter_reduction_db", 0.0),
            "clipping_sample_ratio": stats.get("clipping_sample_ratio", 0.0),
            "audio_stop_to_vad_done_ms": stats.get("audio_stop_to_vad_done", 0.0),
            "enhancement_chain_ms": stats.get("enhancement_chain", 0.0),
            "preprocess_ms": stats.get("preprocess_total_ms", 0.0),
        }
        print(
            f"[audio-enhancement] {json.dumps(summary, ensure_ascii=False)}",
            flush=True,
        )
        if debug:
            print(
                f"[audio-enhancement-debug] {json.dumps(stats, ensure_ascii=False)}",
                flush=True,
            )

    def _generate_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        try_generation_lock: bool = False,
    ) -> str:
        acquired = self._generation_lock.acquire(blocking=not try_generation_lock)
        if not acquired:
            return ""
        try:
            model, tokenizer = self._ensure_llm_loaded()
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            output_parts: list[str] = []
            try:
                iterator = stream_generate(model, tokenizer, prompt, max_tokens=max_tokens)
            except TypeError:
                iterator = stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)

            for response in iterator:
                token = self._stream_chunk_to_text(response)
                if token:
                    output_parts.append(token)
            return "".join(output_parts).strip()
        finally:
            self._generation_lock.release()

    def _stream_generate_chat(self, messages: list[dict[str, str]], max_tokens: int):
        with self._generation_lock:
            model, tokenizer = self._ensure_llm_loaded()
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            try:
                iterator = stream_generate(model, tokenizer, prompt, max_tokens=max_tokens)
            except TypeError:
                iterator = stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)

            for chunk in iterator:
                token = self._stream_chunk_to_text(chunk)
                if token:
                    yield token

    def _stream_chunk_to_text(self, chunk: Any) -> str:
        if isinstance(chunk, str):
            return chunk
        if hasattr(chunk, "text"):
            value = getattr(chunk, "text")
            if isinstance(value, str):
                return value
        if isinstance(chunk, dict) and isinstance(chunk.get("text"), str):
            return chunk["text"]
        return str(chunk)

    def _sse_event(self, payload: Any) -> str:
        if isinstance(payload, str):
            body = payload
        else:
            body = json.dumps(payload, ensure_ascii=False)
        return f"data: {body}\n\n"

    def _response_payload(self, response: InferenceResponse) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump()  # pydantic v2
        return response.dict()  # pydantic v1 fallback

    def _start_style_learning_task(self, final_output: str) -> None:
        if not final_output.strip():
            return
        with self._lock:
            scheduled_at = self._last_active

        thread = threading.Thread(
            target=self._extract_and_merge_style_rules,
            args=(final_output, scheduled_at),
            name="ghosttype-style-learning",
            daemon=True,
        )
        thread.start()

    def _extract_and_merge_style_rules(self, final_output: str, scheduled_at: float) -> None:
        try:
            # Let interactive inference requests run first; style-learning is best-effort.
            time.sleep(4.0)

            with self._lock:
                # If new activity happened since scheduling, skip this background pass.
                if self._last_active > scheduled_at + 0.05:
                    return
                idle_elapsed = time.monotonic() - self._last_active
                if idle_elapsed < self._style_learning_idle_grace_seconds:
                    return

            rules_text = self._generate_chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract abstract writing style rules only. "
                            "Do not retain facts or content. Return JSON: "
                            '{"rules":["..."]}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Given this output text, infer only abstract style traits "
                            "(tone, formatting preference, sentence style).\n\n"
                            f"{final_output}"
                        ),
                    },
                ],
                max_tokens=96,
                try_generation_lock=True,
            )
            if not rules_text.strip():
                return

            extracted = extract_json_object(rules_text)
            new_rules = []
            if isinstance(extracted, dict):
                maybe_rules = extracted.get("rules") or []
                new_rules = [str(rule).strip() for rule in maybe_rules if str(rule).strip()]

            if not new_rules:
                return

            with self._lock:
                profile = self.get_style_profile()
                existing = [str(rule).strip() for rule in profile.get("rules", []) if str(rule).strip()]
                merged = existing[:]
                for rule in new_rules:
                    if rule not in merged:
                        merged.append(rule)
                profile["rules"] = merged[:20]
                profile["updated_at"] = utc_now_iso()
                self.style_profile_path.write_text(
                    json.dumps(profile, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception:
            # Style-learning is intentionally non-critical.
            return

    def _decide_search(self, selected_text: str, question: str) -> SearchDecision:
        prompt = (
            "Decide whether web search is needed to answer the user question using provided context. "
            "Return JSON only: {\"need_search\":true/false,\"query\":\"...\"}. "
            "Use need_search=false when context is sufficient."
        )
        answer = self._generate_chat(
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "Selected context:\n"
                        f"{selected_text}\n\n"
                        "Question:\n"
                        f"{question}"
                    ),
                },
            ],
            max_tokens=120,
        )
        payload = extract_json_object(answer) or {}
        need = bool(payload.get("need_search"))
        query = str(payload.get("query") or question).strip()
        return SearchDecision(need_search=need and bool(query), query=query)

    def _run_duckduckgo(self, query: str, max_results: int) -> list[dict[str, str]]:
        if DDGS is None:
            return []
        sources: list[dict[str, str]] = []
        try:
            with DDGS() as ddgs:
                for item in ddgs.text(query, max_results=max_results):
                    title = str(item.get("title") or "").strip()
                    href = str(item.get("href") or "").strip()
                    body = str(item.get("body") or "").strip()
                    if not title and not href:
                        continue
                    sources.append({"title": title, "url": href, "snippet": body})
        except Exception:
            return []
        return sources

    def run_dictate(self, req: DictateRequest) -> InferenceResponse:
        audio_path = normalize_audio_path(req.audio_path)
        max_tokens = self._clamp_max_tokens(req.max_tokens)
        audio_config = self._audio_config_from_request(req)
        with self._lock:
            self._apply_model_overrides(req.asr_model, req.llm_model)
            self._touch()
            asr_result = self._transcribe_audio(
                audio_path,
                language="auto",
                audio_config=audio_config,
            )
            raw_text = asr_result.text
            t_asr = asr_result.timing_ms.get("asr", 0.0)

            prompt = self._append_personalization_rules(
                self._resolve_system_prompt(req.system_prompt, self._dictate_system_prompt())
            )
            user_prompt = self._dictate_user_prompt(raw_text)
            output, t_llm = self._run_llm_sync(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
            )

        self._start_style_learning_task(output)
        return self._build_inference_response(
            mode="dictate",
            raw_text=raw_text,
            output_text=output,
            t_asr=t_asr,
            t_llm=t_llm,
            extra_timings=asr_result.timing_ms,
        )

    def stream_dictate(self, req: DictateRequest):
        audio_path = normalize_audio_path(req.audio_path)
        max_tokens = self._clamp_max_tokens(req.max_tokens)
        audio_config = self._audio_config_from_request(req)
        with self._lock:
            self._apply_model_overrides(req.asr_model, req.llm_model)
            self._touch()
            asr_result = self._transcribe_audio(
                audio_path,
                language="auto",
                audio_config=audio_config,
            )
            raw_text = asr_result.text
            t_asr = asr_result.timing_ms.get("asr", 0.0)

            prompt = self._append_personalization_rules(
                self._resolve_system_prompt(req.system_prompt, self._dictate_system_prompt())
            )
            user_prompt = self._dictate_user_prompt(raw_text)
            # Ensure model is loaded while we still hold the lock.
            self._ensure_llm_loaded()
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt},
            ]

        def event_iterator():
            yield from self._stream_mode_response(
                mode="dictate",
                raw_text=raw_text,
                messages=messages,
                max_tokens=max_tokens,
                t_asr=t_asr,
                extra_timings=asr_result.timing_ms,
            )

        return event_iterator()

    def run_ask(self, req: AskRequest) -> InferenceResponse:
        audio_path = normalize_audio_path(req.audio_path)
        max_tokens = self._clamp_max_tokens(req.max_tokens)
        audio_config = self._audio_config_from_request(req)
        with self._lock:
            self._apply_model_overrides(req.asr_model, req.llm_model)
            self._touch()
            asr_result = self._transcribe_audio(
                audio_path,
                language="auto",
                audio_config=audio_config,
            )
            question = asr_result.text
            t_asr = asr_result.timing_ms.get("asr", 0.0)

            used_search, web_sources, search_text = self._prepare_ask_search_context(req, question)
            prompt = self._append_personalization_rules(
                self._resolve_system_prompt(req.system_prompt, self._ask_system_prompt())
            )
            question_pack = self._ask_user_prompt(
                selected_text=req.selected_text,
                question=question,
                search_text=search_text,
            )
            output, t_llm = self._run_llm_sync(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question_pack},
                ],
                max_tokens=max_tokens,
            )

        self._start_style_learning_task(output)
        return self._build_inference_response(
            mode="ask",
            raw_text=question,
            output_text=output,
            t_asr=t_asr,
            t_llm=t_llm,
            used_web_search=used_search,
            web_sources=web_sources,
            extra_timings=asr_result.timing_ms,
        )

    def stream_ask(self, req: AskRequest):
        audio_path = normalize_audio_path(req.audio_path)
        max_tokens = self._clamp_max_tokens(req.max_tokens)
        audio_config = self._audio_config_from_request(req)
        with self._lock:
            self._apply_model_overrides(req.asr_model, req.llm_model)
            self._touch()
            asr_result = self._transcribe_audio(
                audio_path,
                language="auto",
                audio_config=audio_config,
            )
            question = asr_result.text
            t_asr = asr_result.timing_ms.get("asr", 0.0)

            used_search, web_sources, search_text = self._prepare_ask_search_context(req, question)
            prompt = self._append_personalization_rules(
                self._resolve_system_prompt(req.system_prompt, self._ask_system_prompt())
            )
            question_pack = self._ask_user_prompt(
                selected_text=req.selected_text,
                question=question,
                search_text=search_text,
            )
            # Ensure model is loaded while we still hold the lock.
            self._ensure_llm_loaded()
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": question_pack},
            ]

        def event_iterator():
            yield from self._stream_mode_response(
                mode="ask",
                raw_text=question,
                messages=messages,
                max_tokens=max_tokens,
                t_asr=t_asr,
                used_web_search=used_search,
                web_sources=web_sources,
                extra_timings=asr_result.timing_ms,
            )

        return event_iterator()

    def run_translate(self, req: TranslateRequest) -> InferenceResponse:
        audio_path = normalize_audio_path(req.audio_path)
        max_tokens = self._clamp_max_tokens(req.max_tokens)
        audio_config = self._audio_config_from_request(req)
        with self._lock:
            self._apply_model_overrides(req.asr_model, req.llm_model)
            self._touch()
            asr_result = self._transcribe_audio(
                audio_path,
                language="auto",
                audio_config=audio_config,
            )
            raw_text = asr_result.text
            t_asr = asr_result.timing_ms.get("asr", 0.0)

            prompt = self._append_personalization_rules(
                self._resolve_system_prompt(req.system_prompt, self._translate_system_prompt(req.target_language))
            )
            user_prompt = self._translate_user_prompt(raw_text)
            output, t_llm = self._run_llm_sync(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
            )

        self._start_style_learning_task(output)
        return self._build_inference_response(
            mode="translate",
            raw_text=raw_text,
            output_text=output,
            t_asr=t_asr,
            t_llm=t_llm,
            extra_timings=asr_result.timing_ms,
        )

    def stream_translate(self, req: TranslateRequest):
        audio_path = normalize_audio_path(req.audio_path)
        max_tokens = self._clamp_max_tokens(req.max_tokens)
        audio_config = self._audio_config_from_request(req)
        with self._lock:
            self._apply_model_overrides(req.asr_model, req.llm_model)
            self._touch()
            asr_result = self._transcribe_audio(
                audio_path,
                language="auto",
                audio_config=audio_config,
            )
            raw_text = asr_result.text
            t_asr = asr_result.timing_ms.get("asr", 0.0)

            prompt = self._append_personalization_rules(
                self._resolve_system_prompt(req.system_prompt, self._translate_system_prompt(req.target_language))
            )
            user_prompt = self._translate_user_prompt(raw_text)
            # Ensure model is loaded while we still hold the lock.
            self._ensure_llm_loaded()
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt},
            ]

        def event_iterator():
            yield from self._stream_mode_response(
                mode="translate",
                raw_text=raw_text,
                messages=messages,
                max_tokens=max_tokens,
                t_asr=t_asr,
                extra_timings=asr_result.timing_ms,
            )

        return event_iterator()

    def run_asr_chunk(self, req: ASRChunkRequest) -> ASRChunkResponse:
        audio_path = normalize_audio_path(req.audio_path)
        audio_config = self._audio_config_from_request(req)
        with self._lock:
            self._apply_model_overrides(req.asr_model, req.llm_model)
            self._touch()
            asr_result = self._transcribe_audio(
                audio_path,
                language="auto",
                audio_config=audio_config,
            )
        return ASRChunkResponse(
            text=asr_result.text,
            timing_ms=asr_result.timing_ms,
        )

    def stream_prepared_transcript(self, req: PreparedTranscriptRequest):
        raw_mode = str(req.mode or "dictate").strip().lower()
        if raw_mode not in {"dictate", "ask", "translate"}:
            raise HTTPException(status_code=400, detail=f"Unsupported mode: {req.mode}")

        raw_text = str(req.raw_text or "").strip()
        if not raw_text:
            raise HTTPException(status_code=400, detail="raw_text must not be empty.")

        max_tokens = self._clamp_max_tokens(req.max_tokens)

        def event_iterator():
            with self._lock:
                self._apply_model_overrides(req.asr_model, req.llm_model)
                self._touch()

                extra_timings = {
                    str(k): float(v)
                    for k, v in (req.timing_ms or {}).items()
                    if isinstance(v, (int, float))
                }
                t_asr = float(extra_timings.get("asr", 0.0))

                if raw_mode == "dictate":
                    prompt = self._append_personalization_rules(
                        self._resolve_system_prompt(req.system_prompt, self._dictate_system_prompt())
                    )
                    user_prompt = self._dictate_user_prompt(raw_text)
                    used_web_search = False
                    web_sources: list[dict[str, str]] = []
                    mode_name = "dictate"
                elif raw_mode == "ask":
                    ask_req = AskRequest(
                        audio_path="/tmp/ghosttype-prepared-transcript-placeholder.wav",
                        selected_text=req.selected_text,
                        web_search_enabled=req.web_search_enabled,
                        max_search_results=req.max_search_results,
                    )
                    used_web_search, web_sources, search_text = self._prepare_ask_search_context(ask_req, raw_text)
                    prompt = self._append_personalization_rules(
                        self._resolve_system_prompt(req.system_prompt, self._ask_system_prompt())
                    )
                    user_prompt = self._ask_user_prompt(
                        selected_text=req.selected_text,
                        question=raw_text,
                        search_text=search_text,
                    )
                    mode_name = "ask"
                else:
                    prompt = self._append_personalization_rules(
                        self._resolve_system_prompt(
                            req.system_prompt,
                            self._translate_system_prompt(req.target_language),
                        )
                    )
                    user_prompt = self._translate_user_prompt(raw_text)
                    used_web_search = False
                    web_sources = []
                    mode_name = "translate"

                self._ensure_llm_loaded()
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_prompt},
                ]

            yield from self._stream_mode_response(
                mode=mode_name,
                raw_text=raw_text,
                messages=messages,
                max_tokens=max_tokens,
                t_asr=t_asr,
                used_web_search=used_web_search,
                web_sources=web_sources,
                extra_timings=extra_timings,
            )

        return event_iterator()


def extract_json_object(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    if candidate.startswith("{") and candidate.endswith("}"):
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def normalize_audio_path(path_value: str) -> str:
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {path}")
    return str(path)


def create_app(runtime: ResidentModelRuntime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runtime.start()
        yield
        runtime.stop()

    app = FastAPI(title="GhostType Service", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(ASRRequestError)
    async def handle_asr_request_error(_: Request, exc: ASRRequestError):
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    @app.get("/health")
    def health() -> dict[str, Any]:
        return runtime.health()

    @app.post("/dictate", response_model=InferenceResponse)
    def dictate(req: DictateRequest) -> InferenceResponse:
        return runtime.run_dictate(req)

    @app.post("/dictate/stream")
    def dictate_stream(req: DictateRequest) -> StreamingResponse:
        return StreamingResponse(
            runtime.stream_dictate(req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.post("/ask", response_model=InferenceResponse)
    def ask(req: AskRequest) -> InferenceResponse:
        return runtime.run_ask(req)

    @app.post("/ask/stream")
    def ask_stream(req: AskRequest) -> StreamingResponse:
        return StreamingResponse(
            runtime.stream_ask(req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.post("/translate", response_model=InferenceResponse)
    def translate(req: TranslateRequest) -> InferenceResponse:
        return runtime.run_translate(req)

    @app.post("/translate/stream")
    def translate_stream(req: TranslateRequest) -> StreamingResponse:
        return StreamingResponse(
            runtime.stream_translate(req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.post("/asr/transcribe", response_model=ASRChunkResponse)
    def asr_transcribe(req: ASRChunkRequest) -> ASRChunkResponse:
        return runtime.run_asr_chunk(req)

    @app.post("/llm/stream")
    def llm_stream(req: PreparedTranscriptRequest) -> StreamingResponse:
        return StreamingResponse(
            runtime.stream_prepared_transcript(req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.post("/config/memory-timeout")
    def update_memory_timeout(req: MemoryTimeoutRequest) -> dict[str, Any]:
        runtime.set_idle_timeout(req.idle_timeout_seconds)
        return {"ok": True, "idle_timeout_seconds": runtime.get_idle_timeout()}

    @app.post("/release")
    def release() -> dict[str, Any]:
        runtime.release_models()
        return {"ok": True}

    @app.get("/dictionary")
    def get_dictionary() -> dict[str, Any]:
        items = runtime.get_dictionary_items()
        return {"items": items, "terms": runtime.get_dictionary_terms()}

    @app.post("/dictionary")
    def update_dictionary(req: DictionaryUpdateRequest) -> dict[str, Any]:
        terms = runtime.update_dictionary_terms(req.terms)
        return {"ok": True, "terms": terms}

    @app.get("/style-profile")
    def get_style_profile() -> dict[str, Any]:
        return runtime.get_style_profile()

    @app.post("/style/clear")
    def clear_style_profile() -> dict[str, Any]:
        return runtime.clear_style_profile()

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GhostType resident inference service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state-dir", default=str(Path(__file__).parent / "state"))
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--idle-timeout", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timeout = None if args.idle_timeout <= 0 else args.idle_timeout
    runtime = ResidentModelRuntime(
        state_dir=Path(args.state_dir).expanduser().resolve(),
        asr_model=args.asr_model,
        llm_model=args.llm_model,
        idle_timeout_seconds=timeout,
    )
    app = create_app(runtime)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    initialize_models()
    raise SystemExit(main())

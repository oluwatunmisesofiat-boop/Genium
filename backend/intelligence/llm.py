"""Swappable structured + multimodal Cloud LLM providers (Gemini & NVIDIA NIM)."""
from __future__ import annotations

import base64
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Protocol, Tuple, Type, TypeVar, runtime_checkable

import httpx
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
DEFAULT_GEMINI_MODEL = "gemini-4-hydrogen"
FALLBACK_GEMINI_MODELS = (
    "gemini-4-carbon",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
)

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
DEFAULT_NVIDIA_VISION_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
FALLBACK_NVIDIA_VISION_MODELS = (
    "meta/llama-3.2-90b-vision-instruct",
    "qwen/qwen2.5-vl-72b-instruct",
    "google/gemma-3-27b-it",
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _extract_json_object(raw_text: str) -> str:
    """Strip reasoning tags / markdown fences, repair trailing commas, and extract the JSON object."""
    cleaned = _THINK_BLOCK_RE.sub("", raw_text or "").strip()
    cleaned = _CODE_FENCE_RE.sub("", cleaned).strip()
    if not (cleaned.startswith("{") and cleaned.endswith("}")):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return cleaned


@runtime_checkable
class LLMProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        *,
        system_instruction: Optional[str] = None,
    ) -> T: ...

    def generate_structured_with_images(
        self,
        prompt: str,
        images: List[Tuple[bytes, str]],
        schema: Type[T],
        *,
        system_instruction: Optional[str] = None,
    ) -> T: ...


class GeminiProvider:
    """Google Gemini provider using the official `google-genai` SDK."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self._api_key = (
            api_key
            if api_key is not None
            else (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "")
        ).strip()
        self._model = (
            model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        ).strip()

    @property
    def model_name(self) -> str:
        return self._model

    def _client(self):
        if not self._api_key:
            raise ValueError(
                "Gemini API key missing: set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment"
            )
        from google import genai  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415

        http_options = types.HttpOptions(
            api_version="v1beta",
            timeout=90_000,
            retry_options=types.HttpRetryOptions(
                attempts=2,
                initial_delay=1.0,
                max_delay=3.0,
            ),
        )
        return genai.Client(api_key=self._api_key, http_options=http_options)

    def _call(
        self,
        contents,
        schema: Type[T],
        system_instruction: Optional[str],
    ) -> T:
        from google.genai import types  # noqa: PLC0415

        client = self._client()
        models_to_try = [self._model] + [m for m in FALLBACK_GEMINI_MODELS if m != self._model]
        last_exc: Optional[Exception] = None
        attempt_errors: List[str] = []
        resp = None
        try:
            for idx, model_name in enumerate(models_to_try):
                thinking_config = None
                if model_name.startswith(("gemini-3", "gemini-4")) and "lite" not in model_name:
                    thinking_config = types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
                elif model_name.startswith("gemini-2.5"):
                    thinking_config = types.ThinkingConfig(thinking_budget=0)
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    system_instruction=system_instruction,
                    temperature=0.1,
                    thinking_config=thinking_config,
                )
                try:
                    resp = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config,
                    )
                    self._model = model_name
                    break
                except Exception as exc:  # noqa: BLE001
                    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                    msg = str(exc)
                    short_err = msg.splitlines()[0][:140]
                    attempt_errors.append(f"{model_name}: {short_err}")
                    is_fallbackable = (
                        status in (404, 429, 500, 503, 504)
                        or "404" in msg
                        or "NOT_FOUND" in msg
                        or "500" in msg
                        or "INTERNAL" in msg
                        or "503" in msg
                        or "UNAVAILABLE" in msg
                        or "504" in msg
                        or "DEADLINE_EXCEEDED" in msg
                        or "429" in msg
                        or "timed out" in msg.lower()
                    )
                    if is_fallbackable and idx < len(models_to_try) - 1:
                        next_model = models_to_try[idx + 1]
                        print(
                            f"  [llm] {model_name} failed ({status or type(exc).__name__}); "
                            f"falling back to {next_model}...",
                            file=sys.stderr,
                            flush=True,
                        )
                        last_exc = exc
                        continue
                    if attempt_errors and len(attempt_errors) > 1:
                        raise RuntimeError(
                            "All Gemini models failed: " + " | ".join(attempt_errors)
                        ) from exc
                    raise
        finally:
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                close_fn()
        if resp is None:
            raise ValueError(f"All Gemini models failed: {last_exc}") from last_exc
        raw_text = getattr(resp, "text", None)
        if not raw_text:
            raise ValueError(f"Empty structured response from Gemini model {self._model!r}")
        return schema.model_validate_json(raw_text)

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        *,
        system_instruction: Optional[str] = None,
    ) -> T:
        return self._call(prompt, schema, system_instruction)

    def generate_structured_with_images(
        self,
        prompt: str,
        images: List[Tuple[bytes, str]],
        schema: Type[T],
        *,
        system_instruction: Optional[str] = None,
    ) -> T:
        from google.genai import types  # noqa: PLC0415

        parts = [types.Part.from_bytes(data=data, mime_type=mime) for data, mime in images]
        contents = [prompt, *parts]
        return self._call(contents, schema, system_instruction)


class NvidiaProvider:
    """NVIDIA NIM OpenAI-compatible provider (Nemotron 3.5 Lightning + Vision-Language)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        vision_model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 90.0,
    ) -> None:
        self._api_key = (
            api_key if api_key is not None else (os.environ.get("NVIDIA_API_KEY") or "")
        ).strip()
        self._model = (
            model or os.environ.get("NVIDIA_MODEL") or DEFAULT_NVIDIA_MODEL
        ).strip()
        self._vision_model = (
            vision_model
            or os.environ.get("NVIDIA_VISION_MODEL")
            or DEFAULT_NVIDIA_VISION_MODEL
        ).strip()
        self._base_url = (
            base_url or os.environ.get("NVIDIA_BASE_URL") or NVIDIA_BASE_URL
        ).rstrip("/")
        self._timeout = timeout

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def vision_model_name(self) -> str:
        return self._vision_model

    def _build_system_prompt(self, schema: Type[T], system_instruction: Optional[str]) -> str:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        prefix = (system_instruction or "").strip()
        suffix = (
            "Return ONLY a single valid JSON object (no markdown fences, no commentary) "
            f"conforming strictly to this JSON Schema:\n{schema_json}"
        )
        return f"{prefix}\n\n{suffix}" if prefix else suffix

    def _post_chat(
        self,
        *,
        models_to_try: List[str],
        messages: List[Dict[str, Any]],
        schema: Type[T],
        is_vision: bool = False,
    ) -> T:
        if not self._api_key:
            raise ValueError("NVIDIA API key missing: set NVIDIA_API_KEY in the environment")

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        attempt_errors: List[str] = []

        with httpx.Client(timeout=self._timeout) as client:
            for idx, model_name in enumerate(models_to_try):
                body: Dict[str, Any] = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "max_tokens": 8192,
                    "stream": False,
                }
                if not is_vision:
                    body["response_format"] = {"type": "json_object"}
                    body["chat_template_kwargs"] = {"enable_thinking": False}

                try:
                    resp = client.post(url, headers=headers, json=body)
                    if resp.status_code == 400 and "response_format" in body:
                        body.pop("response_format", None)
                        resp = client.post(url, headers=headers, json=body)
                    if resp.status_code in (404, 429, 500, 502, 503, 504):
                        detail = resp.text[:160].replace("\n", " ")
                        attempt_errors.append(f"{model_name}: HTTP {resp.status_code} {detail}")
                        if idx < len(models_to_try) - 1:
                            next_model = models_to_try[idx + 1]
                            print(
                                f"  [llm] {model_name} failed (HTTP {resp.status_code}); "
                                f"falling back to {next_model}...",
                                file=sys.stderr,
                                flush=True,
                            )
                            continue
                        resp.raise_for_status()
                    resp.raise_for_status()
                    data = resp.json()
                    choices = data.get("choices") or []
                    if not choices:
                        raise ValueError(f"Empty choices returned from NVIDIA model {model_name!r}")
                    content = (choices[0].get("message") or {}).get("content") or ""
                    if not content.strip():
                        raise ValueError(f"Empty content returned from NVIDIA model {model_name!r}")
                    json_payload = _extract_json_object(content)
                    parsed = schema.model_validate_json(json_payload)
                    if is_vision:
                        self._vision_model = model_name
                    else:
                        self._model = model_name
                    return parsed
                except (httpx.HTTPError, ValueError) as exc:
                    attempt_errors.append(f"{model_name}: {type(exc).__name__}: {str(exc)[:160]}")
                    if idx < len(models_to_try) - 1:
                        next_model = models_to_try[idx + 1]
                        print(
                            f"  [llm] {model_name} failed ({type(exc).__name__}); "
                            f"falling back to {next_model}...",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    if len(attempt_errors) > 1:
                        raise RuntimeError(
                            "All NVIDIA models failed: " + " | ".join(attempt_errors)
                        ) from exc
                    raise

        raise RuntimeError("All NVIDIA models failed: " + " | ".join(attempt_errors))

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        *,
        system_instruction: Optional[str] = None,
    ) -> T:
        messages = [
            {"role": "system", "content": self._build_system_prompt(schema, system_instruction)},
            {"role": "user", "content": prompt},
        ]
        return self._post_chat(
            models_to_try=[self._model],
            messages=messages,
            schema=schema,
            is_vision=False,
        )

    def generate_structured_with_images(
        self,
        prompt: str,
        images: List[Tuple[bytes, str]],
        schema: Type[T],
        *,
        system_instruction: Optional[str] = None,
    ) -> T:
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for raw_bytes, mime in images:
            b64 = base64.b64encode(raw_bytes).decode("ascii")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )
        messages = [
            {"role": "system", "content": self._build_system_prompt(schema, system_instruction)},
            {"role": "user", "content": user_content},
        ]
        models_to_try = [self._vision_model] + [
            m for m in FALLBACK_NVIDIA_VISION_MODELS if m != self._vision_model
        ]
        return self._post_chat(
            models_to_try=models_to_try,
            messages=messages,
            schema=schema,
            is_vision=True,
        )


def get_default_provider() -> LLMProvider:
    """Return the configured LLMProvider (honors GENIE_LLM_PROVIDER; auto-picks NVIDIA if NVIDIA_API_KEY is set)."""
    provider_name = (os.environ.get("GENIE_LLM_PROVIDER") or "").strip().lower()
    if not provider_name:
        provider_name = "nvidia" if (os.environ.get("NVIDIA_API_KEY") or "").strip() else "gemini"
    if provider_name == "nvidia":
        return NvidiaProvider()
    if provider_name == "gemini":
        return GeminiProvider()
    raise ValueError(f"Unsupported GENIE_LLM_PROVIDER: {provider_name!r}")

#!/usr/bin/env python3
"""Minimal DeepSeek chat client with structured (JSON) output.

DeepSeek exposes an OpenAI-compatible HTTP API. This helper sends a
system + user prompt, requests JSON output, and validates the response
against a pydantic model. It is a drop-in replacement for the small slice of
anthropic's ``messages.parse(output_format=...)`` that the sort scripts use.

Auth comes from the ``DEEPSEEK_API_KEY`` env var (or an explicit ``api_key``).
Override the endpoint/model with ``DEEPSEEK_BASE_URL`` / ``DEEPSEEK_MODEL``.
"""

import json
import os
import time
from typing import Optional, Type, TypeVar

import requests
from pydantic import BaseModel, ValidationError

DEFAULT_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

T = TypeVar("T", bound=BaseModel)


class DeepSeekError(RuntimeError):
    """Raised when a DeepSeek request fails (HTTP error or unparseable output)."""


class DeepSeekAuthError(DeepSeekError):
    """Fatal, run-wide failure: bad API key (401) or insufficient balance (402).

    These are not per-file problems — every subsequent call will fail the same
    way — so callers should abort the whole run rather than silently skipping
    files (which would otherwise let a billing outage corrupt a sort while the
    job still reports SUCCESS)."""


def _strip_code_fences(text: str) -> str:
    """Drop a ```json ... ``` wrapper if the model added one despite JSON mode."""
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


class DeepSeekClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        default_model: str = DEFAULT_MODEL,
        timeout_s: float = 60.0,
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or ""
        if not self.api_key:
            raise DeepSeekError("DEEPSEEK_API_KEY is not set")
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout_s = timeout_s

    def parse(
        self,
        *,
        system: str,
        user: str,
        schema: Type[T],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        retries: int = 2,
    ) -> T:
        """Return a validated instance of ``schema`` from a DeepSeek completion.

        Raises DeepSeekError on auth/billing failure, exhausted retries, or
        output that cannot be validated against the schema.
        """
        schema_json = json.dumps(schema.model_json_schema())
        sys_msg = (
            f"{system}\n\n"
            "Respond with ONLY a single valid JSON object (no markdown, no code "
            "fences, no commentary) that conforms to this JSON schema:\n"
            f"{schema_json}"
        )
        payload = {
            "model": model or self.default_model,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
            except requests.RequestException as e:
                last_err = e
                time.sleep(min(2 ** attempt, 8))
                continue

            # Fatal, non-retryable: bad key (401) or insufficient balance (402).
            if r.status_code in (401, 402):
                raise DeepSeekAuthError(f"DeepSeek auth/billing error {r.status_code}: {r.text[:300]}")
            # Transient: rate limit / server errors — back off and retry.
            if r.status_code == 429 or r.status_code >= 500:
                last_err = DeepSeekError(f"DeepSeek transient error {r.status_code}: {r.text[:200]}")
                time.sleep(min(2 ** attempt, 8))
                continue
            if r.status_code != 200:
                raise DeepSeekError(f"DeepSeek error {r.status_code}: {r.text[:300]}")

            try:
                content = r.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, ValueError) as e:
                raise DeepSeekError(f"Unexpected DeepSeek response shape: {e}")

            try:
                return schema.model_validate_json(_strip_code_fences(content))
            except ValidationError as e:
                last_err = e  # malformed object — retry once more
                continue

        raise DeepSeekError(f"DeepSeek request failed after {retries + 1} attempts: {last_err}")

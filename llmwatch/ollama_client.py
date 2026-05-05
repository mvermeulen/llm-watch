"""
Thin Ollama HTTP client for llm-watch.

Wraps the Ollama ``/api/chat`` endpoint using the ``requests`` library
(already a project dependency).  Designed for shared use by agents that
need local-LLM inference – currently the editor and the consolidator
(Phase 3 semantic deduplication).

Usage
-----
::

    from llmwatch.ollama_client import OllamaClient, OllamaUnavailableError

    client = OllamaClient(model="laguna-xs.2")
    reply = client.chat("Summarise the following text in one sentence: ...")
"""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_TIMEOUT = 120  # seconds


class OllamaUnavailableError(RuntimeError):
    """Raised when the Ollama server cannot be reached or returns an error."""


class OllamaClient:
    """
    Minimal client for the Ollama ``/api/chat`` endpoint.

    Parameters
    ----------
    model:
        Name of the Ollama model to use (e.g. ``"laguna-xs.2"``).
    base_url:
        Base URL of the Ollama server.  Defaults to ``http://localhost:11434``.
    timeout:
        Request timeout in seconds.
    """

    def __init__(
        self,
        model: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        prompt: str,
        system: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> str:
        """
        Send a single user message and return the model's text reply.

        Parameters
        ----------
        prompt:
            The user message content.
        system:
            Optional system prompt prepended to the conversation.
        extra_params:
            Additional keys merged into the request body (e.g.
            ``{"options": {"temperature": 0.2}}``).

        Returns
        -------
        str
            The assistant's reply text, stripped of leading/trailing whitespace.

        Raises
        ------
        OllamaUnavailableError
            If the request fails for any reason (connection error, non-2xx
            status, unexpected response shape).
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if extra_params:
            payload.update(extra_params)

        url = f"{self.base_url}/api/chat"
        logger.debug("OllamaClient: POST %s model=%s prompt_len=%d", url, self.model, len(prompt))

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise OllamaUnavailableError(
                f"Cannot connect to Ollama at {self.base_url}: {exc}"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaUnavailableError(
                f"Ollama request timed out after {self.timeout}s: {exc}"
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise OllamaUnavailableError(
                f"Ollama returned HTTP {response.status_code}: {exc}"
            ) from exc

        try:
            data = response.json()
            text: str = data["message"]["content"]
        except (KeyError, ValueError) as exc:
            raise OllamaUnavailableError(
                f"Unexpected Ollama response shape: {exc}"
            ) from exc

        logger.debug("OllamaClient: reply_len=%d", len(text))
        return text.strip()

    def is_available(self) -> bool:
        """
        Return ``True`` if the Ollama server responds to a health probe.

        Does not raise – safe to use as a pre-flight check.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

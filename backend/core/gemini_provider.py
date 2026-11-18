import json
import logging
import os
import random
import re
import time
from typing import Any, Callable, Iterator, Sequence
from google import genai
from google.genai import types
from google.genai.errors import APIError

logger = logging.getLogger(__name__)

class GeminiStreamChunk:
    """Standardized streaming chunk yielded by GeminiProvider."""
    def __init__(self, text: str | None = None, tool_calls: list[dict[str, Any]] | None = None):
        self.text = text
        self.tool_calls = tool_calls or []

class GeminiProvider:
    """Manages communication with Google Gemini API using google-genai SDK,

    with bounded exponential backoff, jitter, model fallback, and timeout protection.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        fallback_models: list[str] | None = None,
        client: Any = None,
        max_attempts: int = 3,
        backoff_delays: tuple[float, ...] = (2.0, 5.0, 10.0),
        request_timeout_seconds: float = 260.0,
        sleep_func: Callable[[float], None] = time.sleep,
    ):
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        self.model = model
        self.fallback_models = (
            fallback_models
            if fallback_models is not None
            else [m.strip() for m in os.getenv("ORBIT_FALLBACK_MODELS", "gemini-2.0-flash,gemini-1.5-flash").split(",") if m.strip()]
        )
        self._client = client
        self.max_attempts = max_attempts
        self.backoff_delays = backoff_delays
        self.request_timeout_seconds = request_timeout_seconds
        self.sleep_func = sleep_func
        self._confirmed_models: set[str] | None = None

    @property
    def client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured in backend environment variables. "
                "Please configure GEMINI_API_KEY in your hosting dashboard or .env file."
            )
        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _sanitize(self, message: str) -> str:
        """Strip sensitive credentials from error strings and log messages."""
        if not message:
            return ""
        sanitized = message
        if self.api_key and len(self.api_key) > 4:
            sanitized = sanitized.replace(self.api_key, "[REDACTED_API_KEY]")
        # Strip any query param or json field with api_key
        sanitized = re.sub(r'(api_?key["\':=\s]+)[\w\-]+', r'\1[REDACTED]', sanitized, flags=re.I)
        sanitized = re.sub(r'([?&]key=)[\w\-]+', r'\1[REDACTED]', sanitized, flags=re.I)
        return sanitized

    def _extract_status_code(self, exc: Exception) -> int | None:
        """Extract HTTP status code from exception if available."""
        code = getattr(exc, "code", None)
        if isinstance(code, int):
            return code
        msg = str(exc)
        match = re.search(r'\b(503|500|504|502|429|401|403|404|400)\b', msg)
        if match:
            return int(match.group(1))
        return None

    def _is_transient_error(self, exc: Exception, status_code: int | None) -> bool:
        """Determine whether an error should be retried.

        Never retry authentication, invalid model, permission, or quota errors.
        """
        # Non-retryable HTTP status codes
        if status_code in (400, 401, 403, 404, 429):
            return False

        msg = str(exc).lower()
        status_str = str(getattr(exc, "status", "")).upper()

        # Non-retryable status descriptors
        if status_str in ("RESOURCE_EXHAUSTED", "PERMISSION_DENIED", "UNAUTHENTICATED", "INVALID_ARGUMENT", "NOT_FOUND"):
            return False
        if "quota" in msg or "resource_exhausted" in msg or "api_key_invalid" in msg or "permission" in msg:
            return False
        if isinstance(exc, ValueError):
            return False

        # Transient HTTP status codes
        if status_code in (500, 502, 503, 504):
            return True

        # Transient status descriptors
        if status_str in ("UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED"):
            return True
        if "unavailable" in msg or "overloaded" in msg or "temporarily" in msg or "deadline exceeded" in msg:
            return True
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return True

        return False

    def get_confirmed_models(self) -> set[str]:
        """Fetch models confirmed available for this API key via Gemini API."""
        if self._confirmed_models is not None:
            return self._confirmed_models
        try:
            confirmed: set[str] = set()
            if hasattr(self.client, "models") and hasattr(self.client.models, "list"):
                pager = self.client.models.list()
                for m in pager:
                    name = getattr(m, "name", None)
                    if not name or not isinstance(name, str):
                        continue
                    actions = getattr(m, "supported_actions", None) or []
                    if actions and not any("generatecontent" in a.lower() for a in actions):
                        continue
                    clean_name = name.removeprefix("models/").strip().lower()
                    confirmed.add(clean_name)
                    confirmed.add(name.strip().lower())
            self._confirmed_models = confirmed
            return self._confirmed_models
        except Exception as exc:
            safe_msg = self._sanitize(str(exc))
            logger.warning(f"Could not retrieve confirmed models from Gemini API: {safe_msg}")
            self._confirmed_models = set()
            return self._confirmed_models

    def get_eligible_fallback_models(self, primary_model: str) -> list[str]:
        """Return fallback models confirmed available for this API key, excluding primary."""
        confirmed = self.get_confirmed_models()
        eligible: list[str] = []
        clean_primary = primary_model.removeprefix("models/").strip().lower()
        for cand in self.fallback_models:
            cand_clean = cand.removeprefix("models/").strip().lower()
            if cand_clean == clean_primary:
                continue
            # Only use fallback models confirmed available through the Gemini API for this key
            if not confirmed or (cand_clean in confirmed or cand.strip().lower() in confirmed):
                if cand not in eligible:
                    eligible.append(cand)
        return eligible

    def convert_tool_definitions(self, tool_definitions: Sequence[dict[str, Any]]) -> list[types.Tool]:
        """Convert tool definitions to Gemini types.Tool."""
        declarations = []
        for defn in tool_definitions:
            fn = defn.get("function", {}) if "function" in defn else defn
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                continue
            declarations.append(
                types.FunctionDeclaration(
                    name=name,
                    description=fn.get("description", ""),
                    parameters=fn.get("parameters") or None,
                )
            )
        if not declarations:
            return []
        return [types.Tool(function_declarations=declarations)]

    def history_to_contents(self, history: list[dict[str, Any]]) -> list[types.Content]:
        """Convert conversation history dicts to a list of Gemini types.Content."""
        contents: list[types.Content] = []
        i = 0
        while i < len(history):
            msg = history[i]
            role = msg.get("role")
            if role == "user":
                text = msg.get("content") or ""
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=text)],
                    )
                )
                i += 1
            elif role == "assistant":
                parts: list[types.Part] = []
                if msg.get("content"):
                    parts.append(types.Part.from_text(text=msg["content"]))
                calls = msg.get("tool_calls") or []
                for call in calls:
                    fn_name = call.get("function", {}).get("name") if "function" in call else call.get("name")
                    raw_args = (
                        call.get("function", {}).get("arguments")
                        if "function" in call
                        else (call.get("arguments") or call.get("args"))
                    )
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                    else:
                        args = raw_args or {}
                    parts.append(types.Part.from_function_call(name=fn_name, args=args))
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
                i += 1
            elif role == "tool":
                tool_parts: list[types.Part] = []
                while i < len(history) and history[i].get("role") == "tool":
                    t_msg = history[i]
                    name = t_msg.get("name", "unknown")
                    raw_content = t_msg.get("content", {})
                    if isinstance(raw_content, str):
                        try:
                            resp = json.loads(raw_content)
                        except (json.JSONDecodeError, TypeError):
                            resp = {"output": raw_content}
                    elif isinstance(raw_content, dict):
                        resp = raw_content
                    else:
                        resp = {"output": str(raw_content)}
                    tool_parts.append(types.Part.from_function_response(name=name, response=resp))
                    i += 1
                contents.append(types.Content(role="user", parts=tool_parts))
            else:
                i += 1
        return contents

    def _parse_raw_chunk(self, chunk: Any) -> GeminiStreamChunk:
        """Parse raw SDK chunk into standardized GeminiStreamChunk."""
        if isinstance(chunk, GeminiStreamChunk):
            return chunk
        chunk_text = None
        tool_calls = []
        if getattr(chunk, "candidates", None) and chunk.candidates[0].content and chunk.candidates[0].content.parts:
            text_pieces = []
            for part in chunk.candidates[0].content.parts:
                if getattr(part, "text", None):
                    text_pieces.append(part.text)
                if getattr(part, "function_call", None):
                    fc = part.function_call
                    tool_calls.append({
                        "name": fc.name,
                        "arguments": fc.args if isinstance(fc.args, dict) else {},
                    })
            if text_pieces:
                chunk_text = "".join(text_pieces)
        return GeminiStreamChunk(text=chunk_text, tool_calls=tool_calls)

    def stream_chat(
        self,
        history: list[dict[str, Any]],
        system_instruction: str,
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        model: str | None = None,
        deadline: float | None = None,
    ) -> Iterator[GeminiStreamChunk]:
        """Stream chunks from Gemini API with retries, exponential backoff, jitter, and fallback."""
        target_model = model or self.model
        request_deadline = deadline if deadline is not None else (time.time() + self.request_timeout_seconds)

        contents = self.history_to_contents(history)
        if not contents:
            return

        config_kwargs: dict[str, Any] = {
            "system_instruction": system_instruction,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if tools:
            gemini_tools = self.convert_tool_definitions(tools)
            if gemini_tools:
                config_kwargs["tools"] = gemini_tools

        models_to_try = [target_model]
        fallbacks_added = False
        model_idx = 0

        while model_idx < len(models_to_try):
            current_model = models_to_try[model_idx]

            for attempt in range(1, self.max_attempts + 1):
                now = time.time()
                if now >= request_deadline:
                    logger.error(
                        f"Request deadline reached for model '{current_model}' "
                        f"before attempt {attempt}/{self.max_attempts}. Terminating cleanly."
                    )
                    raise TimeoutError("Request time budget exceeded before completion.")

                config = types.GenerateContentConfig(**config_kwargs)

                try:
                    stream = self.client.models.generate_content_stream(
                        model=current_model,
                        contents=contents,
                        config=config,
                    )
                    # Pull the first chunk to ensure stream connection succeeded
                    stream_iter = iter(stream)
                    first_raw = next(stream_iter, None)

                    # If successful, yield chunks
                    if first_raw is not None:
                        yield self._parse_raw_chunk(first_raw)
                    for remaining_raw in stream_iter:
                        yield self._parse_raw_chunk(remaining_raw)
                    return  # Generation completed successfully

                except Exception as exc:
                    status_code = self._extract_status_code(exc)
                    is_transient = self._is_transient_error(exc, status_code)
                    safe_msg = self._sanitize(str(exc))

                    # Non-retryable errors abort immediately
                    if not is_transient:
                        logger.error(
                            f"Non-retryable Gemini error on model '{current_model}' "
                            f"(status={status_code}): {safe_msg}"
                        )
                        raise

                    # Transient error (503, 500, 504, network drop)
                    if attempt < self.max_attempts:
                        delay_idx = attempt - 1
                        base_delay = self.backoff_delays[delay_idx] if delay_idx < len(self.backoff_delays) else self.backoff_delays[-1]
                        # Delays of approx 2, 5, 10 seconds with jitter
                        jitter = random.uniform(-0.15 * base_delay, 0.15 * base_delay)
                        delay = max(0.5, base_delay + jitter)

                        if time.time() + delay >= request_deadline:
                            logger.error(
                                f"Request deadline would be exceeded by backoff sleep ({delay:.2f}s). "
                                f"Terminating model '{current_model}' after attempt {attempt}."
                            )
                            raise TimeoutError("Request time budget exceeded before completion.")

                        logger.warning(
                            f"Gemini API transient error on model '{current_model}' "
                            f"(attempt {attempt}/{self.max_attempts}, status={status_code}): {safe_msg}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        self.sleep_func(delay)
                        continue
                    else:
                        logger.warning(
                            f"Model '{current_model}' exhausted {self.max_attempts} attempts "
                            f"(status={status_code}): {safe_msg}"
                        )

                        # Check for confirmed fallback models if not yet added
                        if not fallbacks_added:
                            fallbacks_added = True
                            eligible_fallbacks = self.get_eligible_fallback_models(target_model)
                            for fb in eligible_fallbacks:
                                if fb not in models_to_try:
                                    models_to_try.append(fb)

                        # If another model is available, switch to it
                        if model_idx + 1 < len(models_to_try):
                            next_model = models_to_try[model_idx + 1]
                            logger.warning(
                                f"Switching from failing model '{current_model}' to "
                                f"confirmed fallback model '{next_model}'..."
                            )
                            break
                        else:
                            logger.error(
                                f"All retry attempts and model fallbacks exhausted. "
                                f"Final error on model '{current_model}' (status={status_code}): {safe_msg}"
                            )
                            raise

            model_idx += 1

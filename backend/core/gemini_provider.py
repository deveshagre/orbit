import json
import logging
import os
from typing import Any, Iterator, Sequence
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
    """Manages communication with Google Gemini API using google-genai SDK."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash", client: Any = None):
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        self.model = model
        self._client = client

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
                # In Gemini, consecutive function responses must be packaged in a single user content turn
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

    def stream_chat(
        self,
        history: list[dict[str, Any]],
        system_instruction: str,
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        model: str | None = None,
    ) -> Iterator[GeminiStreamChunk]:
        """Stream chunks from the Gemini API and yield standardized GeminiStreamChunk objects."""
        target_model = model or self.model
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

        config = types.GenerateContentConfig(**config_kwargs)

        try:
            stream = self.client.models.generate_content_stream(
                model=target_model,
                contents=contents,
                config=config,
            )
            for chunk in stream:
                chunk_text = None
                tool_calls = []
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    text_pieces = []
                    for part in chunk.candidates[0].content.parts:
                        if part.text:
                            text_pieces.append(part.text)
                        if part.function_call:
                            fc = part.function_call
                            tool_calls.append({
                                "name": fc.name,
                                "arguments": fc.args if isinstance(fc.args, dict) else {},
                            })
                    if text_pieces:
                        chunk_text = "".join(text_pieces)
                yield GeminiStreamChunk(text=chunk_text, tool_calls=tool_calls)
        except APIError as exc:
            logger.error(f"Gemini API error: {exc.message} (code {exc.code})")
            raise
        except Exception as exc:
            logger.error(f"Gemini request failed: {exc}")
            raise


import json
import re
import time
from typing import Any, Iterator
from core.gemini_provider import GeminiProvider, GeminiStreamChunk

SYSTEM_PROMPT = (
    "You are ORBIT, a concise and capable AI assistant powered by Google Gemini. "
    "Answer ordinary questions directly. For web search or actions requiring external tools, "
    "use the available tools and follow their results faithfully. "
    "Never claim an action succeeded unless confirmed by a tool. Do not reveal private reasoning. "
    "Note: ORBIT is running in a cloud deployment; opening local desktop applications and accessing "
    "the local Mac filesystem are disabled."
)

TOOL_SYSTEM_PROMPT = (
    "You are ORBIT, an AI agent powered by Google Gemini. The user requested an action. "
    "Call the appropriate available tool. Follow tool results faithfully and never claim an action "
    "succeeded unless confirmed. Do not reveal private reasoning. "
    "Note: ORBIT is running in a cloud deployment; opening local desktop applications and accessing "
    "the local Mac filesystem are disabled."
)

# Only explicit requests that need an external action enter the tool path.
_POLITE = r"(?:(?:please|could you|can you|would you|orbit)[, ]+)*"
_APP_ACTION = re.compile(
    rf"^\s*{_POLITE}(?:open|launch|start)\s+(?:the\s+)?(?:vs\s*code|visual studio code|safari|google chrome|chrome|notes|calendar|terminal)\b",
    re.I,
)
_WEB_ACTION = re.compile(
    r"\b(search|look up|browse|find online|on the web|on the internet|latest|current|today's|right now)\b",
    re.I,
)
_FILE_ACTION = re.compile(
    r"\b(list|show|read|create|make|write|save|find|open)\b.*\b(files?|folders?|workspace|directory|text file|notes\.txt)\b",
    re.I,
)
_FILE_OBJECT = re.compile(r"\b(files?|folders?|workspace|directory|text file|notes\.txt)\b", re.I)

def requires_tools(text: str) -> bool:
    """Conservative intent router: general questions stay on the direct chat path."""
    if _APP_ACTION.search(text):
        return True
    if _WEB_ACTION.search(text):
        return True
    return bool(
        _FILE_ACTION.search(text)
        or (
            _FILE_OBJECT.search(text)
            and re.match(rf"\s*{_POLITE}(?:list|show|read|create|make|write|save|find|open)\b", text, re.I)
        )
    )

class Agent:
    def __init__(
        self,
        provider: GeminiProvider | Any,
        registry,
        model: str | None = None,
        max_rounds: int = 5,
        max_context_messages: int = 8,
        options: dict | None = None,
    ):
        self.provider = provider
        self.registry = registry
        self.model = model or getattr(provider, "model", "gemini-2.5-flash")
        self.max_rounds = max_rounds
        self.max_context_messages = max_context_messages
        self.options = options or {"temperature": 0.2, "max_output_tokens": 1024}

    def _recent_context(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tail = history[-self.max_context_messages:]
        first_user = next((i for i, message in enumerate(tail) if message.get("role") == "user"), len(tail))
        return tail[first_user:]

    def stream_respond(self, history: list[dict[str, Any]], user_text: str) -> Iterator[dict[str, Any]]:
        """Yield SSE-ready events while preserving bounded context and Gemini tool calling."""
        started = time.perf_counter()
        tool_mode = requires_tools(user_text)
        mode = "tools" if tool_mode else "simple"
        history.append({"role": "user", "content": user_text})
        activities: list[dict[str, Any]] = []
        model_ms = 0.0
        first_token_ms = None
        yield {"type": "start", "mode": mode}

        for round_index in range(self.max_rounds if tool_mode else 1):
            recent_history = self._recent_context(history)
            system_instruction = TOOL_SYSTEM_PROMPT if tool_mode else SYSTEM_PROMPT
            tools = self.registry.definitions if tool_mode else None

            call_started = time.perf_counter()
            chunks = self.provider.stream_chat(
                history=recent_history,
                system_instruction=system_instruction,
                tools=tools,
                temperature=self.options.get("temperature", 0.2),
                max_output_tokens=self.options.get("max_output_tokens", 1024),
                model=self.model,
            )

            content_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []

            for chunk in chunks:
                text = getattr(chunk, "text", None)
                if text:
                    if first_token_ms is None:
                        first_token_ms = round((time.perf_counter() - started) * 1000)
                    content_parts.append(text)
                    yield {"type": "delta", "text": text}
                calls = getattr(chunk, "tool_calls", None) or []
                if calls:
                    tool_calls.extend(calls)

            model_ms += (time.perf_counter() - call_started) * 1000
            assistant_message: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            history.append(assistant_message)

            if not tool_calls:
                answer = assistant_message["content"]
                yield {
                    "type": "done",
                    "answer": answer,
                    "tool_activity": activities,
                    "timing": {
                        "mode": mode,
                        "total_ms": round((time.perf_counter() - started) * 1000),
                        "model_ms": round(model_ms),
                        "ttft_ms": first_token_ms,
                    },
                }
                break

            if len(activities) + len(tool_calls) > self.max_rounds:
                answer = "I reached the tool-call limit for this request. Please try a more focused request."
                yield {"type": "delta", "text": answer}
                yield {
                    "type": "done",
                    "answer": answer,
                    "tool_activity": activities,
                    "timing": {
                        "mode": mode,
                        "total_ms": round((time.perf_counter() - started) * 1000),
                        "model_ms": round(model_ms),
                        "ttft_ms": first_token_ms,
                    },
                }
                break

            for tool_call in tool_calls:
                name = tool_call.get("name") or "unknown"
                arguments = tool_call.get("arguments") or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (json.JSONDecodeError, TypeError):
                        arguments = None

                running = {"tool": name, "ok": True, "summary": "Running…", "status": "running"}
                yield {"type": "activity", "activity": running}

                result = self.registry.execute(name, arguments)
                activity = {
                    "tool": name,
                    "ok": result.get("ok") is True,
                    "summary": result.get("summary") or result.get("error") or "Completed",
                    "status": "done",
                }
                activities.append(activity)
                yield {"type": "activity", "activity": activity}
                history.append({
                    "role": "tool",
                    "content": json.dumps(result, ensure_ascii=False),
                    "name": name,
                })
        else:
            answer = "I couldn't complete the request within the tool-call limit."
            yield {"type": "delta", "text": answer}
            yield {
                "type": "done",
                "answer": answer,
                "tool_activity": activities,
                "timing": {
                    "mode": mode,
                    "total_ms": round((time.perf_counter() - started) * 1000),
                    "model_ms": round(model_ms),
                    "ttft_ms": first_token_ms,
                },
            }

        # Keep in-memory history bounded after the complete turn.
        if len(history) > self.max_context_messages * 2:
            del history[:-self.max_context_messages * 2]

    def respond(self, history: list[dict[str, Any]], user_text: str) -> tuple[str, list[dict[str, Any]]]:
        """Non-streaming compatibility helper used by callers and unit tests."""
        answer = ""
        activities = []
        for event in self.stream_respond(history, user_text):
            if event["type"] == "delta":
                answer += event["text"]
            elif event["type"] == "activity" and event["activity"].get("status") == "done":
                activities.append(event["activity"])
        return answer, activities

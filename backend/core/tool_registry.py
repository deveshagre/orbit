from typing import Any, Callable

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, tuple[dict[str, Any], Callable[[dict[str, Any]], dict[str, Any]]]] = {}

    def register(self, definition: dict[str, Any], handler: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        name = definition.get("function", {}).get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Tool definition must include function.name")
        self._tools[name] = (definition, handler)

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return [item[0] for item in self._tools.values()]

    def execute(self, name: str, arguments: Any) -> dict[str, Any]:
        if name not in self._tools:
            return {"ok": False, "error": f"Unknown tool: {name}"}
        if not isinstance(arguments, dict):
            return {"ok": False, "error": "Tool arguments must be a JSON object"}
        try:
            result = self._tools[name][1](arguments)
            return result if isinstance(result, dict) else {"ok": False, "error": "Tool returned an invalid result"}
        except Exception as exc:
            return {"ok": False, "error": str(exc) or exc.__class__.__name__}

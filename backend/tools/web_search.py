import logging
from ddgs import DDGS

logger = logging.getLogger(__name__)

def web_search(args: dict) -> dict:
    query = args.get("query")
    if not isinstance(query, str) or not query.strip() or len(query) > 400:
        return {"ok": False, "error": "query must be a non-empty string of at most 400 characters"}
    try:
        rows = list(DDGS(timeout=10).text(query.strip(), max_results=5))
        results = [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("href", "")),
                "snippet": str(r.get("body", "")),
            }
            for r in rows
        ]
        if not results:
            return {"ok": True, "summary": "No web search results found", "results": []}
        return {"ok": True, "summary": f"Found {len(results)} search results", "results": results}
    except Exception as exc:
        err_msg = str(exc)
        if "no results found" in err_msg.lower():
            return {"ok": True, "summary": "No web search results found", "results": []}
        logger.warning(f"Web search attempt failed: {exc}")
        return {"ok": False, "error": f"Web search failed: {exc.__class__.__name__}"}

DEFINITION = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the public web and return titles, URLs, and snippets. Use for current information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
}

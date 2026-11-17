import os
from pathlib import Path

def is_local_filesystem_enabled() -> bool:
    return os.getenv("ORBIT_ENABLE_LOCAL_FILESYSTEM", "false").lower() in ("true", "1")

class WorkspaceFiles:
    def __init__(self, workspace: Path, enabled: bool | None = None):
        self.workspace = Path(workspace).expanduser().resolve()
        self.enabled = is_local_filesystem_enabled() if enabled is None else enabled
        if self.enabled:
            self.workspace.mkdir(parents=True, exist_ok=True)

    def _target(self, raw: str) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("path must be a non-empty string")
        rel = Path(raw)
        if rel.is_absolute():
            raise ValueError("absolute paths are not allowed")
        target = (self.workspace / rel).resolve()
        if target != self.workspace and self.workspace not in target.parents:
            raise ValueError("path is outside the ORBIT workspace")
        return target

    def run(self, args: dict) -> dict:
        if not self.enabled:
            return {
                "ok": False,
                "error": (
                    "Local filesystem access is disabled in cloud deployments. "
                    "ORBIT is hosted on a remote cloud server and cannot access the local files on your Mac or personal computer."
                ),
            }
        try:
            return self._run(args)
        except (ValueError, OSError) as exc:
            return {"ok": False, "error": str(exc) or "Filesystem operation failed"}

    def _run(self, args: dict) -> dict:
        action = args.get("action")
        if action == "list":
            target = self._target(args.get("path", "."))
            if not target.exists() or not target.is_dir():
                return {"ok": False, "error": "Folder does not exist"}
            entries = [{"name": p.name, "type": "folder" if p.is_dir() else "file"} for p in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))]
            return {"ok": True, "summary": f"Listed {len(entries)} entries", "entries": entries}
        if action == "read":
            target = self._target(args.get("path"))
            if not target.is_file():
                return {"ok": False, "error": "File does not exist"}
            if target.stat().st_size > 200_000:
                return {"ok": False, "error": "File exceeds the 200 KB read limit"}
            try:
                content = target.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return {"ok": False, "error": "Only UTF-8 text files can be read"}
            return {"ok": True, "summary": f"Read {target.name}", "content": content}
        if action == "create_file":
            target = self._target(args.get("path"))
            content = args.get("content")
            if not isinstance(content, str) or len(content) > 200_000:
                return {"ok": False, "error": "content must be text of at most 200 KB"}
            if not target.parent.is_dir():
                return {"ok": False, "error": "Parent folder does not exist; create it first"}
            try:
                with target.open("x", encoding="utf-8") as f:
                    f.write(content)
            except FileExistsError:
                return {"ok": False, "error": "File already exists; it was not overwritten"}
            return {"ok": True, "summary": f"Created {target.name}", "path": str(target.relative_to(self.workspace))}
        if action == "create_folder":
            target = self._target(args.get("path"))
            try:
                target.mkdir(parents=False, exist_ok=False)
            except FileExistsError:
                return {"ok": False, "error": "Folder or file already exists"}
            except FileNotFoundError:
                return {"ok": False, "error": "Parent folder does not exist"}
            return {"ok": True, "summary": f"Created folder {target.name}", "path": str(target.relative_to(self.workspace))}
        return {"ok": False, "error": "action must be list, read, create_file, or create_folder"}

FILESYSTEM_DEFINITION = {
    "type": "function",
    "function": {
        "name": "filesystem",
        "description": (
            "List workspace files, read UTF-8 text files, or create text files/folders inside the ORBIT workspace. "
            "Note: This is a local-only tool disabled by default in cloud deployments."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "read", "create_file", "create_folder"]},
                "path": {"type": "string", "description": "Workspace-relative path"},
                "content": {"type": "string", "description": "Text for create_file"},
            },
            "required": ["action"],
        },
    },
}

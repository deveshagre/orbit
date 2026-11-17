import os
import platform
import subprocess

APP_ALLOWLIST = {
    "VS Code": "Visual Studio Code",
    "Safari": "Safari",
    "Google Chrome": "Google Chrome",
    "Notes": "Notes",
    "Calendar": "Calendar",
    "Terminal": "Terminal",
}

def is_application_launcher_enabled() -> bool:
    return os.getenv("ORBIT_ENABLE_APP_LAUNCHER", "false").lower() in ("true", "1")

def launch_application(args: dict) -> dict:
    if not is_application_launcher_enabled():
        return {
            "ok": False,
            "error": (
                "Application launching is a local-only feature and is disabled in cloud deployments. "
                "ORBIT is hosted on a remote cloud server and cannot launch applications on your local machine."
            ),
        }

    app = args.get("application")
    if not isinstance(app, str) or app not in APP_ALLOWLIST:
        return {"ok": False, "error": f"Application is not allowlisted. Available applications: {', '.join(APP_ALLOWLIST)}"}
    if platform.system() != "Darwin":
        return {"ok": False, "error": "Application launching is available only on macOS"}
    try:
        subprocess.Popen(["open", "-a", APP_ALLOWLIST[app]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"ok": True, "summary": f"Opened {app}"}
    except (OSError, subprocess.SubprocessError):
        return {"ok": False, "error": f"Could not open {app}"}

DEFINITION = {
    "type": "function",
    "function": {
        "name": "launch_application",
        "description": (
            "Open one common macOS application from the allowlist. "
            "Note: This is a local-only tool disabled by default in cloud deployments."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "application": {"type": "string", "enum": list(APP_ALLOWLIST)}
            },
            "required": ["application"],
        },
    },
}

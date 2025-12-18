"""Dashboard launcher with subprocess management.

Manages the Streamlit server as a subprocess, allowing the dashboard to be
started and stopped from the CLI or programmatically.

Uses a PID file to track the dashboard process across CLI invocations.
"""

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Global process reference (for in-process use)
_dashboard_process: Optional[subprocess.Popen] = None


def _get_pid_file() -> Path:
    """Get path to the dashboard PID file."""
    pid_dir = Path.home() / ".foundry-mcp"
    pid_dir.mkdir(parents=True, exist_ok=True)
    return pid_dir / "dashboard.pid"


def _save_pid(pid: int) -> None:
    """Save PID to file."""
    _get_pid_file().write_text(str(pid))


def _load_pid() -> Optional[int]:
    """Load PID from file, return None if not found or invalid."""
    pid_file = _get_pid_file()
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return None


def _clear_pid() -> None:
    """Remove the PID file."""
    pid_file = _get_pid_file()
    if pid_file.exists():
        pid_file.unlink()


def _is_process_running(pid: int) -> bool:
    """Check if a process with given PID is running."""
    try:
        os.kill(pid, 0)  # Signal 0 doesn't kill, just checks existence
        return True
    except (OSError, ProcessLookupError):
        return False


def launch_dashboard(
    host: str = "127.0.0.1",
    port: int = 8501,
    open_browser: bool = True,
) -> dict:
    """Launch the Streamlit dashboard server.

    Args:
        host: Host to bind to (default: localhost only)
        port: Port to run on (default: 8501)
        open_browser: Whether to open browser automatically

    Returns:
        dict with:
            - success: bool
            - url: Dashboard URL
            - pid: Process ID
            - message: Status message
    """
    global _dashboard_process

    # Check if already running
    status = get_dashboard_status()
    if status.get("running"):
        return {
            "success": True,
            "url": f"http://{host}:{port}",
            "pid": status["pid"],
            "message": "Dashboard already running",
        }

    # Path to the main Streamlit app
    app_path = Path(__file__).parent / "app.py"

    if not app_path.exists():
        return {
            "success": False,
            "message": f"Dashboard app not found at {app_path}",
        }

    # Build Streamlit command
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.address",
        host,
        "--browser.gatherUsageStats",
        "false",
        "--theme.base",
        "dark",
    ]

    if not open_browser:
        cmd.extend(["--server.headless", "true"])

    # Environment with dashboard mode flag
    env = {
        **os.environ,
        "FOUNDRY_MCP_DASHBOARD_MODE": "1",
    }

    # Pass config file path to dashboard subprocess so it can find error/metrics storage
    # If FOUNDRY_MCP_CONFIG_FILE is already set, it will be inherited from os.environ
    # Otherwise, find and pass the config file path explicitly
    if "FOUNDRY_MCP_CONFIG_FILE" not in env:
        for config_name in ["foundry-mcp.toml", ".foundry-mcp.toml"]:
            config_path = Path(config_name).resolve()
            if config_path.exists():
                env["FOUNDRY_MCP_CONFIG_FILE"] = str(config_path)
                logger.debug("Passing config file to dashboard: %s", config_path)
                break

    try:
        # Start subprocess
        _dashboard_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Brief wait to check for immediate failures
        time.sleep(1)
        poll = _dashboard_process.poll()
        if poll is not None:
            stderr = _dashboard_process.stderr.read().decode() if _dashboard_process.stderr else ""
            return {
                "success": False,
                "message": f"Dashboard failed to start (exit code {poll}): {stderr}",
            }

        pid = _dashboard_process.pid
        _save_pid(pid)
        logger.info("Dashboard started at http://%s:%s (pid=%s)", host, port, pid)

        return {
            "success": True,
            "url": f"http://{host}:{port}",
            "pid": pid,
            "message": "Dashboard started successfully",
        }

    except FileNotFoundError:
        return {
            "success": False,
            "message": "Streamlit not installed. Install with: pip install foundry-mcp[dashboard]",
        }
    except Exception as e:
        logger.exception("Failed to start dashboard")
        return {
            "success": False,
            "message": f"Failed to start dashboard: {e}",
        }


def stop_dashboard() -> dict:
    """Stop the running dashboard server.

    Returns:
        dict with:
            - success: bool
            - message: Status message
    """
    global _dashboard_process

    # First check in-memory process reference
    if _dashboard_process is not None:
        try:
            _dashboard_process.terminate()
            _dashboard_process.wait(timeout=5)
            pid = _dashboard_process.pid
            _dashboard_process = None
            _clear_pid()

            logger.info("Dashboard stopped (pid=%s)", pid)

            return {
                "success": True,
                "message": f"Dashboard stopped (pid={pid})",
            }

        except subprocess.TimeoutExpired:
            _dashboard_process.kill()
            _dashboard_process = None
            _clear_pid()
            return {
                "success": True,
                "message": "Dashboard killed (did not terminate gracefully)",
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to stop dashboard: {e}",
            }

    # Fall back to PID file (for cross-process stop)
    pid = _load_pid()
    if pid is None:
        return {
            "success": False,
            "message": "No dashboard process to stop",
        }

    if not _is_process_running(pid):
        _clear_pid()
        return {
            "success": False,
            "message": f"Dashboard process (pid={pid}) not running, cleaned up stale PID file",
        }

    try:
        os.kill(pid, signal.SIGTERM)

        # Wait for process to terminate
        for _ in range(50):  # 5 seconds total
            time.sleep(0.1)
            if not _is_process_running(pid):
                break
        else:
            # Force kill if still running
            os.kill(pid, signal.SIGKILL)

        _clear_pid()
        logger.info("Dashboard stopped (pid=%s)", pid)

        return {
            "success": True,
            "message": f"Dashboard stopped (pid={pid})",
        }

    except ProcessLookupError:
        _clear_pid()
        return {
            "success": True,
            "message": f"Dashboard process (pid={pid}) already terminated",
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to stop dashboard: {e}",
        }


def get_dashboard_status() -> dict:
    """Check if dashboard is running.

    Returns:
        dict with:
            - running: bool
            - pid: Process ID (if running)
            - exit_code: Exit code (if not running)
    """
    global _dashboard_process

    # First check in-memory process reference
    if _dashboard_process is not None:
        poll = _dashboard_process.poll()
        if poll is not None:
            _clear_pid()
            return {"running": False, "exit_code": poll}
        return {"running": True, "pid": _dashboard_process.pid}

    # Fall back to PID file (for cross-process status)
    pid = _load_pid()
    if pid is None:
        return {"running": False}

    if _is_process_running(pid):
        return {"running": True, "pid": pid}

    # Process not running but PID file exists - clean up
    _clear_pid()
    return {"running": False}

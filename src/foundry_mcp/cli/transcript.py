"""Parse Claude Code transcript files to extract token usage metrics.

Ported from claude-sdd-toolkit context_tracker module.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TokenMetrics:
    """Token usage metrics extracted from a transcript."""

    input_tokens: int
    output_tokens: int
    cached_tokens: int
    total_tokens: int
    context_length: int

    def context_percentage(self, max_context: int = 155000) -> float:
        """Calculate context usage percentage."""
        return (self.context_length / max_context) * 100 if max_context > 0 else 0.0


def is_clear_command(entry: dict) -> bool:
    """
    Check if a transcript entry is a /clear command.

    The /clear command resets the conversation context, so we should
    reset token counters when we encounter it.

    Args:
        entry: A parsed JSONL entry from the transcript

    Returns:
        True if this entry represents a /clear command
    """
    if entry.get("type") != "user":
        return False

    message = entry.get("message", {})
    content = message.get("content", "")

    # Handle both string content and list content
    if isinstance(content, str):
        return "<command-name>/clear</command-name>" in content

    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
                if "<command-name>/clear</command-name>" in text:
                    return True

    return False


def parse_transcript(transcript_path: str | Path) -> Optional[TokenMetrics]:
    """
    Parse a Claude Code transcript JSONL file and extract token metrics.

    Args:
        transcript_path: Path to the transcript JSONL file

    Returns:
        TokenMetrics object with aggregated token data, or None if parsing fails
    """
    transcript_path = Path(transcript_path)

    if not transcript_path.exists():
        return None

    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    context_length = 0

    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Check for /clear command - reset all counters
                if is_clear_command(entry):
                    input_tokens = 0
                    output_tokens = 0
                    cached_tokens = 0
                    context_length = 0
                    continue  # Don't process /clear entry itself

                # Skip sidechain and error messages
                if entry.get("isSidechain") or entry.get("isApiErrorMessage"):
                    continue

                # Extract usage data
                message = entry.get("message", {})
                usage = message.get("usage", {})

                if usage:
                    # Accumulate token counts
                    input_tokens += usage.get("input_tokens", 0)
                    output_tokens += usage.get("output_tokens", 0)

                    # Cached tokens come from both read and creation
                    cache_read = usage.get("cache_read_input_tokens", 0)
                    cache_creation = usage.get("cache_creation_input_tokens", 0)
                    cached_tokens += cache_read + cache_creation

                    # Context length is from the most recent valid entry
                    # (input tokens + cached tokens, excluding output)
                    context_length = (
                        usage.get("input_tokens", 0)
                        + usage.get("cache_read_input_tokens", 0)
                        + usage.get("cache_creation_input_tokens", 0)
                    )

    except Exception:
        return None

    total_tokens = input_tokens + output_tokens + cached_tokens

    return TokenMetrics(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        total_tokens=total_tokens,
        context_length=context_length,
    )


def find_transcript_by_marker(
    cwd: Path,
    marker: str,
    max_retries: int = 10,
) -> Optional[Path]:
    """
    Search transcripts for a specific SESSION_MARKER to identify current session.

    This function searches all .jsonl transcript files in the project directory
    for a specific marker string. The transcript containing that exact marker
    is the current session's transcript.

    To handle race conditions where the marker may not be flushed to disk yet,
    this function will retry with exponential backoff if the marker is not found
    on the first attempt.

    Transcripts are searched in reverse chronological order (most recently modified
    first) to prioritize active sessions over historical ones.

    If the exact CWD-based transcript directory doesn't exist, this function will
    search parent directories as a fallback to handle cases where commands are run
    from subdirectories of the project root.

    Args:
        cwd: Current working directory (used to find project-specific transcripts)
        marker: Specific marker to search for (e.g., "SESSION_MARKER_abc12345")
        max_retries: Maximum number of retry attempts (default: 10)

    Returns:
        Path to transcript containing the marker, or None if not found
    """
    # Build list of candidate transcript directories to search
    # Start with the exact CWD, then try parent directories
    candidate_dirs = []

    # Claude Code stores transcripts in project-specific directories
    current_path = cwd.resolve()
    while True:
        project_dir_name = str(current_path).replace("/", "-").replace("_", "-")
        transcript_dir = Path.home() / ".claude" / "projects" / project_dir_name
        if transcript_dir.exists():
            candidate_dirs.append(transcript_dir)

        # Stop at root or after checking enough parents
        if current_path.parent == current_path or len(candidate_dirs) >= 5:
            break
        current_path = current_path.parent

    if not candidate_dirs:
        return None

    # Extended retry with exponential backoff, capped at 30 seconds total
    # Delays: 100ms, 200ms, 400ms, 800ms, 1.6s, 3.2s, 6.4s, 12.8s, ...
    delays = [min(0.1 * (2**i), 10.0) for i in range(max_retries)]

    for attempt in range(max_retries):
        current_time = time.time()

        # Search all candidate directories (prioritizing exact CWD match first)
        for transcript_dir in candidate_dirs:
            try:
                # Get all recent transcript files
                transcript_files = []
                for transcript_path in transcript_dir.glob("*.jsonl"):
                    try:
                        mtime = transcript_path.stat().st_mtime
                        # Only check recent transcripts (modified in last 24 hours)
                        if (current_time - mtime) > 86400:
                            continue
                        transcript_files.append((transcript_path, mtime))
                    except (OSError, IOError):
                        continue

                # Sort by modification time (most recent first)
                # This prioritizes the active session over historical transcripts
                transcript_files.sort(key=lambda x: x[1], reverse=True)

                # Search through transcripts in order of recency
                for transcript_path, _ in transcript_files:
                    try:
                        # Search for the specific marker in the transcript
                        with open(transcript_path, "r", encoding="utf-8") as f:
                            for line in f:
                                if marker in line:
                                    return transcript_path
                    except (OSError, IOError, UnicodeDecodeError):
                        continue
            except (OSError, IOError):
                continue

        # If not found and we have retries left, wait before next attempt
        if attempt < max_retries - 1:
            time.sleep(delays[attempt])

    return None

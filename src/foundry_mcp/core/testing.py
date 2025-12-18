"""
Testing operations for foundry-mcp.
Provides functions for running tests and test discovery.

Supports multiple test runners (pytest, go, npm, jest, etc.) via configuration.
"""

import re
import subprocess
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from foundry_mcp.config import RunnerConfig, TestConfig


# Schema version for compatibility tracking
SCHEMA_VERSION = "1.0.0"


# Default runner configurations (used when no TOML config provided)
DEFAULT_RUNNERS: Dict[str, Dict[str, Any]] = {
    "pytest": {
        "command": ["python", "-m", "pytest"],
        "run_args": ["-v", "--tb=short"],
        "discover_args": ["--collect-only", "-q"],
        "pattern": "test_*.py",
        "timeout": 300,
    },
    "go": {
        "command": ["go", "test"],
        "run_args": ["-v"],
        "discover_args": ["-list", ".*"],
        "pattern": "*_test.go",
        "timeout": 300,
    },
    "npm": {
        "command": ["npm", "test"],
        "run_args": ["--"],
        "discover_args": [],
        "pattern": "*.test.js",
        "timeout": 300,
    },
    "jest": {
        "command": ["npx", "jest"],
        "run_args": ["--verbose"],
        "discover_args": ["--listTests"],
        "pattern": "*.test.{js,ts,jsx,tsx}",
        "timeout": 300,
    },
    "make": {
        "command": ["make", "test"],
        "run_args": [],
        "discover_args": [],
        "pattern": "*",
        "timeout": 300,
    },
}


# Presets for common test configurations
TEST_PRESETS = {
    "quick": {
        "timeout": 60,
        "verbose": False,
        "fail_fast": True,
        "markers": "not slow",
    },
    "full": {
        "timeout": 300,
        "verbose": True,
        "fail_fast": False,
        "markers": None,
    },
    "unit": {
        "timeout": 120,
        "verbose": True,
        "fail_fast": False,
        "markers": "unit",
        "pattern": "test_*.py",
    },
    "integration": {
        "timeout": 300,
        "verbose": True,
        "fail_fast": False,
        "markers": "integration",
    },
    "smoke": {
        "timeout": 30,
        "verbose": False,
        "fail_fast": True,
        "markers": "smoke",
    },
}


# Data structures

@dataclass
class TestResult:
    """
    Result of a single test.
    """
    name: str
    outcome: str  # passed, failed, skipped, error
    duration: float = 0.0
    message: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


@dataclass
class TestRunResult:
    """
    Result of a test run.
    """
    success: bool
    execution_id: str = ""
    schema_version: str = SCHEMA_VERSION
    timestamp: str = ""
    duration: float = 0.0
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    tests: List[TestResult] = field(default_factory=list)
    command: str = ""
    cwd: str = ""
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.execution_id:
            self.execution_id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class DiscoveredTest:
    """
    A discovered test.
    """
    name: str
    file_path: str
    line_number: Optional[int] = None
    markers: List[str] = field(default_factory=list)
    docstring: Optional[str] = None


@dataclass
class TestDiscoveryResult:
    """
    Result of test discovery.
    """
    success: bool
    schema_version: str = SCHEMA_VERSION
    timestamp: str = ""
    total: int = 0
    tests: List[DiscoveredTest] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.total = len(self.tests)


# Abstract test runner interface


class BaseTestRunner(ABC):
    """Abstract base class for test runners."""

    @abstractmethod
    def build_run_command(
        self,
        target: Optional[str] = None,
        verbose: bool = True,
        fail_fast: bool = False,
        extra_args: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Build the command to run tests."""
        pass

    @abstractmethod
    def build_discover_command(
        self,
        target: Optional[str] = None,
        pattern: str = "*",
    ) -> List[str]:
        """Build the command to discover tests."""
        pass

    @abstractmethod
    def parse_run_output(
        self,
        stdout: str,
        stderr: str,
        returncode: int,
    ) -> tuple:
        """Parse test run output. Returns (tests, passed, failed, skipped, errors)."""
        pass

    @abstractmethod
    def parse_discover_output(self, stdout: str) -> tuple:
        """Parse test discovery output. Returns (tests, test_files)."""
        pass

    @property
    @abstractmethod
    def default_timeout(self) -> int:
        """Default timeout in seconds."""
        pass

    @property
    @abstractmethod
    def not_found_error(self) -> str:
        """Error message when the runner is not found."""
        pass


class PytestRunner(BaseTestRunner):
    """Test runner for pytest-based projects."""

    def build_run_command(
        self,
        target: Optional[str] = None,
        verbose: bool = True,
        fail_fast: bool = False,
        extra_args: Optional[List[str]] = None,
        markers: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        cmd = ["python", "-m", "pytest"]

        if target:
            cmd.append(target)

        if verbose:
            cmd.append("-v")

        if fail_fast:
            cmd.append("-x")

        if markers:
            cmd.extend(["-m", markers])

        cmd.append("--tb=short")

        if extra_args:
            cmd.extend(extra_args)

        return cmd

    def build_discover_command(
        self,
        target: Optional[str] = None,
        pattern: str = "test_*.py",
    ) -> List[str]:
        cmd = ["python", "-m", "pytest", "--collect-only", "-q"]
        if target:
            cmd.append(target)
        return cmd

    def parse_run_output(
        self,
        stdout: str,
        stderr: str,
        returncode: int,
    ) -> tuple:
        """Parse pytest output to extract test results."""
        tests = []
        passed = 0
        failed = 0
        skipped = 0
        errors = 0

        lines = stdout.split("\n")

        for line in lines:
            line = line.strip()

            # Parse individual test results
            if "::" in line:
                if " PASSED" in line:
                    name = line.split(" PASSED")[0].strip()
                    tests.append(TestResult(name=name, outcome="passed"))
                    passed += 1
                elif " FAILED" in line:
                    name = line.split(" FAILED")[0].strip()
                    tests.append(TestResult(name=name, outcome="failed"))
                    failed += 1
                elif " SKIPPED" in line:
                    name = line.split(" SKIPPED")[0].strip()
                    tests.append(TestResult(name=name, outcome="skipped"))
                    skipped += 1
                elif " ERROR" in line:
                    name = line.split(" ERROR")[0].strip()
                    tests.append(TestResult(name=name, outcome="error"))
                    errors += 1

            # Parse summary line
            if "passed" in line.lower() and (
                "failed" in line.lower()
                or "error" in line.lower()
                or "skipped" in line.lower()
            ):
                passed_match = re.search(r"(\d+) passed", line)
                failed_match = re.search(r"(\d+) failed", line)
                skipped_match = re.search(r"(\d+) skipped", line)
                error_match = re.search(r"(\d+) error", line)

                if passed_match:
                    passed = int(passed_match.group(1))
                if failed_match:
                    failed = int(failed_match.group(1))
                if skipped_match:
                    skipped = int(skipped_match.group(1))
                if error_match:
                    errors = int(error_match.group(1))

        return tests, passed, failed, skipped, errors

    def parse_discover_output(self, stdout: str) -> tuple:
        """Parse pytest --collect-only output."""
        tests = []
        test_files: set[str] = set()

        for line in stdout.split("\n"):
            line = line.strip()
            if "::" in line and not line.startswith("="):
                parts = line.split("::")
                if parts:
                    file_path = parts[0]
                    test_files.add(file_path)
                    tests.append(DiscoveredTest(name=line, file_path=file_path))

        return tests, list(test_files)

    @property
    def default_timeout(self) -> int:
        return 300

    @property
    def not_found_error(self) -> str:
        return "pytest not found. Install with: pip install pytest"


class GenericRunner(BaseTestRunner):
    """Generic test runner that uses RunnerConfig from TOML configuration."""

    def __init__(
        self,
        command: List[str],
        run_args: Optional[List[str]] = None,
        discover_args: Optional[List[str]] = None,
        pattern: str = "*",
        timeout: int = 300,
        runner_name: str = "generic",
    ):
        self.command = command
        self.run_args = run_args or []
        self.discover_args = discover_args or []
        self.pattern = pattern
        self.timeout = timeout
        self.runner_name = runner_name

    @classmethod
    def from_runner_config(
        cls, config: "RunnerConfig", runner_name: str = "generic"
    ) -> "GenericRunner":
        """Create a GenericRunner from a RunnerConfig object."""
        return cls(
            command=list(config.command),
            run_args=list(config.run_args),
            discover_args=list(config.discover_args),
            pattern=config.pattern,
            timeout=config.timeout,
            runner_name=runner_name,
        )

    @classmethod
    def from_default(cls, runner_name: str) -> "GenericRunner":
        """Create a GenericRunner from DEFAULT_RUNNERS."""
        if runner_name not in DEFAULT_RUNNERS:
            raise ValueError(f"Unknown default runner: {runner_name}")
        cfg = DEFAULT_RUNNERS[runner_name]
        return cls(
            command=list(cfg["command"]),
            run_args=list(cfg.get("run_args", [])),
            discover_args=list(cfg.get("discover_args", [])),
            pattern=cfg.get("pattern", "*"),
            timeout=cfg.get("timeout", 300),
            runner_name=runner_name,
        )

    def build_run_command(
        self,
        target: Optional[str] = None,
        verbose: bool = True,
        fail_fast: bool = False,
        extra_args: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        cmd = list(self.command) + list(self.run_args)
        if target:
            cmd.append(target)
        if extra_args:
            cmd.extend(extra_args)
        return cmd

    def build_discover_command(
        self,
        target: Optional[str] = None,
        pattern: str = "*",
    ) -> List[str]:
        cmd = list(self.command) + list(self.discover_args)
        if target:
            cmd.append(target)
        return cmd

    def parse_run_output(
        self,
        stdout: str,
        stderr: str,
        returncode: int,
    ) -> tuple:
        """Parse generic test output - basic heuristics."""
        tests: List[TestResult] = []
        passed = 0
        failed = 0
        skipped = 0
        errors = 0

        # Go test output parsing
        if self.runner_name == "go":
            for line in stdout.split("\n"):
                line = line.strip()
                if line.startswith("--- PASS:"):
                    name = line.split("--- PASS:")[1].split()[0]
                    tests.append(TestResult(name=name, outcome="passed"))
                    passed += 1
                elif line.startswith("--- FAIL:"):
                    name = line.split("--- FAIL:")[1].split()[0]
                    tests.append(TestResult(name=name, outcome="failed"))
                    failed += 1
                elif line.startswith("--- SKIP:"):
                    name = line.split("--- SKIP:")[1].split()[0]
                    tests.append(TestResult(name=name, outcome="skipped"))
                    skipped += 1
            # If no individual tests parsed, check return code
            if not tests:
                if returncode == 0:
                    passed = 1
                else:
                    failed = 1

        # Jest/npm output parsing
        elif self.runner_name in ("jest", "npm"):
            for line in stdout.split("\n"):
                line = line.strip()
                if "✓" in line or "PASS" in line:
                    passed += 1
                elif "✕" in line or "FAIL" in line:
                    failed += 1
                elif "○" in line or "skipped" in line.lower():
                    skipped += 1
            if passed == 0 and failed == 0:
                if returncode == 0:
                    passed = 1
                else:
                    failed = 1

        # Generic fallback - just check return code
        else:
            if returncode == 0:
                passed = 1
            else:
                failed = 1

        return tests, passed, failed, skipped, errors

    def parse_discover_output(self, stdout: str) -> tuple:
        """Parse generic discovery output."""
        tests: List[DiscoveredTest] = []
        test_files: set[str] = set()

        for line in stdout.split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("="):
                # Try to extract file path
                if "/" in line or "\\" in line:
                    # Looks like a file path
                    file_path = line.split()[0] if " " in line else line
                    test_files.add(file_path)
                    tests.append(DiscoveredTest(name=line, file_path=file_path))
                elif line:
                    tests.append(DiscoveredTest(name=line, file_path=""))

        return tests, list(test_files)

    @property
    def default_timeout(self) -> int:
        return self.timeout

    @property
    def not_found_error(self) -> str:
        cmd_name = self.command[0] if self.command else "test runner"
        return f"{cmd_name} not found. Ensure it is installed and in PATH."


def get_runner(
    runner_name: Optional[str] = None,
    test_config: Optional["TestConfig"] = None,
) -> BaseTestRunner:
    """Factory function to get the appropriate test runner.

    Args:
        runner_name: Name of the runner to use. If None, uses default_runner from config.
        test_config: TestConfig from foundry-mcp.toml. If None, uses DEFAULT_RUNNERS.

    Returns:
        BaseTestRunner instance.

    Raises:
        ValueError: If the specified runner is not found.
    """
    # Determine which runner to use
    if runner_name is None:
        if test_config is not None:
            runner_name = test_config.default_runner
        else:
            runner_name = "pytest"

    # Special case: pytest always uses the optimized PytestRunner
    if runner_name == "pytest":
        return PytestRunner()

    # Check if runner is defined in test_config
    if test_config is not None:
        runner_cfg = test_config.get_runner(runner_name)
        if runner_cfg is not None:
            return GenericRunner.from_runner_config(runner_cfg, runner_name)

    # Fall back to DEFAULT_RUNNERS
    if runner_name in DEFAULT_RUNNERS:
        return GenericRunner.from_default(runner_name)

    # List available runners for error message
    available = list(DEFAULT_RUNNERS.keys())
    if test_config is not None:
        available.extend(test_config.runners.keys())
    available = sorted(set(available))

    raise ValueError(
        f"Unknown runner: {runner_name}. Available runners: {', '.join(available)}"
    )


def get_available_runners(test_config: Optional["TestConfig"] = None) -> List[str]:
    """Get list of available runner names.

    Args:
        test_config: Optional TestConfig for custom runners.

    Returns:
        List of available runner names.
    """
    runners = list(DEFAULT_RUNNERS.keys())
    if test_config is not None:
        runners.extend(test_config.runners.keys())
    return sorted(set(runners))


# Main test runner


class TestRunner:
    """
    Test runner that supports multiple backends (pytest, go, npm, etc.).
    """

    def __init__(
        self,
        workspace: Optional[Path] = None,
        runner: Optional[BaseTestRunner] = None,
    ):
        """
        Initialize test runner.

        Args:
            workspace: Repository root (defaults to current directory)
            runner: Test runner backend (defaults to PytestRunner)
        """
        self.workspace = workspace or Path.cwd()
        self._runner = runner or PytestRunner()

    def run_tests(
        self,
        target: Optional[str] = None,
        preset: Optional[str] = None,
        timeout: Optional[int] = None,
        verbose: bool = True,
        fail_fast: bool = False,
        markers: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
    ) -> TestRunResult:
        """
        Run tests using the configured test runner backend.

        Args:
            target: Test target (file, directory, or test name)
            preset: Use a preset configuration (quick, full, unit, integration, smoke)
            timeout: Timeout in seconds (defaults to runner's default)
            verbose: Enable verbose output
            fail_fast: Stop on first failure
            markers: Pytest markers expression (only applicable for pytest runner)
            extra_args: Additional arguments passed to the runner

        Returns:
            TestRunResult with test outcomes
        """
        # Apply preset if specified
        if preset and preset in TEST_PRESETS:
            preset_config = TEST_PRESETS[preset]
            timeout = preset_config.get("timeout", timeout)
            verbose = preset_config.get("verbose", verbose)
            fail_fast = preset_config.get("fail_fast", fail_fast)
            markers = preset_config.get("markers", markers)

        # Use runner's default timeout if not specified
        if timeout is None:
            timeout = self._runner.default_timeout

        # Build command using the runner backend
        cmd = self._runner.build_run_command(
            target=target,
            verbose=verbose,
            fail_fast=fail_fast,
            extra_args=extra_args,
            markers=markers,
        )

        command_str = " ".join(cmd)

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            # Parse output using the runner backend
            tests, passed, failed, skipped, errors = self._runner.parse_run_output(
                result.stdout, result.stderr, result.returncode
            )

            return TestRunResult(
                success=result.returncode == 0,
                duration=0.0,  # Would need timing wrapper
                total=len(tests) if tests else max(passed + failed + skipped + errors, 1),
                passed=passed,
                failed=failed,
                skipped=skipped,
                errors=errors,
                tests=tests,
                command=command_str,
                cwd=str(self.workspace),
                stdout=result.stdout,
                stderr=result.stderr,
                metadata={
                    "return_code": result.returncode,
                    "preset": preset,
                    "target": target,
                    "runner": type(self._runner).__name__,
                },
            )

        except subprocess.TimeoutExpired:
            return TestRunResult(
                success=False,
                command=command_str,
                cwd=str(self.workspace),
                error=f"Test run timed out after {timeout} seconds",
                metadata={"timeout": timeout},
            )

        except FileNotFoundError:
            return TestRunResult(
                success=False,
                command=command_str,
                cwd=str(self.workspace),
                error=self._runner.not_found_error,
            )

        except Exception as e:
            return TestRunResult(
                success=False,
                command=command_str,
                cwd=str(self.workspace),
                error=str(e),
            )

    def discover_tests(
        self,
        target: Optional[str] = None,
        pattern: str = "test_*.py",
    ) -> TestDiscoveryResult:
        """
        Discover tests without running them.

        Args:
            target: Directory or file to search
            pattern: File pattern for test files

        Returns:
            TestDiscoveryResult with discovered tests
        """
        cmd = self._runner.build_discover_command(target=target, pattern=pattern)

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=60,
            )

            tests, test_files = self._runner.parse_discover_output(result.stdout)

            return TestDiscoveryResult(
                success=result.returncode == 0,
                tests=tests,
                test_files=test_files,
                metadata={
                    "target": target,
                    "pattern": pattern,
                    "runner": type(self._runner).__name__,
                },
            )

        except subprocess.TimeoutExpired:
            return TestDiscoveryResult(
                success=False,
                error="Test discovery timed out",
            )

        except FileNotFoundError:
            return TestDiscoveryResult(
                success=False,
                error=self._runner.not_found_error,
            )

        except Exception as e:
            return TestDiscoveryResult(
                success=False,
                error=str(e),
            )


# Convenience functions


def run_tests(
    target: Optional[str] = None,
    preset: Optional[str] = None,
    workspace: Optional[Path] = None,
    runner_name: Optional[str] = None,
    test_config: Optional["TestConfig"] = None,
    **kwargs: Any,
) -> TestRunResult:
    """
    Run tests using the specified runner.

    Args:
        target: Test target
        preset: Preset configuration
        workspace: Repository root
        runner_name: Name of the runner to use (pytest, go, npm, etc.)
        test_config: TestConfig from foundry-mcp.toml
        **kwargs: Additional arguments for TestRunner.run_tests

    Returns:
        TestRunResult with test outcomes
    """
    runner_backend = get_runner(runner_name, test_config)
    runner = TestRunner(workspace, runner=runner_backend)
    return runner.run_tests(target, preset, **kwargs)


def discover_tests(
    target: Optional[str] = None,
    workspace: Optional[Path] = None,
    pattern: str = "test_*.py",
    runner_name: Optional[str] = None,
    test_config: Optional["TestConfig"] = None,
) -> TestDiscoveryResult:
    """
    Discover tests without running them.

    Args:
        target: Directory or file to search
        workspace: Repository root
        pattern: File pattern
        runner_name: Name of the runner to use (pytest, go, npm, etc.)
        test_config: TestConfig from foundry-mcp.toml

    Returns:
        TestDiscoveryResult with discovered tests
    """
    runner_backend = get_runner(runner_name, test_config)
    runner = TestRunner(workspace, runner=runner_backend)
    return runner.discover_tests(target, pattern)


def get_presets() -> Dict[str, Dict[str, Any]]:
    """
    Get available test presets.

    Returns:
        Dict of preset names to configurations
    """
    return TEST_PRESETS.copy()

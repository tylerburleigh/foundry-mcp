"""
Fixture management for parity tests.
"""

import json
import shutil
from pathlib import Path
from typing import Optional


class FixtureManager:
    """Manages test fixtures for parity tests."""

    FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "sample_specs"

    def __init__(self, test_root: Path):
        """
        Initialize fixture manager.

        Args:
            test_root: Root directory for test (typically tmp_path)
        """
        self.test_root = test_root
        self.specs_dir = test_root / "specs"

    def setup(self, spec_name: str, status: str = "active") -> Path:
        """
        Copy a fixture spec to the test directory.

        Args:
            spec_name: Name of fixture (without .json)
            status: Status folder (active, pending, completed, archived)

        Returns:
            Path to the copied spec file
        """
        # Create directory structure
        for folder in ["pending", "active", "completed", "archived"]:
            (self.specs_dir / folder).mkdir(parents=True, exist_ok=True)

        # Copy fixture
        source = self.FIXTURE_DIR / f"{spec_name}.json"
        if not source.exists():
            raise FileNotFoundError(f"Fixture not found: {source}")

        # Determine destination filename from spec_id in the fixture
        with open(source) as f:
            spec_data = json.load(f)

        spec_id = spec_data.get("spec_id", spec_name.replace("_", "-"))
        dest = self.specs_dir / status / f"{spec_id}.json"

        # Write spec to destination
        with open(dest, "w") as f:
            json.dump(spec_data, f, indent=2)

        return dest

    def setup_multiple(self, specs: dict[str, str]) -> dict[str, Path]:
        """
        Setup multiple fixtures at once.

        Args:
            specs: Dict mapping spec_name to status

        Returns:
            Dict mapping spec_name to destination path
        """
        return {
            name: self.setup(name, status)
            for name, status in specs.items()
        }

    def cleanup(self):
        """Remove test directory."""
        if self.test_root.exists():
            shutil.rmtree(self.test_root)

    def get_spec_path(self, spec_id: str, status: str = "active") -> Path:
        """Get path to a spec file."""
        return self.specs_dir / status / f"{spec_id}.json"

    def read_spec(self, spec_id: str, status: str = "active") -> dict:
        """Read and return spec data."""
        path = self.get_spec_path(spec_id, status)
        with open(path) as f:
            return json.load(f)

    def write_spec(self, spec_id: str, data: dict, status: str = "active"):
        """Write spec data to file."""
        path = self.get_spec_path(spec_id, status)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

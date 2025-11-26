"""
Output normalization for parity testing.

Normalizes outputs from foundry-mcp and sdd-toolkit for comparison,
handling timestamps, paths, and field name differences.
"""

import re
from typing import Any, Dict, Set


class OutputNormalizer:
    """Normalizes outputs for comparison between systems."""

    # Fields that contain timestamps and should be normalized
    TIMESTAMP_FIELDS: Set[str] = {
        "timestamp",
        "created_at",
        "updated_at",
        "completed_at",
        "started_at",
        "executed_at",
        "marked_at",
        "unblocked_at",
        "blocked_at",
    }

    # Fields that contain file paths
    PATH_FIELDS: Set[str] = {
        "path",
        "file_path",
        "spec_path",
        "specs_dir",
        "backup_path",
        "source_path",
        "dest_path",
    }

    # Fields unique to foundry-mcp that should be removed for comparison
    FOUNDRY_ONLY_FIELDS: Set[str] = {
        "author",  # foundry-mcp adds author field
    }

    # Fields unique to sdd-toolkit that should be removed for comparison
    SDD_ONLY_FIELDS: Set[str] = {
        "_exit_code",  # CLI-specific
        "_stdout",  # CLI-specific
        "_stderr",  # CLI-specific
        "dry_run",  # CLI option
    }

    @classmethod
    def normalize(cls, data: Any, source: str = "foundry") -> Any:
        """
        Normalize output data for comparison.

        Args:
            data: Data to normalize (dict, list, or primitive)
            source: Source system ("foundry" or "sdd")

        Returns:
            Normalized data
        """
        if isinstance(data, dict):
            return cls._normalize_dict(data, source)
        elif isinstance(data, list):
            return [cls.normalize(item, source) for item in data]
        else:
            return data

    @classmethod
    def _normalize_dict(cls, data: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Normalize a dictionary."""
        result = {}

        for key, value in data.items():
            # Skip source-specific fields
            if source == "foundry" and key in cls.SDD_ONLY_FIELDS:
                continue
            if source == "sdd" and key in cls.FOUNDRY_ONLY_FIELDS:
                continue

            # Normalize timestamps to placeholder
            if key in cls.TIMESTAMP_FIELDS and value:
                result[key] = "<TIMESTAMP>"
                continue

            # Normalize paths to relative form
            if key in cls.PATH_FIELDS and value:
                result[key] = cls._normalize_path(value)
                continue

            # Recursively normalize nested structures
            result[key] = cls.normalize(value, source)

        return result

    @classmethod
    def _normalize_path(cls, path: str) -> str:
        """
        Normalize a path to relative form.

        Removes absolute path prefix, keeping relative part from specs/.
        """
        if not isinstance(path, str):
            return path

        # Remove absolute path prefix, keep relative part from specs/
        if "specs/" in path:
            return path[path.index("specs/"):]
        return path

    @classmethod
    def _is_timestamp_string(cls, value: str) -> bool:
        """Check if a string looks like a timestamp."""
        # ISO 8601 format
        iso_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        return bool(re.match(iso_pattern, value))


class FieldMapper:
    """Maps field names between systems for comparison."""

    # Mappings from foundry-mcp field names to canonical names
    # These are applied to sdd output to match foundry naming
    FIELD_MAPPINGS: Dict[str, str] = {
        # Progress fields
        "percentage": "progress_percentage",
        "total": "total_tasks",
        "completed": "completed_tasks",
        "pending": "pending_tasks",
        "in_progress": "in_progress_tasks",
        # Task fields
        "task_count": "total_tasks",
        # Validation fields
        "valid": "is_valid",
    }

    # Inverse mappings (sdd -> foundry canonical)
    INVERSE_MAPPINGS: Dict[str, str] = {v: k for k, v in FIELD_MAPPINGS.items()}

    @classmethod
    def map_to_canonical(cls, data: Dict[str, Any], source: str) -> Dict[str, Any]:
        """
        Map field names to canonical form.

        Args:
            data: Data to map
            source: Source system ("foundry" or "sdd")

        Returns:
            Data with mapped field names
        """
        if source == "sdd":
            # Map sdd field names to foundry (canonical) names
            return cls._apply_mappings(data, cls.FIELD_MAPPINGS)
        return data

    @classmethod
    def _apply_mappings(
        cls, data: Dict[str, Any], mappings: Dict[str, str]
    ) -> Dict[str, Any]:
        """Apply field name mappings recursively."""
        if not isinstance(data, dict):
            return data

        result = {}
        for key, value in data.items():
            # Apply mapping if exists
            new_key = mappings.get(key, key)

            # Recursively map nested structures
            if isinstance(value, dict):
                result[new_key] = cls._apply_mappings(value, mappings)
            elif isinstance(value, list):
                result[new_key] = [
                    cls._apply_mappings(item, mappings)
                    if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                result[new_key] = value

        return result


def normalize_for_comparison(
    foundry_result: Dict[str, Any],
    sdd_result: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Normalize both results for comparison.

    Convenience function that applies normalization and field mapping.

    Args:
        foundry_result: Result from foundry-mcp
        sdd_result: Result from sdd-toolkit

    Returns:
        Tuple of (normalized_foundry, normalized_sdd)
    """
    foundry_norm = OutputNormalizer.normalize(foundry_result, "foundry")
    sdd_norm = OutputNormalizer.normalize(sdd_result, "sdd")
    sdd_norm = FieldMapper.map_to_canonical(sdd_norm, "sdd")

    return foundry_norm, sdd_norm

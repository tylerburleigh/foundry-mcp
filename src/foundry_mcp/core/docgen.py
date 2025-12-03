"""Codebase documentation generator for foundry-mcp.

This module provides documentation generation for codebases with optional
AI-enhanced narrative generation. It scans a repository, builds structural
metadata (modules, classes, functions), and emits markdown and JSON artifacts.

Documentation modes:
    - Basic Mode: Deterministic, LLM-free documentation for reproducibility
    - AI-Enhanced Mode: Uses ConsultationOrchestrator for rich narrative docs

AI-enhanced mode uses:
    - DOC_GEN_PROJECT_OVERVIEW_V1: Project overview narrative generation
    - DOC_GEN_ARCHITECTURE_V1: Architecture documentation generation
    - DOC_GEN_COMPONENT_INVENTORY_V1: Component inventory generation

When AI mode is requested (use_ai=True) but no providers are available,
raises AIProviderUnavailableError rather than falling back silently.

Example:
    from foundry_mcp.core.docgen import DocumentationGenerator

    generator = DocumentationGenerator(project_root, output_dir)

    # Basic mode (deterministic)
    result = generator.generate(project_name="MyProject")

    # AI-enhanced mode
    result = generator.generate(
        project_name="MyProject",
        use_ai=True,
        ai_provider="gemini",
    )
"""

from __future__ import annotations

import ast
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Directories we never scan
DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    "venv",
    ".venv",
    "build",
    "dist",
    ".idea",
    ".vscode",
}

# File extensions mapped to language names
LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".md": "markdown",
    ".rst": "rst",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
}

# Artifacts written by the generator
DOC_ARTIFACTS = [
    "codebase.json",
    "project-overview.md",
    "architecture.md",
    "component-inventory.md",
    "index.md",
    "doc-generation-state.json",
]


class AIProviderUnavailableError(Exception):
    """Raised when AI mode is requested but no providers are available.

    This exception is raised when use_ai=True is passed to generate()
    but no AI providers could be initialized. The error includes the
    AI_NO_PROVIDER error code for structured error handling.

    Attributes:
        provider_id: The requested provider ID, if any.
        message: Human-readable error message.
    """

    def __init__(
        self,
        message: str = "AI-enhanced mode requested but no providers available",
        provider_id: Optional[str] = None,
    ) -> None:
        self.message = message
        self.provider_id = provider_id
        super().__init__(message)


@dataclass
class ClassRecord:
    name: str
    file: str
    line: int
    language: str = "python"
    docstring: Optional[str] = None
    bases: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    is_exported: bool = False
    is_public: bool = True
    instantiation_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "file": self.file,
            "line": self.line,
            "language": self.language,
            "docstring": self.docstring,
            "bases": self.bases,
            "methods": self.methods,
            "properties": self.properties,
            "is_exported": self.is_exported,
            "is_public": self.is_public,
            "instantiation_count": self.instantiation_count,
        }


@dataclass
class FunctionRecord:
    name: str
    file: str
    line: int
    language: str = "python"
    docstring: Optional[str] = None
    arguments: List[str] = field(default_factory=list)
    returns: Optional[str] = None
    is_async: bool = False
    calls: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "file": self.file,
            "line": self.line,
            "language": self.language,
            "docstring": self.docstring,
            "arguments": self.arguments,
            "returns": self.returns,
            "is_async": self.is_async,
            "calls": self.calls,
            "decorators": self.decorators,
        }


@dataclass
class ModuleRecord:
    name: str
    file: str
    language: str
    docstring: Optional[str]
    classes: List[str]
    functions: List[str]
    imports: List[str]
    lines: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "file": self.file,
            "language": self.language,
            "docstring": self.docstring,
            "classes": self.classes,
            "functions": self.functions,
            "imports": self.imports,
            "lines": self.lines,
            "exports": [],
        }


@dataclass
class CodebaseStats:
    total_files: int = 0
    total_lines: int = 0
    language_breakdown: Dict[str, Dict[str, int]] = field(default_factory=dict)
    python_files: int = 0
    python_lines: int = 0
    class_count: int = 0
    function_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files": self.total_files,
            "total_lines": self.total_lines,
            "languages": list(self.language_breakdown.keys()),
            "by_language": self.language_breakdown,
            "python": {
                "files": self.python_files,
                "lines": self.python_lines,
                "classes": self.class_count,
                "functions": self.function_count,
            },
        }


@dataclass
class GeneratedArtifact:
    name: str
    path: Path
    artifact_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "type": self.artifact_type,
        }


@dataclass
class DocGenerationResult:
    project_name: str
    project_root: Path
    output_dir: Path
    stats: CodebaseStats
    modules: List[ModuleRecord]
    classes: List[ClassRecord]
    functions: List[FunctionRecord]
    artifacts: List[GeneratedArtifact]
    warnings: List[str]


class DocumentationGenerator:
    """
    Scans a repository and produces documentation artifacts.

    Supports two modes:
        - Basic Mode: Deterministic, LLM-free documentation (default)
        - AI-Enhanced Mode: Uses ConsultationOrchestrator for rich narratives

    Attributes:
        project_root: Root directory of the project to document
        output_dir: Directory where documentation artifacts will be written
        excludes: Set of directory names to exclude from scanning
    """

    def __init__(
        self,
        project_root: Path,
        output_dir: Path,
        *,
        excludes: Optional[Sequence[str]] = None,
    ) -> None:
        self.project_root = project_root.resolve()
        if not self.project_root.exists():
            raise FileNotFoundError(f"Project root not found: {project_root}")

        if output_dir.is_absolute():
            self.output_dir = output_dir
        else:
            self.output_dir = (self.project_root / output_dir).resolve()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.excludes = set(DEFAULT_EXCLUDE_DIRS)
        if excludes:
            self.excludes.update(excludes)

        # Ensure generated docs directory is always excluded
        try:
            rel_output = self.output_dir.relative_to(self.project_root)
            self.excludes.add(str(rel_output))
            self._output_rel_parts = rel_output.parts
        except ValueError:
            self._output_rel_parts = None

        # AI consultation orchestrator (lazy-initialized)
        self._orchestrator: Optional["ConsultationOrchestrator"] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        *,
        project_name: str,
        description: Optional[str] = None,
        use_cache: bool = True,
        resume: bool = False,
        use_ai: bool = False,
        ai_provider: Optional[str] = None,
        ai_timeout: float = 120.0,
        consultation_cache: bool = True,
    ) -> DocGenerationResult:
        """
        Generate documentation artifacts for the repository.

        Args:
            project_name: Name of the project
            description: Optional project description
            use_cache: Use cached results where possible
            resume: Resume from previous incomplete generation
            use_ai: Enable AI-enhanced documentation generation
            ai_provider: Preferred AI provider ID (auto-selects if None)
            ai_timeout: Timeout in seconds for AI consultations
            consultation_cache: Use AI consultation cache (default: True)

        Returns:
            DocGenerationResult with generated artifacts and metadata

        Raises:
            AIProviderUnavailableError: If use_ai=True but no providers available
        """
        # Store consultation cache setting for AI builders
        self._consultation_cache = consultation_cache
        stats = CodebaseStats()
        modules: List[ModuleRecord] = []
        classes: List[ClassRecord] = []
        functions: List[FunctionRecord] = []
        warnings: List[str] = []

        dir_stats: Dict[str, int] = {}
        tree_entries: List[Tuple[int, Path]] = []

        for file_path in self._iter_files():
            rel_path = file_path.relative_to(self.project_root)
            language = LANGUAGE_EXTENSIONS.get(file_path.suffix.lower(), "other")
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = file_path.read_text(encoding="utf-8", errors="ignore")

            line_count = text.count("\n") + 1
            stats.total_files += 1
            stats.total_lines += line_count
            lang_stats = stats.language_breakdown.setdefault(
                language,
                {"files": 0, "lines": 0},
            )
            lang_stats["files"] += 1
            lang_stats["lines"] += line_count

            top_level = rel_path.parts[0] if rel_path.parts else ""
            if top_level:
                dir_stats[top_level] = dir_stats.get(top_level, 0) + 1

            if len(tree_entries) < 2000:
                tree_entries.append((len(rel_path.parts), rel_path))

            if language != "python" or not file_path.suffix.lower() == ".py":
                continue

            stats.python_files += 1
            stats.python_lines += line_count

            module_data = self._parse_python_module(rel_path, text)
            modules.append(module_data["module"])
            classes.extend(module_data["classes"])
            functions.extend(module_data["functions"])

        stats.class_count = len(classes)
        stats.function_count = len(functions)

        # Initialize AI orchestrator if AI mode enabled
        ai_available = False
        if use_ai:
            ai_available = self._init_ai_orchestrator(ai_provider, ai_timeout)
            if not ai_available:
                # Raise error instead of falling back per spec requirement
                provider_msg = f" (requested: {ai_provider})" if ai_provider else ""
                raise AIProviderUnavailableError(
                    f"AI-enhanced mode requested but no providers available{provider_msg}. "
                    "Set use_ai=False for basic mode or configure an AI provider.",
                    provider_id=ai_provider,
                )

        artifacts = self._write_artifacts(
            project_name=project_name,
            description=description,
            stats=stats,
            modules=modules,
            classes=classes,
            functions=functions,
            dir_stats=dir_stats,
            tree_entries=tree_entries,
            use_cache=use_cache,
            resume=resume,
            use_ai=use_ai and ai_available,
        )

        return DocGenerationResult(
            project_name=project_name,
            project_root=self.project_root,
            output_dir=self.output_dir,
            stats=stats,
            modules=modules,
            classes=classes,
            functions=functions,
            artifacts=artifacts,
            warnings=warnings,
        )

    def _init_ai_orchestrator(
        self,
        provider_id: Optional[str],
        timeout: float,
    ) -> bool:
        """
        Initialize the AI consultation orchestrator.

        Args:
            provider_id: Preferred provider ID
            timeout: Default timeout for consultations

        Returns:
            True if AI is available, False otherwise
        """
        try:
            from foundry_mcp.core.ai_consultation import ConsultationOrchestrator

            self._orchestrator = ConsultationOrchestrator(
                preferred_providers=[provider_id] if provider_id else [],
                default_timeout=timeout,
            )
            return self._orchestrator.is_available(provider_id)
        except ImportError:
            logger.warning("AI consultation module not available")
            return False
        except Exception as exc:
            logger.warning("Failed to initialize AI orchestrator: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iter_files(self) -> Iterable[Path]:
        """Yield files under project_root while respecting exclusions."""

        def should_skip_dir(rel_parts: Tuple[str, ...]) -> bool:
            if not rel_parts:
                return False
            if rel_parts[0] in self.excludes:
                return True
            if (
                self._output_rel_parts
                and rel_parts[: len(self._output_rel_parts)] == self._output_rel_parts
            ):
                return True
            rel_str = str(Path(*rel_parts))
            return rel_str in self.excludes

        for dirpath, dirnames, filenames in os.walk(self.project_root):
            rel_dir = Path(dirpath).relative_to(self.project_root)
            if should_skip_dir(rel_dir.parts):
                dirnames[:] = []
                continue

            dirnames[:] = [
                d for d in dirnames if not should_skip_dir((*(rel_dir.parts), d))
            ]

            for filename in filenames:
                full_path = Path(dirpath) / filename
                rel_path = full_path.relative_to(self.project_root)
                if should_skip_dir(rel_path.parts[:-1]):
                    continue
                if (
                    self._output_rel_parts
                    and rel_path.parts[: len(self._output_rel_parts)]
                    == self._output_rel_parts
                ):
                    continue
                yield full_path

    def _parse_python_module(
        self,
        relative_path: Path,
        source: str,
    ) -> Dict[str, Any]:
        module_name = ".".join(relative_path.with_suffix("").parts)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = ast.Module(body=[], type_ignores=[])

        module_doc = ast.get_docstring(tree)
        imports = self._collect_imports(tree)
        module_classes: List[ClassRecord] = []
        module_functions: List[FunctionRecord] = []

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                module_classes.append(self._class_from_node(node, relative_path))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                module_functions.append(self._function_from_node(node, relative_path))

        module_record = ModuleRecord(
            name=module_name or relative_path.stem,
            file=str(relative_path),
            language="python",
            docstring=module_doc,
            classes=[cls.name for cls in module_classes],
            functions=[func.name for func in module_functions],
            imports=imports,
            lines=source.count("\n") + 1,
        )

        return {
            "module": module_record,
            "classes": module_classes,
            "functions": module_functions,
        }

    def _class_from_node(self, node: ast.ClassDef, relative_path: Path) -> ClassRecord:
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(self._format_attribute(base))

        methods = [
            child.name
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        properties = [
            target.id
            for child in node.body
            for target in getattr(child, "targets", [])
            if isinstance(target, ast.Name)
        ]

        return ClassRecord(
            name=node.name,
            file=str(relative_path),
            line=getattr(node, "lineno", 1),
            docstring=ast.get_docstring(node),
            bases=bases,
            methods=methods,
            properties=properties,
            is_exported=not node.name.startswith("_"),
            is_public=not node.name.startswith("_"),
        )

    def _function_from_node(
        self,
        node: ast.AST,
        relative_path: Path,
    ) -> FunctionRecord:
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        arg_names: List[str] = [
            arg.arg for arg in getattr(node.args, "posonlyargs", [])
        ]
        arg_names += [arg.arg for arg in node.args.args]

        defaults = list(node.args.defaults or [])
        if defaults:
            defaults_start = len(arg_names) - len(defaults)
            for idx in range(defaults_start, len(arg_names)):
                arg_names[idx] = f"{arg_names[idx]}=?"

        if node.args.kwonlyargs:
            kw_defaults = list(node.args.kw_defaults or [])
            while len(kw_defaults) < len(node.args.kwonlyargs):
                kw_defaults.append(None)
            for kw_arg, default in zip(node.args.kwonlyargs, kw_defaults):
                name = kw_arg.arg
                if default is not None:
                    name = f"{name}=?"
                arg_names.append(name)

        vararg = getattr(node.args, "vararg", None)
        if isinstance(vararg, ast.arg):
            arg_names.append(f"*{vararg.arg}")

        kwarg = getattr(node.args, "kwarg", None)
        if isinstance(kwarg, ast.arg):
            arg_names.append(f"**{kwarg.arg}")

        returns = None
        if node.returns:
            returns = self._format_annotation(node.returns)

        decorators = [self._format_attribute(d) for d in node.decorator_list]
        calls = self._collect_calls(node)

        return FunctionRecord(
            name=node.name,
            file=str(relative_path),
            line=getattr(node, "lineno", 1),
            docstring=ast.get_docstring(node),
            arguments=arg_names,
            returns=returns,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            calls=calls,
            decorators=decorators,
        )

    def _format_attribute(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{self._format_attribute(node.value)}.{node.attr}"
        if isinstance(node, ast.Subscript):
            return self._format_attribute(node.value)
        if isinstance(node, ast.Call):
            return self._format_attribute(node.func)
        return "unknown"

    def _format_annotation(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return self._format_attribute(node)
        if isinstance(node, ast.Subscript):
            return self._format_attribute(node.value)
        return "expression"

    def _collect_calls(self, func_node: ast.AST) -> List[str]:
        calls: List[str] = []
        formatter = self

        class CallVisitor(ast.NodeVisitor):
            def visit_Call(self, call_node: ast.Call) -> None:  # type: ignore
                target = call_node.func
                if isinstance(target, ast.Name):
                    calls.append(target.id)
                elif isinstance(target, ast.Attribute):
                    calls.append(formatter._format_attribute(target))
                self.generic_visit(call_node)

        CallVisitor().visit(func_node)
        return calls

    def _collect_imports(self, tree: ast.AST) -> List[str]:
        imports: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    full_name = f"{module}.{alias.name}" if module else alias.name
                    imports.append(full_name)
        return imports

    # ------------------------------------------------------------------
    # Artifact writers
    # ------------------------------------------------------------------

    def _write_artifacts(
        self,
        *,
        project_name: str,
        description: Optional[str],
        stats: CodebaseStats,
        modules: List[ModuleRecord],
        classes: List[ClassRecord],
        functions: List[FunctionRecord],
        dir_stats: Dict[str, int],
        tree_entries: List[Tuple[int, Path]],
        use_cache: bool,
        resume: bool,
        use_ai: bool = False,
    ) -> List[GeneratedArtifact]:
        artifacts: List[GeneratedArtifact] = []
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # Write codebase.json first (used as context for AI prompts)
        codebase_payload = {
            "metadata": {
                "project_name": project_name,
                "generated_at": timestamp,
                "schema_version": "1.0",
                "description": description,
            },
            "statistics": stats.to_dict(),
            "modules": [m.to_dict() for m in modules],
            "classes": [c.to_dict() for c in classes],
            "functions": [f.to_dict() for f in functions],
        }
        artifacts.append(self._write_json("codebase.json", codebase_payload, "json"))

        # Build project context for AI prompts (summarized to avoid context overflow)
        project_context = self._build_project_context(
            project_name, description, stats, dir_stats, modules
        )
        key_files = self._select_key_files(modules, classes, functions)

        # Generate project overview (write-as-you-go pattern)
        if use_ai:
            overview = self._build_project_overview_ai(
                project_name, description, stats, dir_stats, project_context, key_files
            )
        else:
            overview = self._build_project_overview(
                project_name, description, stats, dir_stats
            )
        artifacts.append(self._write_text("project-overview.md", overview, "markdown"))

        # Generate architecture doc (write-as-you-go pattern)
        if use_ai:
            architecture = self._build_architecture_doc_ai(
                project_name, stats, dir_stats, project_context, key_files
            )
        else:
            architecture = self._build_architecture_doc(project_name, stats, dir_stats)
        artifacts.append(self._write_text("architecture.md", architecture, "markdown"))

        # Generate component inventory (write-as-you-go pattern)
        if use_ai:
            inventory = self._build_component_inventory_ai(
                project_name, tree_entries, project_context, dir_stats
            )
        else:
            inventory = self._build_component_inventory(project_name, tree_entries)
        artifacts.append(
            self._write_text("component-inventory.md", inventory, "markdown")
        )

        index_md = self._build_index_doc(artifacts)
        artifacts.append(self._write_text("index.md", index_md, "markdown"))

        state_payload = {
            "project_root": str(self.project_root),
            "output_folder": str(self.output_dir),
            "started_at": timestamp,
            "last_updated": timestamp,
            "completed_shards": [
                "codebase",
                "project_overview",
                "architecture",
                "component_inventory",
                "index",
            ],
            "failed_shards": [],
            "pending_shards": [],
            "options": {
                "use_cache": use_cache,
                "resume": resume,
                "use_ai": use_ai,
            },
        }
        artifacts.append(
            self._write_json("doc-generation-state.json", state_payload, "state")
        )

        return artifacts

    def _build_project_context(
        self,
        project_name: str,
        description: Optional[str],
        stats: CodebaseStats,
        dir_stats: Dict[str, int],
        modules: List[ModuleRecord],
    ) -> str:
        """
        Build summarized project context for AI prompts.

        Creates a compact representation of project metadata to inject
        into prompt templates without exceeding context limits.
        """
        primary_languages = sorted(stats.language_breakdown.keys()) or ["unknown"]
        top_dirs = sorted(dir_stats.items(), key=lambda item: item[1], reverse=True)[:5]

        lines = [
            f"- **Project Name:** {project_name}",
            f"- **Description:** {description or 'Auto-generated documentation'}",
            f"- **Primary Languages:** {', '.join(primary_languages)}",
            f"- **Total Files:** {stats.total_files}",
            f"- **Total Lines:** {stats.total_lines:,}",
            f"- **Python Files:** {stats.python_files}",
            f"- **Python Classes:** {stats.class_count}",
            f"- **Python Functions:** {stats.function_count}",
            "",
            "### Top Directories by File Count",
        ]

        for directory, count in top_dirs:
            lines.append(f"- `{directory}/`: {count} files")

        if modules:
            lines.append("")
            lines.append("### Module Overview (top 10)")
            sorted_modules = sorted(
                modules, key=lambda m: m.lines, reverse=True
            )[:10]
            for mod in sorted_modules:
                lines.append(
                    f"- `{mod.name}`: {mod.lines} lines, "
                    f"{len(mod.classes)} classes, {len(mod.functions)} functions"
                )

        return "\n".join(lines)

    def _select_key_files(
        self,
        modules: List[ModuleRecord],
        classes: List[ClassRecord],
        functions: List[FunctionRecord],
        max_files: int = 15,
    ) -> str:
        """
        Select the most important files for AI analysis.

        Prioritizes:
        1. Entry points (main.py, __main__.py, app.py, etc.)
        2. Files with most classes
        3. Files with most functions
        """
        file_scores: Dict[str, int] = {}

        # Entry point patterns get high scores
        entry_patterns = ["main.py", "__main__.py", "app.py", "cli.py", "__init__.py"]

        for mod in modules:
            score = mod.lines  # Base score on size
            if any(mod.file.endswith(p) for p in entry_patterns):
                score += 1000  # Boost entry points
            score += len(mod.classes) * 50  # Boost files with classes
            score += len(mod.functions) * 10  # Boost files with functions
            file_scores[mod.file] = score

        # Sort by score and take top files
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [f"- `{f}`" for f, _ in sorted_files[:max_files]]

        return "\n".join(selected) if selected else "- No Python files found"

    def _write_json(
        self, name: str, payload: Dict[str, Any], artifact_type: str
    ) -> GeneratedArtifact:
        target = self.output_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return GeneratedArtifact(name=name, path=target, artifact_type=artifact_type)

    def _write_text(
        self, name: str, content: str, artifact_type: str
    ) -> GeneratedArtifact:
        target = self.output_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return GeneratedArtifact(name=name, path=target, artifact_type=artifact_type)

    def _build_project_overview(
        self,
        project_name: str,
        description: Optional[str],
        stats: CodebaseStats,
        dir_stats: Dict[str, int],
    ) -> str:
        primary_languages = sorted(stats.language_breakdown.keys()) or ["unknown"]
        top_dirs = sorted(dir_stats.items(), key=lambda item: item[1], reverse=True)[
            :10
        ]
        description_line = (
            description
            or "Auto-generated project overview focusing on structural insights."
        )

        lines = [
            f"# {project_name} - Project Overview\n",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d')}\n",
            f"**Summary:** {description_line}\n",
            "## Repository Snapshot",
            f"- Total files: {stats.total_files}",
            f"- Total lines: {stats.total_lines}",
            f"- Primary languages: {', '.join(primary_languages)}",
            f"- Python files: {stats.python_files}",
            f"- Python LOC: {stats.python_lines}",
            "",
            "## Directory Highlights",
        ]
        if top_dirs:
            for directory, count in top_dirs:
                lines.append(f"- `{directory}`: {count} files")
        else:
            lines.append("No directories detected")

        lines.append("\n## Language Breakdown")
        for lang, lang_stats in sorted(stats.language_breakdown.items()):
            lines.append(
                f"- **{lang}**: {lang_stats['files']} files • {lang_stats['lines']} lines"
            )

        lines.append("\n## Notes")
        lines.append(
            "- Documentation generated without LLM dependencies for deterministic reproducibility."
        )
        lines.append(
            "- Re-run `sdd llm-doc generate` after significant codebase changes to refresh artifacts."
        )

        return "\n".join(lines) + "\n"

    def _build_architecture_doc(
        self,
        project_name: str,
        stats: CodebaseStats,
        dir_stats: Dict[str, int],
    ) -> str:
        primary_languages = sorted(stats.language_breakdown.keys()) or ["unknown"]
        total_modules = len(dir_stats)
        top_dirs = sorted(dir_stats.items(), key=lambda item: item[1], reverse=True)[:5]

        lines = [
            f"# {project_name} - Architecture Overview\n",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d')}\n",
            "## Technology Stack",
            f"- Languages: {', '.join(primary_languages)}",
            f"- Python functions indexed: {stats.function_count}",
            f"- Python classes indexed: {stats.class_count}",
            "",
            "## Structural Layers",
            "1. **CLI Layer** – commands under `src/foundry_mcp/cli` emit JSON envelopes.",
            "2. **Core Services** – business logic under `src/foundry_mcp/core`.",
            "3. **Tooling Layer** – canonical MCP tools in `src/foundry_mcp/tools`.",
            "4. **Specs & Docs** – spec data in `specs/` and generated docs in `docs/generated`.\n",
            "## Dominant Directories",
        ]
        if top_dirs:
            for directory, count in top_dirs:
                lines.append(f"- `{directory}` contains {count} tracked files")
        else:
            lines.append("No directory breakdown available")

        lines.extend(
            [
                "\n## Change Management",
                "- Keep specs, docs, and code synchronized to preserve tooling accuracy.",
                "- Regenerate documentation whenever modules change to keep doc-query results fresh.",
                "\n## Metrics",
                f"- Total tracked modules: {total_modules}",
                f"- Total files scanned: {stats.total_files}",
            ]
        )

        return "\n".join(lines) + "\n"

    def _build_component_inventory(
        self,
        project_name: str,
        tree_entries: List[Tuple[int, Path]],
    ) -> str:
        lines = [
            f"# {project_name} - Component Inventory\n",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d')}\n",
            "## Directory Tree (truncated)\n",
            "```",
        ]
        sorted_entries = sorted(tree_entries, key=lambda item: (item[0], item[1]))[:400]
        for _, rel_path in sorted_entries:
            indent = "  " * (len(rel_path.parts) - 1)
            prefix = "└── " if len(rel_path.parts) > 1 else ""
            lines.append(f"{indent}{prefix}{rel_path.name}")
        lines.append("```\n")
        lines.append(
            "_Tree limited to first 400 entries for readability. Re-run generation after major structural changes._\n"
        )
        return "\n".join(lines)

    def _build_index_doc(self, artifacts: List[GeneratedArtifact]) -> str:
        lines = [
            "# Documentation Index\n",
            "The following artifacts were generated for the current repository:\n",
        ]
        for artifact in artifacts:
            if artifact.name == "doc-generation-state.json":
                continue
            lines.append(f"- `{artifact.name}` — {artifact.artifact_type}")
        lines.append(
            "\nArtifacts live under `docs/generated`. Run `sdd llm-doc generate` to refresh them.\n"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # AI-Enhanced Builders
    # ------------------------------------------------------------------

    def _build_project_overview_ai(
        self,
        project_name: str,
        description: Optional[str],
        stats: CodebaseStats,
        dir_stats: Dict[str, int],
        project_context: str,
        key_files: str,
    ) -> str:
        """
        Build AI-enhanced project overview using DOC_GEN_PROJECT_OVERVIEW_V1.

        Uses the ConsultationOrchestrator to generate narrative documentation
        with richer insights than the deterministic basic mode.

        Falls back to basic mode if AI consultation fails.
        """
        if not self._orchestrator:
            logger.warning("AI orchestrator not initialized, falling back to basic mode")
            return self._build_project_overview(project_name, description, stats, dir_stats)

        try:
            from foundry_mcp.core.ai_consultation import (
                ConsultationRequest,
                ConsultationWorkflow,
            )

            request = ConsultationRequest(
                workflow=ConsultationWorkflow.DOC_GENERATION,
                prompt_id="DOC_GEN_PROJECT_OVERVIEW_V1",
                context={
                    "project_context": project_context,
                    "key_files": key_files,
                },
            )

            result = self._orchestrator.consult(
                request, use_cache=self._consultation_cache
            )

            if result.success and result.content:
                # Compose the final document with AI findings
                return self._compose_overview_doc(
                    project_name,
                    description,
                    stats,
                    dir_stats,
                    result.content,
                    result.provider_id,
                )
            else:
                logger.warning(
                    "AI consultation for project overview failed: %s. Falling back to basic mode.",
                    result.error,
                )
                return self._build_project_overview(project_name, description, stats, dir_stats)

        except Exception as exc:
            logger.warning(
                "AI-enhanced overview generation failed: %s. Falling back to basic mode.",
                exc,
            )
            return self._build_project_overview(project_name, description, stats, dir_stats)

    def _compose_overview_doc(
        self,
        project_name: str,
        description: Optional[str],
        stats: CodebaseStats,
        dir_stats: Dict[str, int],
        ai_findings: str,
        provider_id: str,
    ) -> str:
        """Compose final project overview with AI findings."""
        primary_languages = sorted(stats.language_breakdown.keys()) or ["unknown"]

        lines = [
            f"# {project_name} - Project Overview\n",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d')}",
            f"**Mode:** AI-Enhanced (provider: {provider_id})\n",
            "## Project Classification\n",
            f"- **Primary Language(s):** {', '.join(primary_languages)}",
            f"- **Total Files:** {stats.total_files}",
            f"- **Total Lines:** {stats.total_lines:,}",
            f"- **Python Classes:** {stats.class_count}",
            f"- **Python Functions:** {stats.function_count}",
            "",
            "---",
            "",
            ai_findings,
            "",
            "---",
            "",
            "## Related Documentation",
            "",
            "- `index.md` - Master documentation index",
            "- `architecture.md` - Detailed architecture",
            "- `component-inventory.md` - Component inventory",
            "",
            "---",
            "",
            "*Generated using AI-enhanced documentation workflow*",
        ]
        return "\n".join(lines)

    def _build_architecture_doc_ai(
        self,
        project_name: str,
        stats: CodebaseStats,
        dir_stats: Dict[str, int],
        project_context: str,
        key_files: str,
    ) -> str:
        """
        Build AI-enhanced architecture documentation using DOC_GEN_ARCHITECTURE_V1.

        Uses the ConsultationOrchestrator to generate rich architecture analysis
        with pattern identification and design decision insights.

        Falls back to basic mode if AI consultation fails.
        """
        if not self._orchestrator:
            logger.warning("AI orchestrator not initialized, falling back to basic mode")
            return self._build_architecture_doc(project_name, stats, dir_stats)

        try:
            from foundry_mcp.core.ai_consultation import (
                ConsultationRequest,
                ConsultationWorkflow,
            )

            request = ConsultationRequest(
                workflow=ConsultationWorkflow.DOC_GENERATION,
                prompt_id="DOC_GEN_ARCHITECTURE_V1",
                context={
                    "project_context": project_context,
                    "key_files": key_files,
                },
            )

            result = self._orchestrator.consult(
                request, use_cache=self._consultation_cache
            )

            if result.success and result.content:
                return self._compose_architecture_doc(
                    project_name,
                    stats,
                    dir_stats,
                    result.content,
                    result.provider_id,
                )
            else:
                logger.warning(
                    "AI consultation for architecture doc failed: %s. Falling back to basic mode.",
                    result.error,
                )
                return self._build_architecture_doc(project_name, stats, dir_stats)

        except Exception as exc:
            logger.warning(
                "AI-enhanced architecture generation failed: %s. Falling back to basic mode.",
                exc,
            )
            return self._build_architecture_doc(project_name, stats, dir_stats)

    def _compose_architecture_doc(
        self,
        project_name: str,
        stats: CodebaseStats,
        dir_stats: Dict[str, int],
        ai_findings: str,
        provider_id: str,
    ) -> str:
        """Compose final architecture document with AI findings."""
        primary_languages = sorted(stats.language_breakdown.keys()) or ["unknown"]

        lines = [
            f"# {project_name} - Architecture Documentation\n",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d')}",
            f"**Mode:** AI-Enhanced (provider: {provider_id})\n",
            "## Technology Stack Details\n",
            "### Core Technologies\n",
            f"- **Languages:** {', '.join(primary_languages)}",
            f"- **Python Classes:** {stats.class_count}",
            f"- **Python Functions:** {stats.function_count}",
            "",
            "---",
            "",
            ai_findings,
            "",
            "---",
            "",
            "## Related Documentation",
            "",
            "- `index.md` - Master documentation index",
            "- `project-overview.md` - Project overview",
            "- `component-inventory.md` - Component inventory",
            "",
            "---",
            "",
            "*Generated using AI-enhanced documentation workflow*",
        ]
        return "\n".join(lines)

    def _build_component_inventory_ai(
        self,
        project_name: str,
        tree_entries: List[Tuple[int, Path]],
        project_context: str,
        dir_stats: Dict[str, int],
    ) -> str:
        """
        Build AI-enhanced component inventory using DOC_GEN_COMPONENT_INVENTORY_V1.

        Uses the ConsultationOrchestrator to analyze directory structure and
        provide insights about component organization and purposes.

        Falls back to basic mode if AI consultation fails.
        """
        if not self._orchestrator:
            logger.warning("AI orchestrator not initialized, falling back to basic mode")
            return self._build_component_inventory(project_name, tree_entries)

        try:
            from foundry_mcp.core.ai_consultation import (
                ConsultationRequest,
                ConsultationWorkflow,
            )

            # Build directory structure string
            sorted_entries = sorted(tree_entries, key=lambda item: (item[0], item[1]))[:200]
            tree_lines = []
            for _, rel_path in sorted_entries:
                indent = "  " * (len(rel_path.parts) - 1)
                prefix = "└── " if len(rel_path.parts) > 1 else ""
                tree_lines.append(f"{indent}{prefix}{rel_path.name}")
            directory_structure = "\n".join(tree_lines)

            # Build directories to analyze
            top_dirs = sorted(dir_stats.items(), key=lambda x: x[1], reverse=True)[:15]
            directories_to_analyze = "\n".join(f"- `{d}/`" for d, _ in top_dirs)

            request = ConsultationRequest(
                workflow=ConsultationWorkflow.DOC_GENERATION,
                prompt_id="DOC_GEN_COMPONENT_INVENTORY_V1",
                context={
                    "project_context": project_context,
                    "directory_structure": directory_structure,
                    "directories_to_analyze": directories_to_analyze,
                },
            )

            result = self._orchestrator.consult(
                request, use_cache=self._consultation_cache
            )

            if result.success and result.content:
                return self._compose_component_inventory_doc(
                    project_name,
                    tree_entries,
                    result.content,
                    result.provider_id,
                )
            else:
                logger.warning(
                    "AI consultation for component inventory failed: %s. Falling back to basic mode.",
                    result.error,
                )
                return self._build_component_inventory(project_name, tree_entries)

        except Exception as exc:
            logger.warning(
                "AI-enhanced component inventory generation failed: %s. Falling back to basic mode.",
                exc,
            )
            return self._build_component_inventory(project_name, tree_entries)

    def _compose_component_inventory_doc(
        self,
        project_name: str,
        tree_entries: List[Tuple[int, Path]],
        ai_findings: str,
        provider_id: str,
    ) -> str:
        """Compose final component inventory with AI findings."""
        lines = [
            f"# {project_name} - Component Inventory\n",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d')}",
            f"**Mode:** AI-Enhanced (provider: {provider_id})\n",
            "## Complete Directory Structure\n",
            "```",
        ]

        sorted_entries = sorted(tree_entries, key=lambda item: (item[0], item[1]))[:400]
        for _, rel_path in sorted_entries:
            indent = "  " * (len(rel_path.parts) - 1)
            prefix = "└── " if len(rel_path.parts) > 1 else ""
            lines.append(f"{indent}{prefix}{rel_path.name}")
        lines.append("```\n")

        lines.extend([
            "---",
            "",
            ai_findings,
            "",
            "---",
            "",
            "## Related Documentation",
            "",
            "- `index.md` - Master documentation index",
            "- `project-overview.md` - Project overview",
            "- `architecture.md` - Architecture documentation",
            "",
            "---",
            "",
            "*Generated using AI-enhanced documentation workflow*",
        ])
        return "\n".join(lines)


def resolve_output_directory(project_root: Path) -> Path:
    """Return the default output directory for generated docs."""
    return (project_root / "docs" / "generated").resolve()

"""
AST parsing module.
Extracts functions, classes, methods, and imports from Python source files
using the built-in `ast` module.
"""

from __future__ import annotations

import ast
import logging
import textwrap
from pathlib import Path

from core.models import CodeElement, FileAnalysis

logger = logging.getLogger(__name__)


class CodeParser:
    """Parses Python source files into structured CodeElement objects via AST."""

    def parse_file(self, file_info: dict) -> FileAnalysis:
        """
        Parse a single Python file and extract all code elements.

        Args:
            file_info: Dict with 'absolute_path' and 'relative_path'.

        Returns:
            FileAnalysis with extracted elements, or with parse_error set.
        """
        abs_path: Path = file_info["absolute_path"]
        rel_path: str = file_info["relative_path"]

        try:
            source = abs_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return FileAnalysis(
                file_path=rel_path,
                raw_source="",
                loc=0,
                parse_error=f"Could not read file: {e}",
            )

        source_lines = source.splitlines()
        loc = len([l for l in source_lines if l.strip() and not l.strip().startswith("#")])

        try:
            tree = ast.parse(source, filename=rel_path)
        except SyntaxError as e:
            return FileAnalysis(
                file_path=rel_path,
                raw_source=source,
                loc=loc,
                parse_error=f"SyntaxError at line {e.lineno}: {e.msg}",
            )

        elements: list[CodeElement] = []
        imports: list[str] = []

        for node in ast.walk(tree):
            # ── Imports ────────────────────────────────────────────────
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = ", ".join(a.name for a in node.names)
                imports.append(f"from {module} import {names}")

        # Process top-level and nested elements
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                elements.append(self._extract_function(node, source_lines, rel_path))

            elif isinstance(node, ast.ClassDef):
                elements.append(self._extract_class(node, source_lines, rel_path))
                # Also extract methods within the class
                for item in ast.iter_child_nodes(node):
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        elements.append(
                            self._extract_function(item, source_lines, rel_path, is_method=True)
                        )

        return FileAnalysis(
            file_path=rel_path,
            language="python",
            elements=elements,
            raw_source=source,
            loc=loc,
            imports=imports,
            parse_error=None,
        )

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source_lines: list[str],
        file_path: str,
        is_method: bool = False,
    ) -> CodeElement:
        """Extract a function or method definition."""
        line_start = node.lineno
        line_end = node.end_lineno or node.lineno
        code = self._get_source_segment(source_lines, line_start, line_end)

        # Determine element type
        if isinstance(node, ast.AsyncFunctionDef):
            element_type = "async_function"
        elif is_method:
            element_type = "method"
        else:
            element_type = "function"

        # Complexity hints
        complexity = self._analyze_complexity(node)

        return CodeElement(
            name=node.name,
            element_type=element_type,
            code=code,
            line_start=line_start,
            line_end=line_end,
            docstring=ast.get_docstring(node),
            complexity_hints=complexity,
        )

    def _extract_class(
        self, node: ast.ClassDef, source_lines: list[str], file_path: str
    ) -> CodeElement:
        """Extract a class definition."""
        line_start = node.lineno
        line_end = node.end_lineno or node.lineno
        code = self._get_source_segment(source_lines, line_start, line_end)

        bases = [self._get_name(b) for b in node.bases]
        method_count = sum(
            1
            for item in ast.iter_child_nodes(node)
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        )

        return CodeElement(
            name=node.name,
            element_type="class",
            code=code,
            line_start=line_start,
            line_end=line_end,
            docstring=ast.get_docstring(node),
            complexity_hints={
                "bases": bases,
                "method_count": method_count,
                "total_lines": line_end - line_start + 1,
            },
        )

    def _analyze_complexity(self, node: ast.AST) -> dict:
        """Compute complexity hints for a function/method."""
        hints: dict = {}

        # Argument count
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            total_args = (
                len(args.args)
                + len(args.posonlyargs)
                + len(args.kwonlyargs)
                + (1 if args.vararg else 0)
                + (1 if args.kwarg else 0)
            )
            # Exclude 'self' and 'cls'
            if args.args and args.args[0].arg in ("self", "cls"):
                total_args -= 1
            hints["arg_count"] = total_args

        # Count returns
        return_count = sum(1 for _ in ast.walk(node) if isinstance(_, ast.Return))
        hints["return_count"] = return_count

        # Nesting depth
        hints["max_nesting"] = self._max_nesting_depth(node)

        # Bare excepts
        bare_excepts = sum(
            1
            for child in ast.walk(node)
            if isinstance(child, ast.ExceptHandler) and child.type is None
        )
        if bare_excepts > 0:
            hints["bare_excepts"] = bare_excepts

        # Total lines
        if hasattr(node, "end_lineno") and hasattr(node, "lineno"):
            hints["total_lines"] = (node.end_lineno or node.lineno) - node.lineno + 1

        return hints

    def _max_nesting_depth(self, node: ast.AST, current: int = 0) -> int:
        """Calculate the maximum nesting depth of control flow statements."""
        max_depth = current
        nesting_nodes = (ast.For, ast.While, ast.If, ast.With, ast.Try)

        for child in ast.iter_child_nodes(node):
            if isinstance(child, nesting_nodes):
                depth = self._max_nesting_depth(child, current + 1)
                max_depth = max(max_depth, depth)
            else:
                depth = self._max_nesting_depth(child, current)
                max_depth = max(max_depth, depth)

        return max_depth

    @staticmethod
    def _get_source_segment(
        lines: list[str], start: int, end: int
    ) -> str:
        """Get source code between line numbers (1-indexed)."""
        return "\n".join(lines[start - 1 : end])

    @staticmethod
    def _get_name(node: ast.AST) -> str:
        """Safely extract a name from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{CodeParser._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        return "<unknown>"

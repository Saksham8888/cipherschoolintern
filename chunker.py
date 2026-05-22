"""
Smart code chunking module.
Splits large files into LLM-friendly chunks that respect AST boundaries,
preserving context (imports, class signatures) in each chunk.
"""

from __future__ import annotations

import logging

import tiktoken

from config.settings import CHUNK_TOKEN_LIMIT, CONTEXT_PREAMBLE_TOKENS
from core.models import CodeElement, FileAnalysis

logger = logging.getLogger(__name__)

# Use cl100k_base encoding (GPT-4 / GPT-4o-mini tokenizer)
try:
    _encoder = tiktoken.encoding_for_model("gpt-4o-mini")
except Exception:
    _encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    return len(_encoder.encode(text))


class CodeChunk:
    """A single chunk of code ready to be sent to the LLM."""

    def __init__(
        self,
        code: str,
        file_path: str,
        line_start: int,
        line_end: int,
        context_preamble: str = "",
        element_names: list[str] | None = None,
    ):
        self.code = code
        self.file_path = file_path
        self.line_start = line_start
        self.line_end = line_end
        self.context_preamble = context_preamble
        self.element_names = element_names or []
        self.token_count = count_tokens(context_preamble + "\n" + code)


class CodeChunker:
    """
    Splits FileAnalysis into LLM-sized chunks.

    Strategy:
    1. Build a context preamble (imports + top-level structure hints).
    2. Group elements that fit within the token limit.
    3. If a single element is too large, split it at logical line boundaries.
    """

    def __init__(self, token_limit: int = CHUNK_TOKEN_LIMIT):
        self.token_limit = token_limit
        self.preamble_limit = CONTEXT_PREAMBLE_TOKENS

    def chunk(self, analysis: FileAnalysis) -> list[CodeChunk]:
        """
        Split a FileAnalysis into chunks suitable for LLM review.

        Args:
            analysis: Parsed file analysis.

        Returns:
            List of CodeChunk objects.
        """
        if analysis.parse_error:
            # For unparseable files, send the raw source as one chunk
            return self._chunk_raw(analysis)

        if not analysis.elements:
            # No extractable elements — chunk the raw source
            return self._chunk_raw(analysis)

        # Build context preamble
        preamble = self._build_preamble(analysis)
        preamble_tokens = count_tokens(preamble)

        # Available tokens for code in each chunk
        available = self.token_limit - min(preamble_tokens, self.preamble_limit)
        if available < 200:
            available = self.token_limit // 2

        chunks: list[CodeChunk] = []
        current_elements: list[CodeElement] = []
        current_tokens = 0

        for element in analysis.elements:
            elem_tokens = count_tokens(element.code)

            # If single element exceeds limit, split it
            if elem_tokens > available:
                # Flush current batch first
                if current_elements:
                    chunks.append(self._make_chunk(current_elements, analysis.file_path, preamble))
                    current_elements = []
                    current_tokens = 0

                # Split the large element
                sub_chunks = self._split_large_element(element, analysis.file_path, preamble, available)
                chunks.extend(sub_chunks)
                continue

            # Check if adding this element exceeds the limit
            if current_tokens + elem_tokens > available:
                # Flush current batch
                chunks.append(self._make_chunk(current_elements, analysis.file_path, preamble))
                current_elements = []
                current_tokens = 0

            current_elements.append(element)
            current_tokens += elem_tokens

        # Flush remaining
        if current_elements:
            chunks.append(self._make_chunk(current_elements, analysis.file_path, preamble))

        return chunks

    def _build_preamble(self, analysis: FileAnalysis) -> str:
        """Build a context preamble with imports and structure overview."""
        lines = []

        # File info
        lines.append(f"# File: {analysis.file_path}")
        lines.append(f"# Lines of code: {analysis.loc}")
        lines.append("")

        # Imports
        if analysis.imports:
            lines.append("# --- Imports ---")
            for imp in analysis.imports[:20]:  # Cap imports
                lines.append(imp)
            if len(analysis.imports) > 20:
                lines.append(f"# ... and {len(analysis.imports) - 20} more imports")
            lines.append("")

        # Structure overview
        classes = [e for e in analysis.elements if e.element_type == "class"]
        functions = [e for e in analysis.elements if e.element_type in ("function", "async_function")]

        if classes or functions:
            lines.append("# --- Structure Overview ---")
            for cls in classes:
                lines.append(f"# class {cls.name} (lines {cls.line_start}-{cls.line_end})")
            for fn in functions:
                lines.append(f"# def {fn.name}() (lines {fn.line_start}-{fn.line_end})")
            lines.append("")

        return "\n".join(lines)

    def _make_chunk(
        self, elements: list[CodeElement], file_path: str, preamble: str
    ) -> CodeChunk:
        """Combine multiple elements into a single chunk."""
        code = "\n\n".join(e.code for e in elements)
        line_start = min(e.line_start for e in elements)
        line_end = max(e.line_end for e in elements)
        names = [e.name for e in elements]

        return CodeChunk(
            code=code,
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            context_preamble=preamble,
            element_names=names,
        )

    def _split_large_element(
        self, element: CodeElement, file_path: str, preamble: str, available: int
    ) -> list[CodeChunk]:
        """Split a single large element into multiple chunks by lines."""
        lines = element.code.split("\n")
        chunks: list[CodeChunk] = []

        current_lines: list[str] = []
        current_tokens = 0
        chunk_start_offset = 0

        for i, line in enumerate(lines):
            line_tokens = count_tokens(line)

            if current_tokens + line_tokens > available and current_lines:
                chunk_code = "\n".join(current_lines)
                chunks.append(
                    CodeChunk(
                        code=chunk_code,
                        file_path=file_path,
                        line_start=element.line_start + chunk_start_offset,
                        line_end=element.line_start + chunk_start_offset + len(current_lines) - 1,
                        context_preamble=preamble,
                        element_names=[f"{element.name} (part {len(chunks) + 1})"],
                    )
                )
                current_lines = []
                current_tokens = 0
                chunk_start_offset = i

            current_lines.append(line)
            current_tokens += line_tokens

        # Flush remaining lines
        if current_lines:
            chunk_code = "\n".join(current_lines)
            chunks.append(
                CodeChunk(
                    code=chunk_code,
                    file_path=file_path,
                    line_start=element.line_start + chunk_start_offset,
                    line_end=element.line_end,
                    context_preamble=preamble,
                    element_names=[f"{element.name} (part {len(chunks) + 1})"],
                )
            )

        return chunks

    def _chunk_raw(self, analysis: FileAnalysis) -> list[CodeChunk]:
        """Chunk raw source code when AST parsing failed or yielded no elements."""
        source = analysis.raw_source
        if not source.strip():
            return []

        total_tokens = count_tokens(source)
        if total_tokens <= self.token_limit:
            return [
                CodeChunk(
                    code=source,
                    file_path=analysis.file_path,
                    line_start=1,
                    line_end=analysis.loc or 1,
                    context_preamble=f"# File: {analysis.file_path} (raw — AST parse {'failed' if analysis.parse_error else 'empty'})\n",
                    element_names=["<raw_source>"],
                )
            ]

        # Split by lines
        lines = source.split("\n")
        chunks: list[CodeChunk] = []
        current: list[str] = []
        current_tokens = 0
        start_line = 1

        for i, line in enumerate(lines, 1):
            lt = count_tokens(line)
            if current_tokens + lt > self.token_limit and current:
                chunks.append(
                    CodeChunk(
                        code="\n".join(current),
                        file_path=analysis.file_path,
                        line_start=start_line,
                        line_end=i - 1,
                        context_preamble=f"# File: {analysis.file_path} (part {len(chunks) + 1})\n",
                        element_names=["<raw_source>"],
                    )
                )
                current = []
                current_tokens = 0
                start_line = i

            current.append(line)
            current_tokens += lt

        if current:
            chunks.append(
                CodeChunk(
                    code="\n".join(current),
                    file_path=analysis.file_path,
                    line_start=start_line,
                    line_end=len(lines),
                    context_preamble=f"# File: {analysis.file_path} (part {len(chunks) + 1})\n",
                    element_names=["<raw_source>"],
                )
            )

        return chunks

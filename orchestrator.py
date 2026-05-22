"""
Pipeline orchestrator module.
Coordinates the full ingestion → parse → chunk → review pipeline,
providing progress callbacks for the Streamlit UI.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

from core.ingestion import RepoIngestion, IngestionError
from core.parser import CodeParser
from core.chunker import CodeChunker, CodeChunk
from core.reviewer import CodeReviewer, ReviewerError
from core.models import FileAnalysis, ReviewComment, ReviewReport

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when the review pipeline encounters a fatal error."""
    pass


class CodeReviewAgent:
    """
    Orchestrates the end-to-end code review pipeline:
      1. Clone repository
      2. Discover & parse source files via AST
      3. Chunk code for LLM consumption
      4. Send chunks to LLM for review
      5. Aggregate results into a ReviewReport
    """

    def __init__(self, provider: str = "groq", api_key: str = "", model: str = ""):
        self.ingestion = RepoIngestion()
        self.parser = CodeParser()
        self.chunker = CodeChunker()
        self.provider = provider
        self.api_key = api_key
        self.model = model
        try:
            self.reviewer = CodeReviewer(provider=provider, api_key=api_key, model=model)
        except ReviewerError as e:
            self._init_error = str(e)
            self.reviewer = None  # Will fail at review step

    def run(
        self,
        repo_url: str,
        progress_callback: Callable[[str], None] | None = None,
        step_callback: Callable[[str, float], None] | None = None,
    ) -> ReviewReport:
        """
        Execute the full code review pipeline.

        Args:
            repo_url: GitHub repository URL.
            progress_callback: Called with status message strings.
            step_callback: Called with (step_name, progress_fraction 0.0–1.0).

        Returns:
            ReviewReport with all findings.

        Raises:
            PipelineError: On fatal pipeline failures.
        """
        start_time = time.time()
        repo_path: Path | None = None

        def log(msg: str):
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)

        try:
            # ── Step 1: Clone ──────────────────────────────────────────
            if step_callback:
                step_callback("Cloning repository", 0.0)

            log("📥 **Step 1/4: Cloning repository...**")
            repo_path = self.ingestion.clone(repo_url, progress_callback=log)
            repo_name = self.ingestion.extract_repo_name(repo_url)

            # ── Step 2: Discover & Parse ───────────────────────────────
            if step_callback:
                step_callback("Parsing source files", 0.2)

            log("🔍 **Step 2/4: Discovering & parsing files...**")
            file_infos = self.ingestion.discover_files(repo_path, progress_callback=log)

            if not file_infos:
                log("⚠️ No analyzable Python files found in this repository.")
                return ReviewReport(
                    repo_url=repo_url,
                    repo_name=repo_name,
                    files_analyzed=0,
                    files_skipped=0,
                    total_elements=0,
                    comments=[],
                )

            # Parse each file
            analyses: list[FileAnalysis] = []
            files_skipped = 0
            total_elements = 0

            for i, fi in enumerate(file_infos):
                analysis = self.parser.parse_file(fi)
                if analysis.parse_error:
                    log(f"   ⚠ Parse error in {fi['relative_path']}: {analysis.parse_error}")
                    files_skipped += 1
                else:
                    element_count = len(analysis.elements)
                    total_elements += element_count
                    log(f"   ✓ Parsed {fi['relative_path']} — {element_count} elements, {analysis.loc} LOC")
                analyses.append(analysis)

                if step_callback:
                    progress = 0.2 + 0.2 * ((i + 1) / len(file_infos))
                    step_callback("Parsing source files", progress)

            log(f"📊 Parsed {len(analyses)} files, extracted {total_elements} code elements.")

            # ── Step 3: Chunk ──────────────────────────────────────────
            if step_callback:
                step_callback("Chunking code", 0.4)

            log("✂️ **Step 3/4: Chunking code for LLM review...**")
            all_chunks: list[CodeChunk] = []
            for analysis in analyses:
                chunks = self.chunker.chunk(analysis)
                all_chunks.extend(chunks)

            log(f"📦 Created {len(all_chunks)} chunks for review.")

            # ── Step 4: LLM Review ─────────────────────────────────────
            if step_callback:
                step_callback("AI review in progress", 0.45)

            if not self.reviewer:
                raise PipelineError(
                    getattr(self, "_init_error", "LLM provider API key not configured properly.")
                )

            log("🤖 **Step 4/4: AI code review in progress...**")
            all_comments: list[ReviewComment] = []

            for i, chunk in enumerate(all_chunks):
                comments = self.reviewer.review_chunk(chunk, progress_callback=log)
                all_comments.extend(comments)

                if step_callback:
                    progress = 0.45 + 0.5 * ((i + 1) / len(all_chunks))
                    step_callback("AI review in progress", progress)

            # ── Build Report ───────────────────────────────────────────
            elapsed = time.time() - start_time

            report = ReviewReport(
                repo_url=repo_url,
                repo_name=repo_name,
                files_analyzed=len(analyses) - files_skipped,
                files_skipped=files_skipped,
                total_elements=total_elements,
                comments=all_comments,
            )

            log(f"\n✅ **Review complete!** Found {report.total_comments} issues in {elapsed:.1f}s.")
            log(f"   🔴 Critical: {report.severity_distribution.get('critical', 0)} | "
                f"🟠 Warning: {report.severity_distribution.get('warning', 0)} | "
                f"🔵 Info: {report.severity_distribution.get('info', 0)} | "
                f"🟢 Suggestion: {report.severity_distribution.get('suggestion', 0)}")

            if step_callback:
                step_callback("Complete", 1.0)

            return report

        except IngestionError as e:
            raise PipelineError(f"Ingestion failed: {e}")
        except ReviewerError as e:
            raise PipelineError(f"LLM review failed: {e}")
        except Exception as e:
            logger.exception("Pipeline failed with unexpected error")
            raise PipelineError(f"Pipeline error: {e}")
        finally:
            # Always clean up cloned repo
            if repo_path and repo_path.exists():
                self.ingestion.cleanup(repo_path)
                log("🧹 Cleaned up cloned repository.")

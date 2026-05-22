"""
Repository ingestion module.
Handles cloning GitHub repositories, validating URLs,
discovering source files, and cleanup.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from git import Repo, GitCommandError, InvalidGitRepositoryError

from config.settings import (
    CLONE_DIR,
    MAX_FILE_SIZE_CHARS,
    MAX_FILES_TO_ANALYZE,
    SKIP_DIRS,
    SKIP_FILES,
    SUPPORTED_EXTENSIONS,
)

logger = logging.getLogger(__name__)

# Regex for GitHub HTTPS URLs
GITHUB_URL_PATTERN = re.compile(
    r"^https?://github\.com/[\w.\-]+/[\w.\-]+(?:\.git)?/?$"
)


class IngestionError(Exception):
    """Raised when repository cloning or file discovery fails."""
    pass


class RepoIngestion:
    """Clones and discovers analyzable files in a GitHub repository."""

    def __init__(self) -> None:
        CLONE_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_url(url: str) -> str:
        """
        Validate and normalize a GitHub repository URL.

        Args:
            url: Raw URL input from the user.

        Returns:
            Cleaned URL string.

        Raises:
            IngestionError: If the URL format is invalid.
        """
        url = url.strip().rstrip("/")
        if url.endswith(".git"):
            url = url[:-4]

        # Re-add for clone
        clone_url = url + ".git"

        if not GITHUB_URL_PATTERN.match(url) and not GITHUB_URL_PATTERN.match(clone_url):
            raise IngestionError(
                f"Invalid GitHub URL: '{url}'. "
                "Expected format: https://github.com/owner/repo"
            )
        return url

    @staticmethod
    def extract_repo_name(url: str) -> str:
        """Extract 'owner/repo' from a GitHub URL."""
        parts = url.rstrip("/").split("/")
        if len(parts) >= 2:
            owner = parts[-2]
            repo = parts[-1].replace(".git", "")
            return f"{owner}/{repo}"
        return url

    def clone(self, url: str, progress_callback=None) -> Path:
        """
        Clone a GitHub repository to a local temporary directory.

        Args:
            url: Validated GitHub URL.
            progress_callback: Optional callable(message: str) for status updates.

        Returns:
            Path to the cloned repository.

        Raises:
            IngestionError: On clone failure.
        """
        url = self.validate_url(url)
        repo_name = self.extract_repo_name(url).replace("/", "_")
        local_path = CLONE_DIR / repo_name

        # Clean up any previous clone
        if local_path.exists():
            shutil.rmtree(local_path, ignore_errors=True)

        if progress_callback:
            progress_callback(f"🔄 Cloning {url}...")

        try:
            Repo.clone_from(
                url + ".git",
                str(local_path),
                depth=1,  # Shallow clone for speed
                no_single_branch=True,
            )
        except GitCommandError as e:
            error_msg = str(e)
            if "not found" in error_msg.lower() or "404" in error_msg:
                raise IngestionError(
                    f"Repository not found: {url}. "
                    "Check the URL and ensure the repo is public."
                )
            elif "authentication" in error_msg.lower() or "403" in error_msg:
                raise IngestionError(
                    f"Authentication required for {url}. "
                    "This agent only supports public repositories."
                )
            else:
                raise IngestionError(f"Failed to clone repository: {error_msg}")
        except Exception as e:
            raise IngestionError(f"Unexpected error during clone: {e}")

        if progress_callback:
            progress_callback("✅ Repository cloned successfully.")

        return local_path

    def discover_files(
        self, repo_path: Path, progress_callback=None
    ) -> list[dict]:
        """
        Walk the repository and find all analyzable source files.

        Args:
            repo_path: Path to the cloned repository root.
            progress_callback: Optional callable(message: str).

        Returns:
            List of dicts with keys: 'absolute_path', 'relative_path', 'size'.
        """
        discovered = []

        for file_path in sorted(repo_path.rglob("*")):
            # Skip directories
            if file_path.is_dir():
                continue

            # Skip hidden and excluded directories
            rel_parts = file_path.relative_to(repo_path).parts
            if any(part in SKIP_DIRS or part.startswith(".") for part in rel_parts[:-1]):
                continue

            # Check extension
            if file_path.suffix not in SUPPORTED_EXTENSIONS:
                continue

            # Skip specific files
            if file_path.name in SKIP_FILES:
                continue

            # Check file size
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if len(content) > MAX_FILE_SIZE_CHARS:
                    logger.warning(f"Skipping large file: {file_path} ({len(content)} chars)")
                    continue
            except Exception as e:
                logger.warning(f"Cannot read {file_path}: {e}")
                continue

            discovered.append({
                "absolute_path": file_path,
                "relative_path": str(file_path.relative_to(repo_path)),
                "size": len(content),
            })

            # Cap the number of files
            if len(discovered) >= MAX_FILES_TO_ANALYZE:
                if progress_callback:
                    progress_callback(
                        f"⚠️ File cap reached ({MAX_FILES_TO_ANALYZE}). "
                        "Analyzing the first batch only."
                    )
                break

        if not discovered:
            logger.info("No analyzable Python files found in the repository.")

        if progress_callback:
            progress_callback(f"📂 Found {len(discovered)} Python files to analyze.")

        return discovered

    @staticmethod
    def cleanup(repo_path: Path) -> None:
        """Remove cloned repository from disk."""
        if repo_path.exists():
            shutil.rmtree(repo_path, ignore_errors=True)
            logger.info(f"Cleaned up: {repo_path}")

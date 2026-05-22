"""
GitHub API integration for posting review comments to Pull Requests.
Bonus feature — requires a GitHub personal access token with repo scope.
"""

from __future__ import annotations

import logging
from typing import Optional

from github import Github, GithubException

from config.settings import GITHUB_TOKEN
from core.models import ReviewReport, ReviewComment

logger = logging.getLogger(__name__)


class GitHubAPIError(Exception):
    """Raised when GitHub API operations fail."""
    pass


class GitHubCommentPoster:
    """Posts review comments to a GitHub Pull Request."""

    def __init__(self, token: str = ""):
        self.token = token or GITHUB_TOKEN
        if not self.token:
            raise GitHubAPIError(
                "GitHub token not found. Set GITHUB_TOKEN in your .env file."
            )
        self.gh = Github(self.token)

    def post_review_comments(
        self,
        repo_name: str,
        pr_number: int,
        report: ReviewReport,
        min_confidence: int = 50,
        progress_callback=None,
    ) -> dict:
        """
        Post review comments to a GitHub PR.

        Args:
            repo_name: Repository in 'owner/repo' format.
            pr_number: Pull request number.
            report: The review report containing comments.
            min_confidence: Only post comments with confidence >= this threshold.
            progress_callback: Optional callable for status updates.

        Returns:
            Dict with 'posted' count and any 'errors'.
        """
        try:
            repo = self.gh.get_repo(repo_name)
            pr = repo.get_pull(pr_number)
        except GithubException as e:
            raise GitHubAPIError(f"Failed to access PR #{pr_number} in {repo_name}: {e}")

        # Filter comments by confidence
        eligible = [c for c in report.comments if c.confidence >= min_confidence]

        if not eligible:
            if progress_callback:
                progress_callback("No comments meet the confidence threshold for posting.")
            return {"posted": 0, "errors": []}

        posted = 0
        errors = []

        # Get the latest commit for inline comments
        commits = list(pr.get_commits())
        latest_commit = commits[-1] if commits else None

        for comment in eligible:
            try:
                body = self._format_pr_comment(comment)

                if latest_commit:
                    # Try to post as inline comment
                    try:
                        pr.create_review_comment(
                            body=body,
                            commit=latest_commit,
                            path=comment.file_path,
                            line=comment.line_end,
                        )
                        posted += 1
                        continue
                    except GithubException:
                        pass  # Fall back to general comment

                # Fallback: post as general PR comment
                pr.create_issue_comment(body)
                posted += 1

            except GithubException as e:
                error_msg = f"Failed to post comment for {comment.file_path}:{comment.line_start}: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)

        if progress_callback:
            progress_callback(f"✅ Posted {posted}/{len(eligible)} comments to PR #{pr_number}")

        return {"posted": posted, "errors": errors}

    @staticmethod
    def _format_pr_comment(comment: ReviewComment) -> str:
        """Format a ReviewComment as a GitHub PR comment body."""
        severity_emoji = {
            "critical": "🔴",
            "warning": "🟠",
            "info": "🔵",
            "suggestion": "🟢",
        }
        emoji = severity_emoji.get(comment.severity, "")
        confidence_label = f"{'⚠️ Verify This — ' if comment.needs_verification else ''}"

        lines = [
            f"### {emoji} {comment.severity.upper()}: {comment.title}",
            f"",
            f"**{confidence_label}Confidence: {comment.confidence}%** | Category: {comment.category.replace('_', ' ').title()}",
            f"",
            comment.description,
        ]

        if comment.suggestion:
            lines.append(f"")
            lines.append(f"**💡 Suggestion:** {comment.suggestion}")

        lines.append(f"")
        lines.append(f"*— AI Code Review Agent*")

        return "\n".join(lines)

    def validate_access(self, repo_name: str) -> bool:
        """Check if the token has access to the repository."""
        try:
            repo = self.gh.get_repo(repo_name)
            return True
        except GithubException:
            return False

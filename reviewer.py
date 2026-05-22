"""
LLM-powered code review module.
Sends code chunks to Groq's Llama models and parses structured review comments.
Includes retry logic, rate limiting, and confidence scoring.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Callable

from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from pydantic import ValidationError

from config.settings import (
    PROVIDERS,
    GROQ_API_KEY,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_MAX_RETRIES,
)
from core.chunker import CodeChunk
from core.models import ReviewComment

logger = logging.getLogger(__name__)

# ── System Prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior software engineer performing a thorough code review.
Analyze the provided Python code and identify real, actionable issues.

For EACH issue you find, you must assign a confidence score from 0 to 100:
  • 90–100: You are certain this is a genuine issue (e.g., clear bug, security flaw)
  • 70–89:  Highly likely issue with minor ambiguity
  • 50–69:  Probable issue, but context-dependent — reviewer should verify
  • 30–49:  Possible issue — recommend human verification
  • 0–29:   Speculative — might be intentional or framework-specific

RULES:
1. Only flag genuine issues. Do NOT invent problems that don't exist.
2. Quality over quantity — fewer accurate comments beat many speculative ones.
3. If the code is well-written and correct, return an EMPTY array [].
4. Be specific with line numbers — they must match the code provided.
5. Provide actionable suggestions, not vague advice.
6. Consider the file context (imports, structure) when reviewing.

CATEGORIES (use exactly these strings):
  bug_risk, security, performance, style, best_practice, documentation, error_handling

SEVERITIES (use exactly these strings):
  critical, warning, info, suggestion

You MUST respond with valid JSON only. No markdown, no explanations outside the JSON."""

# ── User Prompt Template ──────────────────────────────────────────────────
USER_PROMPT_TEMPLATE = """Review the following Python code from `{file_path}` (lines {line_start}–{line_end}):

<file_context>
{context_preamble}
</file_context>

<code>
{code}
</code>

Return a JSON object with a single key "comments" containing an array of review comments.
Each comment must have exactly these fields:
{{
  "comments": [
    {{
      "line_start": <int>,
      "line_end": <int>,
      "category": "<bug_risk|security|performance|style|best_practice|documentation|error_handling>",
      "severity": "<critical|warning|info|suggestion>",
      "title": "<short title>",
      "description": "<detailed explanation>",
      "suggestion": "<actionable fix or null>",
      "confidence": <0-100>
    }}
  ]
}}

If no issues are found, return: {{"comments": []}}"""


class ReviewerError(Exception):
    """Raised when the LLM review process fails."""
    pass


class CodeReviewer:
    """
    Sends code chunks to Groq's LLM API and returns structured ReviewComment objects.
    Handles retries, rate limiting, and JSON validation.
    """

    def __init__(self, provider: str = "groq", api_key: str = "", model: str = ""):
        self.provider = provider.lower() if provider else "groq"
        if self.provider not in PROVIDERS:
            raise ReviewerError(f"Unsupported provider: {self.provider}")

        provider_info = PROVIDERS[self.provider]

        # Determine which key to use
        resolved_key = api_key if api_key else GROQ_API_KEY

        if not resolved_key:
            raise ReviewerError(
                f"{provider_info['name']} API key not found. "
                f"Set {provider_info['env_key']} in your .env file or enter it in the sidebar."
            )

        self.model = model or provider_info["default_model"]

        # Initialize OpenAI-compatible client with Groq base URL
        self.client = OpenAI(
            api_key=resolved_key,
            base_url=provider_info["base_url"]
        )

        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
        self.max_retries = LLM_MAX_RETRIES

    def review_chunk(
        self,
        chunk: CodeChunk,
        progress_callback: Callable[[str], None] | None = None,
    ) -> list[ReviewComment]:
        """
        Review a single code chunk via the LLM.

        Args:
            chunk: The code chunk to review.
            progress_callback: Optional callable for status updates.

        Returns:
            List of ReviewComment objects.
        """
        user_prompt = USER_PROMPT_TEMPLATE.format(
            file_path=chunk.file_path,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            context_preamble=chunk.context_preamble,
            code=chunk.code,
        )

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )

                raw_content = response.choices[0].message.content or "{}"
                comments = self._parse_response(raw_content, chunk)

                if progress_callback:
                    progress_callback(
                        f"   ✓ {chunk.file_path} (lines {chunk.line_start}–{chunk.line_end}): "
                        f"{len(comments)} issue(s) found"
                    )

                return comments

            except (RateLimitError, APITimeoutError) as e:
                wait_time = 2 ** (attempt + 1)
                logger.warning(f"Rate limit / timeout (attempt {attempt + 1}), waiting {wait_time}s...")
                if progress_callback:
                    progress_callback(f"   ⏳ Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)

            except APIError as e:
                logger.error(f"API error: {e}")
                if attempt < self.max_retries:
                    time.sleep(2)
                    continue
                if progress_callback:
                    progress_callback(f"   ✗ API error on {chunk.file_path}: {e}")
                return []

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    continue
                if progress_callback:
                    progress_callback(f"   ✗ Malformed JSON from LLM for {chunk.file_path}")
                return []

            except Exception as e:
                logger.error(f"Unexpected error reviewing chunk: {e}")
                if progress_callback:
                    progress_callback(f"   ✗ Error on {chunk.file_path}: {e}")
                return []

        return []

    def _parse_response(
        self, raw_json: str, chunk: CodeChunk
    ) -> list[ReviewComment]:
        """
        Parse and validate the LLM's JSON response into ReviewComment objects.

        Args:
            raw_json: Raw JSON string from the LLM.
            chunk: The original chunk (for file_path context).

        Returns:
            List of validated ReviewComment objects.
        """
        data = json.loads(raw_json)

        # Handle both {"comments": [...]} and bare [...]
        if isinstance(data, dict):
            comments_raw = data.get("comments", [])
        elif isinstance(data, list):
            comments_raw = data
        else:
            logger.warning(f"Unexpected response format: {type(data)}")
            return []

        comments: list[ReviewComment] = []

        for item in comments_raw:
            try:
                # Inject file_path from chunk context
                item["file_path"] = chunk.file_path

                # Clamp confidence to 0–100
                if "confidence" in item:
                    item["confidence"] = max(0, min(100, int(item["confidence"])))

                # Validate line numbers are within chunk range
                if "line_start" in item:
                    item["line_start"] = max(1, int(item["line_start"]))
                if "line_end" in item:
                    item["line_end"] = max(
                        item.get("line_start", 1), int(item["line_end"])
                    )

                comment = ReviewComment(**item)
                comments.append(comment)

            except (ValidationError, KeyError, TypeError, ValueError) as e:
                logger.warning(f"Skipping malformed comment: {e}")
                continue

        return comments

    def estimate_cost(self, total_tokens: int) -> float:
        """
        Estimate API cost. Groq's free tier means no cost.
        """
        return 0.0


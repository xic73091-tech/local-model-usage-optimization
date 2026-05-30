"""
GitHub Monitor for Self-Learning System

Searches GitHub for repositories, commits, and issues related to
LLM inference optimization, GGUF models, and small-VRAM techniques.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from .knowledge_base import GitHubFinding, KnowledgeBase

logger = logging.getLogger(__name__)

# Search queries targeting our optimization domain
DEFAULT_QUERIES = [
    "llama.cpp quantization optimization",
    "GGUF inference speed optimization",
    "LLM small VRAM inference",
    "KV cache optimization LLM",
    "speculative decoding implementation",
    "dynamic layer loading GPU offload",
    "LLM memory optimization techniques",
    "llama.cpp performance benchmark",
]


@dataclass
class GitHubConfig:
    """Configuration for GitHub monitoring."""
    token: str = ""
    queries: List[str] = field(default_factory=lambda: list(DEFAULT_QUERIES))
    max_results_per_query: int = 10
    min_stars: int = 0
    language_filter: str = ""  # e.g. "python", "cpp"
    rate_limit_delay: float = 2.0  # seconds between API calls


class GitHubMonitor:
    """Monitors GitHub for relevant repos, issues, and commits."""

    API_BASE = "https://api.github.com"

    def __init__(self, knowledge_base: KnowledgeBase, config: Optional[GitHubConfig] = None):
        self._kb = knowledge_base
        self._config = config or GitHubConfig()
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        """Initialize HTTP client with auth headers."""
        token = self._config.token or os.environ.get("GITHUB_TOKEN", "")
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.AsyncClient(
            base_url=self.API_BASE,
            headers=headers,
            timeout=30.0,
            follow_redirects=False,  # Security: prevent SSRF via redirects
        )
        logger.info("GitHub monitor initialized (auth=%s)", bool(token))

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def run_search(self) -> List[GitHubFinding]:
        """Run all configured search queries and store findings."""
        all_findings: List[GitHubFinding] = []

        for query in self._config.queries:
            try:
                findings = await self._search_repos(query)
                all_findings.extend(findings)
                await asyncio.sleep(self._config.rate_limit_delay)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 403:
                    logger.warning("GitHub rate limit hit, stopping search")
                    break
                logger.error("GitHub API error for query '%s': %s", query, e)
            except Exception as e:
                logger.error("Search failed for '%s': %s", query, e)

        # Deduplicate by URL
        seen_urls: set = set()
        unique_findings: List[GitHubFinding] = []
        for f in all_findings:
            if f.url not in seen_urls:
                seen_urls.add(f.url)
                unique_findings.append(f)

        # Store to knowledge base
        new_count = 0
        for finding in unique_findings:
            is_new = await self._kb.add_github_finding(finding)
            if is_new:
                new_count += 1

        logger.info("GitHub search: %d total, %d new findings", len(unique_findings), new_count)
        return unique_findings

    async def _search_repos(self, query: str) -> List[GitHubFinding]:
        """Search GitHub repositories."""
        params = {
            "q": query,
            "sort": "updated",
            "order": "desc",
            "per_page": min(self._config.max_results_per_query, 100),
        }
        if self._config.min_stars > 0:
            params["q"] += f" stars:>={self._config.min_stars}"
        if self._config.language_filter:
            params["q"] += f" language:{self._config.language_filter}"

        resp = await self._client.get("/search/repositories", params=params)
        resp.raise_for_status()
        data = resp.json()

        findings = []
        for item in data.get("items", []):
            finding = GitHubFinding(
                url=item.get("html_url", ""),
                title=item.get("full_name", ""),
                description=item.get("description", "") or "",
                source="github",
                repo_name=item.get("full_name", ""),
                finding_type="repo",
                stars=item.get("stargazers_count", 0),
                language=item.get("language", "") or "",
                topics=item.get("topics", []),
                relevance=self._assess_relevance(item),
                score=self._calculate_score(item),
                discovered_at=time.time(),
                metadata={
                    "forks": item.get("forks_count", 0),
                    "open_issues": item.get("open_issues_count", 0),
                    "created_at": item.get("created_at", ""),
                    "updated_at": item.get("updated_at", ""),
                    "license": (item.get("license") or {}).get("spdx_id", ""),
                },
            )
            findings.append(finding)

        return findings

    async def search_issues(self, query: str, max_results: int = 10) -> List[GitHubFinding]:
        """Search GitHub issues and pull requests."""
        params = {
            "q": f"{query} is:issue",
            "sort": "updated",
            "order": "desc",
            "per_page": max_results,
        }

        resp = await self._client.get("/search/issues", params=params)
        resp.raise_for_status()
        data = resp.json()

        findings = []
        for item in data.get("items", []):
            finding = GitHubFinding(
                url=item.get("html_url", ""),
                title=item.get("title", ""),
                description=(item.get("body", "") or "")[:500],
                source="github",
                repo_name=item.get("repository_url", "").split("repos/")[-1],
                finding_type="issue",
                language="",
                relevance="medium",
                score=0.5,
                discovered_at=time.time(),
                metadata={
                    "state": item.get("state", ""),
                    "comments": item.get("comments", 0),
                    "labels": [l.get("name", "") for l in item.get("labels", [])],
                },
            )
            findings.append(finding)

        return findings

    async def check_rate_limit(self) -> Dict[str, Any]:
        """Check GitHub API rate limit status."""
        resp = await self._client.get("/rate_limit")
        resp.raise_for_status()
        data = resp.json()
        core = data.get("resources", {}).get("core", {})
        search = data.get("resources", {}).get("search", {})
        return {
            "core_remaining": core.get("remaining", 0),
            "core_limit": core.get("limit", 0),
            "search_remaining": search.get("remaining", 0),
            "search_limit": search.get("limit", 0),
            "core_reset": core.get("reset", 0),
            "search_reset": search.get("reset", 0),
        }

    @staticmethod
    def _assess_relevance(item: Dict[str, Any]) -> str:
        """Assess relevance of a GitHub repo to our project."""
        text = " ".join([
            item.get("full_name", ""),
            item.get("description", "") or "",
            " ".join(item.get("topics", [])),
        ]).lower()

        high_keywords = [
            "llama.cpp", "gguf", "quantiz", "kv cache", "speculative decod",
            "small vram", "layer offload", "dynamic loading", "inference optim",
        ]
        medium_keywords = [
            "llm", "transformer", "language model", "gpu", "cuda", "metal",
            "onnx", "tensorrt", "vllm", "mlx",
        ]

        for kw in high_keywords:
            if kw in text:
                return "high"
        for kw in medium_keywords:
            if kw in text:
                return "medium"
        return "low"

    @staticmethod
    def _calculate_score(item: Dict[str, Any]) -> float:
        """Calculate a relevance score for a GitHub repo."""
        stars = item.get("stargazers_count", 0)
        forks = item.get("forks_count", 0)

        # Star-based score (logarithmic, 0-1 range, capped at 10k stars)
        import math
        star_score = min(1.0, math.log10(max(1, stars)) / 4.0)  # log10(10000) = 4

        # Fork ratio (forks/stars, higher = more actively forked)
        fork_ratio = min(1.0, forks / max(1, stars)) if stars > 0 else 0

        # Recency (boost for recently updated)
        updated_at = item.get("updated_at", "")
        recency_score = 0.5  # default
        if updated_at:
            try:
                from datetime import datetime, timezone
                updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                days_old = (now - updated).days
                recency_score = max(0, 1.0 - days_old / 365.0)
            except (ValueError, TypeError):
                pass

        return round(0.5 * star_score + 0.2 * fork_ratio + 0.3 * recency_score, 3)

"""
Academic Searcher for Self-Learning System

Searches arxiv and Semantic Scholar for papers related to
LLM inference optimization, quantization, and memory efficiency.
"""

from __future__ import annotations

import asyncio
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from .knowledge_base import KnowledgeBase, Paper

logger = logging.getLogger(__name__)

# arxiv search queries
ARXIV_QUERIES = [
    "quantization large language model inference",
    "KV cache compression LLM",
    "speculative decoding language model",
    "LLM memory optimization GPU",
    "low-bit quantization transformer",
    "layer-wise inference offloading",
    "GGUF format quantization",
    "small GPU large model inference",
]

# Semantic Scholar search queries (broader, citation-rich)
SEMANTIC_SCHOLAR_QUERIES = [
    "LLM inference optimization",
    "quantization language model",
    "KV cache optimization",
    "speculative decoding",
    "memory efficient transformer inference",
]


@dataclass
class AcademicConfig:
    """Configuration for academic search."""
    arxiv_max_results: int = 10
    semantic_scholar_max_results: int = 10
    min_citations: int = 0
    rate_limit_delay: float = 3.0  # seconds between arxiv API calls
    s2_rate_limit_delay: float = 5.0  # seconds between Semantic Scholar calls (stricter)
    arxiv_categories: List[str] = field(default_factory=lambda: ["cs.CL", "cs.LG", "cs.DC"])
    semantic_scholar_api_key: str = ""


class AcademicSearcher:
    """Searches academic databases for relevant papers."""

    ARXIV_API = "https://export.arxiv.org/api/query"
    S2_API = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, knowledge_base: KnowledgeBase, config: Optional[AcademicConfig] = None):
        self._kb = knowledge_base
        self._config = config or AcademicConfig()
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        """Initialize HTTP client."""
        headers = {}
        if self._config.semantic_scholar_api_key:
            headers["x-api-key"] = self._config.semantic_scholar_api_key
        self._client = httpx.AsyncClient(timeout=30.0, headers=headers)
        logger.info("Academic searcher initialized")

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def run_search(self) -> List[Paper]:
        """Run all configured searches and store papers."""
        all_papers: List[Paper] = []

        # Search arxiv
        for query in ARXIV_QUERIES:
            try:
                papers = await self._search_arxiv(query)
                all_papers.extend(papers)
                await asyncio.sleep(self._config.rate_limit_delay)
            except Exception as e:
                logger.error("arxiv search failed for '%s': %s", query, e)

        # Search Semantic Scholar (stricter rate limits without API key)
        for query in SEMANTIC_SCHOLAR_QUERIES:
            try:
                papers = await self._search_semantic_scholar(query)
                all_papers.extend(papers)
                await asyncio.sleep(self._config.s2_rate_limit_delay)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning("Semantic Scholar rate limit hit, stopping search")
                    break
                logger.error("Semantic Scholar search failed for '%s': %s", query, e)
            except Exception as e:
                logger.error("Semantic Scholar search failed for '%s': %s", query, e)

        # Deduplicate by URL
        seen_urls: set = set()
        unique_papers: List[Paper] = []
        for p in all_papers:
            if p.url not in seen_urls:
                seen_urls.add(p.url)
                unique_papers.append(p)

        # Store to knowledge base
        new_count = 0
        for paper in unique_papers:
            is_new = await self._kb.add_paper(paper)
            if is_new:
                new_count += 1

        logger.info("Academic search: %d total, %d new papers", len(unique_papers), new_count)
        return unique_papers

    async def _search_arxiv(self, query: str) -> List[Paper]:
        """Search arxiv API for papers."""
        # Build search query with category filter
        cat_filter = " OR ".join(f"cat:{c}" for c in self._config.arxiv_categories)
        search_query = f"({query}) AND ({cat_filter})"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": self._config.arxiv_max_results,
            "sortBy": "lastUpdatedDate",
            "sortOrder": "descending",
        }

        resp = await self._client.get(self.ARXIV_API, params=params)
        resp.raise_for_status()

        return self._parse_arxiv_response(resp.text)

    def _parse_arxiv_response(self, xml_text: str) -> List[Paper]:
        """Parse arxiv Atom XML response."""
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        root = ET.fromstring(xml_text)
        papers = []

        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            published_el = entry.find("atom:published", ns)

            title = title_el.text.strip().replace("\n", " ") if title_el is not None else ""
            abstract = summary_el.text.strip().replace("\n", " ") if summary_el is not None else ""
            published = published_el.text.strip()[:10] if published_el is not None else ""

            # Get arxiv URL
            url = ""
            for link in entry.findall("atom:link", ns):
                if link.get("type") == "text/html":
                    url = link.get("href", "")
                    break
            if not url:
                id_el = entry.find("atom:id", ns)
                url = id_el.text.strip() if id_el is not None else ""

            # Get authors
            authors = []
            for author in entry.findall("atom:author", ns):
                name_el = author.find("atom:name", ns)
                if name_el is not None:
                    authors.append(name_el.text.strip())

            # Get categories as topics
            topics = []
            for cat in entry.findall("atom:category", ns):
                term = cat.get("term", "")
                if term:
                    topics.append(term)

            paper = Paper(
                url=url,
                title=title,
                abstract=abstract[:1000],  # Truncate long abstracts
                authors=authors,
                source="arxiv",
                published_at=published,
                relevance=self._assess_paper_relevance(title, abstract),
                score=self._calculate_paper_score(title, abstract, 0),
                discovered_at=time.time(),
                topics=topics,
            )
            papers.append(paper)

        return papers

    async def _search_semantic_scholar(self, query: str) -> List[Paper]:
        """Search Semantic Scholar API for papers."""
        params = {
            "query": query,
            "limit": self._config.semantic_scholar_max_results,
            "fields": "title,abstract,url,authors,year,citationCount,venue,externalIds,publicationDate",
        }

        resp = await self._client.get(f"{self.S2_API}/paper/search", params=params)
        resp.raise_for_status()
        data = resp.json()

        papers = []
        for item in data.get("data", []):
            citations = item.get("citationCount", 0)
            if citations < self._config.min_citations:
                continue

            authors = [a.get("name", "") for a in item.get("authors", [])]
            ext_ids = item.get("externalIds", {}) or {}
            doi = ext_ids.get("DOI", "")

            paper = Paper(
                url=item.get("url", ""),
                title=item.get("title", ""),
                abstract=(item.get("abstract", "") or "")[:1000],
                authors=authors,
                source="semantic_scholar",
                doi=doi,
                published_at=item.get("publicationDate", "") or str(item.get("year", "")),
                venue=item.get("venue", "") or "",
                citation_count=citations,
                relevance=self._assess_paper_relevance(
                    item.get("title", ""), item.get("abstract", "") or ""
                ),
                score=self._calculate_paper_score(
                    item.get("title", ""), item.get("abstract", "") or "", citations
                ),
                discovered_at=time.time(),
            )
            papers.append(paper)

        return papers

    @staticmethod
    def _assess_paper_relevance(title: str, abstract: str) -> str:
        """Assess paper relevance to our optimization domain."""
        text = f"{title} {abstract}".lower()

        high_keywords = [
            "quantiz", "gguf", "kv cache", "speculative decod",
            "layer offload", "small vram", "memory optimiz",
            "inference speed", "token per second",
        ]
        medium_keywords = [
            "llm", "language model", "transformer", "inference",
            "gpu", "cuda", "model compress", "pruning", "distill",
        ]

        for kw in high_keywords:
            if kw in text:
                return "high"
        for kw in medium_keywords:
            if kw in text:
                return "medium"
        return "low"

    @staticmethod
    def _calculate_paper_score(title: str, abstract: str, citations: int) -> float:
        """Calculate a relevance score for a paper."""
        import math

        text = f"{title} {abstract}".lower()

        # Keyword relevance (0-0.5)
        kw_score = 0.0
        high_kws = ["quantiz", "kv cache", "speculative", "gguf", "layer offload"]
        for kw in high_kws:
            if kw in text:
                kw_score += 0.1
        kw_score = min(0.5, kw_score)

        # Citation score (0-0.3, logarithmic)
        cite_score = min(0.3, math.log10(max(1, citations)) / 5.0) if citations > 0 else 0.0

        # Recency bonus (0-0.2)
        recency_score = 0.1  # default
        if "2025" in text or "2026" in text:
            recency_score = 0.2
        elif "2024" in text:
            recency_score = 0.15

        return round(kw_score + cite_score + recency_score, 3)

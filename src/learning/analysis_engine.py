"""
Analysis Engine for Self-Learning System

Analyzes collected findings and papers to generate actionable
optimization proposals for the local model optimization project.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .knowledge_base import (
    GitHubFinding,
    KnowledgeBase,
    OptimizationProposal,
    Paper,
)

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result of analyzing findings and papers."""
    proposals: List[OptimizationProposal] = field(default_factory=list)
    high_relevance_findings: List[Dict[str, Any]] = field(default_factory=list)
    high_relevance_papers: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""


class AnalysisEngine:
    """Analyzes findings and generates optimization proposals."""

    # Keywords that map to project modules
    MODULE_KEYWORDS = {
        "src/optimization/dynamic_loader.py": [
            "layer offload", "dynamic load", "layer swap", "memory map",
            "partial load", "layer-wise",
        ],
        "src/optimization/pipeline_parallel.py": [
            "pipeline parallel", "batch split", "micro batch", "pipeline schedul",
        ],
        "src/optimization/kv_cache.py": [
            "kv cache", "key-value cache", "cache compress", "cache optimiz",
            "attention cache",
        ],
        "src/optimization/speculative_decoding.py": [
            "speculative decod", "draft model", "verification decod",
            "parallel decod",
        ],
        "src/optimization/quantization_aware.py": [
            "quantiz", "qat", "quantization-aware", "low-bit", "int4", "int8",
            "gguf", "gptq", "awq",
        ],
        "src/optimization/continuous_batching.py": [
            "continuous batch", "dynamic batch", "request schedul",
        ],
        "src/optimization/model_sharding.py": [
            "model shard", "tensor parallel", "model split", "device map",
        ],
        "src/api/server.py": [
            "api optimiz", "request handl", "inference server",
        ],
    }

    # High-impact optimization patterns
    IMPACT_PATTERNS = {
        "high": [
            "10x faster", "significant improvement", "state-of-the-art",
            "breakthrough", "order of magnitude", "dramatically reduce",
            "memory reduction", "speed up", "novel approach",
        ],
        "medium": [
            "improvement", "better", "optimize", "efficient",
            "reduce memory", "faster inference", "technique",
        ],
    }

    def __init__(self, knowledge_base: KnowledgeBase):
        self._kb = knowledge_base

    async def analyze(self, days_back: int = 7) -> AnalysisResult:
        """Analyze recent findings and generate proposals."""
        result = AnalysisResult()

        # Get recent findings and papers
        since = time.time() - days_back * 86400
        findings = await self._kb.get_github_findings(limit=100, since=since)
        papers = await self._kb.get_papers(limit=100, since=since)

        # Filter high-relevance items
        result.high_relevance_findings = [
            f for f in findings if f.get("relevance") == "high"
        ]
        result.high_relevance_papers = [
            p for p in papers if p.get("relevance") == "high"
        ]

        # Generate proposals from findings
        for finding in result.high_relevance_findings:
            proposals = self._generate_proposals_from_finding(finding)
            result.proposals.extend(proposals)

        # Generate proposals from papers
        for paper in result.high_relevance_papers:
            proposals = self._generate_proposals_from_paper(paper)
            result.proposals.extend(proposals)

        # Deduplicate proposals by title similarity
        result.proposals = self._deduplicate_proposals(result.proposals)

        # Save proposals to knowledge base
        for proposal in result.proposals:
            await self._kb.add_proposal(proposal)

        # Generate summary
        result.summary = self._generate_summary(result)

        logger.info(
            "Analysis complete: %d findings, %d papers → %d proposals",
            len(result.high_relevance_findings),
            len(result.high_relevance_papers),
            len(result.proposals),
        )

        return result

    def _generate_proposals_from_finding(self, finding: Dict[str, Any]) -> List[OptimizationProposal]:
        """Generate optimization proposals from a GitHub finding."""
        proposals = []
        text = f"{finding.get('title', '')} {finding.get('description', '')}".lower()

        affected = self._identify_affected_modules(text)
        impact = self._assess_impact(text)

        # Only generate proposals for medium+ impact with known affected modules
        if impact in ("high", "medium") and affected:
            proposal = OptimizationProposal(
                title=f"Apply insights from: {finding.get('title', 'unknown')}",
                description=(
                    f"Review {finding.get('url', '')} for applicable optimizations.\n\n"
                    f"Description: {finding.get('description', 'N/A')}\n"
                    f"Stars: {finding.get('stars', 0)}, "
                    f"Language: {finding.get('language', 'N/A')}\n"
                    f"Affected modules: {', '.join(affected)}"
                ),
                source_url=finding.get("url", ""),
                affected_modules=affected,
                estimated_impact=impact,
                test_plan=self._generate_test_plan(affected),
                status="pending",
                created_at=time.time(),
            )
            proposals.append(proposal)

        return proposals

    def _generate_proposals_from_paper(self, paper: Dict[str, Any]) -> List[OptimizationProposal]:
        """Generate optimization proposals from an academic paper."""
        proposals = []
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()

        affected = self._identify_affected_modules(text)
        impact = self._assess_impact(text)

        if impact in ("high", "medium") and affected:
            proposal = OptimizationProposal(
                title=f"Implement technique from: {paper.get('title', 'unknown')[:80]}",
                description=(
                    f"Paper: {paper.get('url', '')}\n"
                    f"Authors: {', '.join(paper.get('authors', [])[:3])}\n"
                    f"Published: {paper.get('published_at', 'N/A')}\n"
                    f"Citations: {paper.get('citation_count', 0)}\n\n"
                    f"Abstract: {paper.get('abstract', 'N/A')[:500]}\n\n"
                    f"Affected modules: {', '.join(affected)}"
                ),
                source_url=paper.get("url", ""),
                affected_modules=affected,
                estimated_impact=impact,
                test_plan=self._generate_test_plan(affected),
                status="pending",
                created_at=time.time(),
            )
            proposals.append(proposal)

        return proposals

    def _identify_affected_modules(self, text: str) -> List[str]:
        """Identify which project modules are relevant to the text."""
        affected = []
        for module, keywords in self.MODULE_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    affected.append(module)
                    break
        return affected

    def _assess_impact(self, text: str) -> str:
        """Assess the potential impact level from text content."""
        for kw in self.IMPACT_PATTERNS["high"]:
            if kw in text:
                return "high"
        for kw in self.IMPACT_PATTERNS["medium"]:
            if kw in text:
                return "medium"
        return "low"

    def _generate_test_plan(self, affected_modules: List[str]) -> str:
        """Generate a test plan based on affected modules."""
        plans = []
        for module in affected_modules:
            if "dynamic_loader" in module:
                plans.append("Benchmark layer loading speed and memory usage before/after")
            elif "kv_cache" in module:
                plans.append("Measure KV cache memory reduction and inference latency")
            elif "speculative" in module:
                plans.append("Compare tokens/sec with and without speculative decoding")
            elif "quantization" in module:
                plans.append("Evaluate quantized model quality (perplexity) vs speed gain")
            elif "pipeline" in module:
                plans.append("Benchmark pipeline throughput with different batch sizes")
            elif "batching" in module:
                plans.append("Test continuous batching under concurrent requests")
            elif "sharding" in module:
                plans.append("Measure cross-device latency and memory distribution")
            elif "server" in module:
                plans.append("Load test API endpoints for latency and throughput")

        if not plans:
            plans.append("Run existing test suite and benchmark suite")

        return "\n".join(f"- {p}" for p in plans)

    def _deduplicate_proposals(self, proposals: List[OptimizationProposal]) -> List[OptimizationProposal]:
        """Remove duplicate proposals based on overlapping affected modules."""
        if not proposals:
            return []

        # Group by affected module overlap
        seen_titles: set = set()
        unique = []
        for p in proposals:
            # Simple dedup by exact title
            if p.title not in seen_titles:
                seen_titles.add(p.title)
                unique.append(p)

        return unique

    def _generate_summary(self, result: AnalysisResult) -> str:
        """Generate a human-readable summary of the analysis."""
        lines = [
            f"## Analysis Summary",
            f"",
            f"- **High-relevance GitHub findings**: {len(result.high_relevance_findings)}",
            f"- **High-relevance papers**: {len(result.high_relevance_papers)}",
            f"- **New optimization proposals**: {len(result.proposals)}",
            f"",
        ]

        if result.proposals:
            lines.append("### Top Proposals")
            for i, p in enumerate(result.proposals[:5], 1):
                lines.append(f"{i}. [{p.estimated_impact.upper()}] {p.title}")
                lines.append(f"   Modules: {', '.join(p.affected_modules)}")

        if not result.proposals:
            lines.append("No new actionable proposals generated this cycle.")

        return "\n".join(lines)

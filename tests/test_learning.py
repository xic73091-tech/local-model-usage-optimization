"""
Tests for the self-learning system.
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from src.learning.knowledge_base import (
    DailyReport,
    FindingSource,
    GitHubFinding,
    KnowledgeBase,
    OptimizationProposal,
    Paper,
    RelevanceLevel,
)


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database path."""
    return str(tmp_path / "test_knowledge.db")


@pytest.mark.asyncio
async def test_knowledge_base_init(tmp_db):
    """Test knowledge base initialization creates tables."""
    kb = KnowledgeBase(tmp_db)
    await kb.initialize()
    stats = await kb.get_stats()
    assert stats["github_findings"] == 0
    assert stats["papers"] == 0
    assert stats["optimization_proposals"] == 0
    assert stats["daily_reports"] == 0
    await kb.close()


@pytest.mark.asyncio
async def test_add_and_get_github_finding(tmp_db):
    """Test adding and retrieving GitHub findings."""
    kb = KnowledgeBase(tmp_db)
    await kb.initialize()

    finding = GitHubFinding(
        url="https://github.com/test/repo",
        title="test/repo",
        description="A test repo",
        stars=100,
        language="Python",
        topics=["llm", "optimization"],
        relevance="high",
        score=0.8,
        discovered_at=time.time(),
    )

    is_new = await kb.add_github_finding(finding)
    assert is_new is True

    # Duplicate should return False
    is_new = await kb.add_github_finding(finding)
    assert is_new is False

    findings = await kb.get_github_findings(limit=10)
    assert len(findings) == 1
    assert findings[0]["title"] == "test/repo"
    assert findings[0]["stars"] == 100
    assert findings[0]["topics"] == ["llm", "optimization"]

    await kb.close()


@pytest.mark.asyncio
async def test_add_and_get_paper(tmp_db):
    """Test adding and retrieving papers."""
    kb = KnowledgeBase(tmp_db)
    await kb.initialize()

    paper = Paper(
        url="https://arxiv.org/abs/2024.12345",
        title="Test Paper on LLM Optimization",
        abstract="This paper presents optimization techniques...",
        authors=["Alice", "Bob"],
        source="arxiv",
        published_at="2024-01-15",
        citation_count=42,
        relevance="high",
        score=0.9,
        discovered_at=time.time(),
        topics=["cs.CL", "cs.LG"],
    )

    is_new = await kb.add_paper(paper)
    assert is_new is True

    papers = await kb.get_papers(limit=10)
    assert len(papers) == 1
    assert papers[0]["title"] == "Test Paper on LLM Optimization"
    assert papers[0]["authors"] == ["Alice", "Bob"]
    assert papers[0]["citation_count"] == 42

    await kb.close()


@pytest.mark.asyncio
async def test_add_and_update_proposal(tmp_db):
    """Test adding and updating optimization proposals."""
    kb = KnowledgeBase(tmp_db)
    await kb.initialize()

    proposal = OptimizationProposal(
        title="Implement KV cache compression",
        description="Apply insights from paper XYZ",
        source_url="https://arxiv.org/abs/2024.12345",
        affected_modules=["src/optimization/kv_cache.py"],
        estimated_impact="high",
        test_plan="Benchmark KV cache memory usage",
        status="pending",
    )

    proposal_id = await kb.add_proposal(proposal)
    assert proposal_id > 0

    proposals = await kb.get_proposals(status="pending")
    assert len(proposals) == 1
    assert proposals[0]["title"] == "Implement KV cache compression"

    # Update status
    await kb.update_proposal_status(
        proposal_id, "validated", {"speed_improvement": "15%"}
    )
    proposals = await kb.get_proposals(status="validated")
    assert len(proposals) == 1
    assert proposals[0]["test_results"] == {"speed_improvement": "15%"}

    await kb.close()


@pytest.mark.asyncio
async def test_daily_report(tmp_db):
    """Test saving and retrieving daily reports."""
    kb = KnowledgeBase(tmp_db)
    await kb.initialize()

    report = DailyReport(
        date="2024-01-15",
        github_findings_count=5,
        papers_count=3,
        proposals_count=2,
        top_findings=[{"title": "test/repo", "url": "https://github.com/test/repo"}],
        top_papers=[{"title": "Test Paper", "url": "https://arxiv.org/abs/2024.12345"}],
        proposals=[{"title": "Proposal 1", "impact": "high"}],
        summary="Test summary",
    )

    is_new = await kb.save_daily_report(report)
    assert is_new is True

    # Get by date
    retrieved = await kb.get_daily_report("2024-01-15")
    assert retrieved is not None
    assert retrieved["github_findings_count"] == 5
    assert retrieved["summary"] == "Test summary"

    # Get latest
    latest = await kb.get_latest_report()
    assert latest is not None
    assert latest["date"] == "2024-01-15"

    await kb.close()


@pytest.mark.asyncio
async def test_paper_filter_by_relevance(tmp_db):
    """Test filtering papers by relevance level."""
    kb = KnowledgeBase(tmp_db)
    await kb.initialize()

    for i, rel in enumerate(["high", "medium", "low"]):
        paper = Paper(
            url=f"https://arxiv.org/abs/2024.{i}",
            title=f"Paper {rel}",
            relevance=rel,
            score=1.0 - i * 0.3,
            discovered_at=time.time(),
        )
        await kb.add_paper(paper)

    high_papers = await kb.get_papers(relevance="high")
    assert len(high_papers) == 1
    assert high_papers[0]["relevance"] == "high"

    all_papers = await kb.get_papers()
    assert len(all_papers) == 3

    await kb.close()


@pytest.mark.asyncio
async def test_github_finding_filter_by_since(tmp_db):
    """Test filtering GitHub findings by time."""
    kb = KnowledgeBase(tmp_db)
    await kb.initialize()

    old_finding = GitHubFinding(
        url="https://github.com/old/repo",
        title="old/repo",
        discovered_at=time.time() - 86400 * 10,  # 10 days ago
    )
    new_finding = GitHubFinding(
        url="https://github.com/new/repo",
        title="new/repo",
        discovered_at=time.time(),
    )

    await kb.add_github_finding(old_finding)
    await kb.add_github_finding(new_finding)

    # Filter to last 5 days
    recent = await kb.get_github_findings(since=time.time() - 86400 * 5)
    assert len(recent) == 1
    assert recent[0]["title"] == "new/repo"

    # No filter gets all
    all_findings = await kb.get_github_findings()
    assert len(all_findings) == 2

    await kb.close()


@pytest.mark.asyncio
async def test_stats(tmp_db):
    """Test knowledge base statistics."""
    kb = KnowledgeBase(tmp_db)
    await kb.initialize()

    # Add one of each
    await kb.add_github_finding(GitHubFinding(
        url="https://github.com/test/repo", title="test", discovered_at=time.time()
    ))
    await kb.add_paper(Paper(
        url="https://arxiv.org/abs/2024.1", title="paper", discovered_at=time.time()
    ))
    await kb.add_proposal(OptimizationProposal(
        title="Proposal", description="desc"
    ))

    stats = await kb.get_stats()
    assert stats["github_findings"] == 1
    assert stats["papers"] == 1
    assert stats["optimization_proposals"] == 1
    assert stats["daily_reports"] == 0

    await kb.close()

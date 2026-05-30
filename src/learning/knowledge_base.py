"""
Persistent Knowledge Base for Self-Learning System

SQLite-backed storage for GitHub findings, academic papers,
optimization proposals, and test results.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

logger = logging.getLogger(__name__)


class FindingSource(str, Enum):
    GITHUB = "github"
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    MANUAL = "manual"


class RelevanceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class GitHubFinding:
    """A finding from GitHub (repo, issue, commit, etc.)."""
    url: str
    title: str
    description: str = ""
    source: str = "github"
    repo_name: str = ""
    finding_type: str = "repo"  # repo / issue / commit / pr
    stars: int = 0
    language: str = ""
    topics: List[str] = field(default_factory=list)
    relevance: str = "medium"
    score: float = 0.0
    discovered_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Paper:
    """An academic paper finding."""
    url: str
    title: str
    abstract: str = ""
    authors: List[str] = field(default_factory=list)
    source: str = "arxiv"  # arxiv / semantic_scholar
    doi: str = ""
    published_at: str = ""
    venue: str = ""
    citation_count: int = 0
    relevance: str = "medium"
    score: float = 0.0
    discovered_at: float = field(default_factory=time.time)
    topics: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationProposal:
    """An actionable optimization proposal derived from findings."""
    title: str
    description: str
    source_url: str = ""
    affected_modules: List[str] = field(default_factory=list)
    estimated_impact: str = "medium"  # high / medium / low
    test_plan: str = ""
    status: str = "pending"  # pending / testing / validated / rejected
    created_at: float = field(default_factory=time.time)
    tested_at: Optional[float] = None
    test_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DailyReport:
    """A daily learning report."""
    date: str = ""
    github_findings_count: int = 0
    papers_count: int = 0
    proposals_count: int = 0
    top_findings: List[Dict[str, Any]] = field(default_factory=list)
    top_papers: List[Dict[str, Any]] = field(default_factory=list)
    proposals: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    created_at: float = field(default_factory=time.time)


class KnowledgeBase:
    """SQLite-backed knowledge base for the self-learning system."""

    def __init__(self, db_path: str | Path = "data/knowledge.db"):
        self._db_path = Path(db_path)
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Create database and tables if they don't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row

        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS github_findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                source TEXT DEFAULT 'github',
                repo_name TEXT DEFAULT '',
                finding_type TEXT DEFAULT 'repo',
                stars INTEGER DEFAULT 0,
                language TEXT DEFAULT '',
                topics TEXT DEFAULT '[]',
                relevance TEXT DEFAULT 'medium',
                score REAL DEFAULT 0.0,
                discovered_at REAL NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                abstract TEXT DEFAULT '',
                authors TEXT DEFAULT '[]',
                source TEXT DEFAULT 'arxiv',
                doi TEXT DEFAULT '',
                published_at TEXT DEFAULT '',
                venue TEXT DEFAULT '',
                citation_count INTEGER DEFAULT 0,
                relevance TEXT DEFAULT 'medium',
                score REAL DEFAULT 0.0,
                discovered_at REAL NOT NULL,
                topics TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS optimization_proposals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                source_url TEXT DEFAULT '',
                affected_modules TEXT DEFAULT '[]',
                estimated_impact TEXT DEFAULT 'medium',
                test_plan TEXT DEFAULT '',
                status TEXT DEFAULT 'pending',
                created_at REAL NOT NULL,
                tested_at REAL,
                test_results TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS daily_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                github_findings_count INTEGER DEFAULT 0,
                papers_count INTEGER DEFAULT 0,
                proposals_count INTEGER DEFAULT 0,
                top_findings TEXT DEFAULT '[]',
                top_papers TEXT DEFAULT '[]',
                proposals TEXT DEFAULT '[]',
                summary TEXT DEFAULT '',
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_github_url ON github_findings(url);
            CREATE INDEX IF NOT EXISTS idx_papers_url ON papers(url);
            CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);
            CREATE INDEX IF NOT EXISTS idx_proposals_status ON optimization_proposals(status);
            CREATE INDEX IF NOT EXISTS idx_reports_date ON daily_reports(date);
        """)
        await self._db.commit()
        logger.info("Knowledge base initialized at %s", self._db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # GitHub Findings
    # ------------------------------------------------------------------

    async def add_github_finding(self, finding: GitHubFinding) -> bool:
        """Add a GitHub finding. Returns True if new, False if duplicate."""
        try:
            cursor = await self._db.execute(
                """INSERT INTO github_findings
                   (url, title, description, source, repo_name, finding_type,
                    stars, language, topics, relevance, score, discovered_at, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (finding.url, finding.title, finding.description, finding.source,
                 finding.repo_name, finding.finding_type, finding.stars, finding.language,
                 json.dumps(finding.topics), finding.relevance, finding.score,
                 finding.discovered_at, json.dumps(finding.metadata), time.time()),
            )
            await self._db.commit()
            return cursor.rowcount > 0
        except aiosqlite.IntegrityError:
            return False

    async def get_github_findings(
        self,
        limit: int = 50,
        relevance: Optional[str] = None,
        since: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve GitHub findings with optional filters."""
        query = "SELECT * FROM github_findings WHERE 1=1"
        params: list = []

        if relevance:
            query += " AND relevance = ?"
            params.append(relevance)
        if since:
            query += " AND discovered_at >= ?"
            params.append(since)

        query += " ORDER BY score DESC, discovered_at DESC LIMIT ?"
        params.append(limit)

        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Papers
    # ------------------------------------------------------------------

    async def add_paper(self, paper: Paper) -> bool:
        """Add a paper. Returns True if new, False if duplicate."""
        try:
            await self._db.execute(
                """INSERT OR IGNORE INTO papers
                   (url, title, abstract, authors, source, doi, published_at,
                    venue, citation_count, relevance, score, discovered_at, topics, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (paper.url, paper.title, paper.abstract,
                 json.dumps(paper.authors), paper.source, paper.doi, paper.published_at,
                 paper.venue, paper.citation_count, paper.relevance, paper.score,
                 paper.discovered_at, json.dumps(paper.topics), json.dumps(paper.metadata),
                 time.time()),
            )
            await self._db.commit()
            return True
        except aiosqlite.IntegrityError:
            return False

    async def get_papers(
        self,
        limit: int = 50,
        relevance: Optional[str] = None,
        since: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve papers with optional filters."""
        query = "SELECT * FROM papers WHERE 1=1"
        params: list = []

        if relevance:
            query += " AND relevance = ?"
            params.append(relevance)
        if since:
            query += " AND discovered_at >= ?"
            params.append(since)

        query += " ORDER BY score DESC, discovered_at DESC LIMIT ?"
        params.append(limit)

        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Optimization Proposals
    # ------------------------------------------------------------------

    async def add_proposal(self, proposal: OptimizationProposal) -> int:
        """Add an optimization proposal. Returns the proposal ID."""
        cursor = await self._db.execute(
            """INSERT INTO optimization_proposals
               (title, description, source_url, affected_modules, estimated_impact,
                test_plan, status, created_at, tested_at, test_results)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (proposal.title, proposal.description, proposal.source_url,
             json.dumps(proposal.affected_modules), proposal.estimated_impact,
             proposal.test_plan, proposal.status, proposal.created_at,
             proposal.tested_at, json.dumps(proposal.test_results)),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def update_proposal_status(
        self,
        proposal_id: int,
        status: str,
        test_results: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update proposal status and optionally test results."""
        tested_at = time.time() if status in ("validated", "rejected") else None
        await self._db.execute(
            """UPDATE optimization_proposals
               SET status = ?, tested_at = ?, test_results = ?
               WHERE id = ?""",
            (status, tested_at, json.dumps(test_results or {}), proposal_id),
        )
        await self._db.commit()

    async def get_proposals(
        self,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve optimization proposals."""
        query = "SELECT * FROM optimization_proposals WHERE 1=1"
        params: list = []

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Daily Reports
    # ------------------------------------------------------------------

    async def save_daily_report(self, report: DailyReport) -> bool:
        """Save a daily report. Returns True if new, False if updated."""
        try:
            await self._db.execute(
                """INSERT INTO daily_reports
                   (date, github_findings_count, papers_count, proposals_count,
                    top_findings, top_papers, proposals, summary, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (report.date, report.github_findings_count, report.papers_count,
                 report.proposals_count, json.dumps(report.top_findings),
                 json.dumps(report.top_papers), json.dumps(report.proposals),
                 report.summary, report.created_at),
            )
            await self._db.commit()
            return True
        except aiosqlite.IntegrityError:
            # Update existing report for this date
            await self._db.execute(
                """UPDATE daily_reports SET
                   github_findings_count = ?, papers_count = ?, proposals_count = ?,
                   top_findings = ?, top_papers = ?, proposals = ?, summary = ?, created_at = ?
                   WHERE date = ?""",
                (report.github_findings_count, report.papers_count, report.proposals_count,
                 json.dumps(report.top_findings), json.dumps(report.top_papers),
                 json.dumps(report.proposals), report.summary, report.created_at, report.date),
            )
            await self._db.commit()
            return False

    async def get_daily_report(self, date: str) -> Optional[Dict[str, Any]]:
        """Get a daily report by date."""
        async with self._db.execute(
            "SELECT * FROM daily_reports WHERE date = ?", (date,)
        ) as cursor:
            row = await cursor.fetchone()
            return self._row_to_dict(row) if row else None

    async def get_latest_report(self) -> Optional[Dict[str, Any]]:
        """Get the most recent daily report."""
        async with self._db.execute(
            "SELECT * FROM daily_reports ORDER BY created_at DESC LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
            return self._row_to_dict(row) if row else None

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    async def get_stats(self) -> Dict[str, int]:
        """Get knowledge base statistics."""
        stats = {}
        for table in ["github_findings", "papers", "optimization_proposals", "daily_reports"]:
            async with self._db.execute(f"SELECT COUNT(*) FROM {table}") as cursor:
                row = await cursor.fetchone()
                stats[table] = row[0]
        return stats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: aiosqlite.Row) -> Dict[str, Any]:
        """Convert a database row to a dict, parsing JSON fields."""
        d = dict(row)
        for key in ("topics", "metadata", "authors", "affected_modules", "test_results",
                     "top_findings", "top_papers", "proposals"):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

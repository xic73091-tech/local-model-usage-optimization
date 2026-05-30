"""
Learning Scheduler for Self-Learning System

Orchestrates the daily learning cycle: search → analyze → report.
Uses APScheduler for periodic execution.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .academic_searcher import AcademicConfig, AcademicSearcher
from .analysis_engine import AnalysisEngine
from .github_monitor import GitHubConfig, GitHubMonitor
from .knowledge_base import KnowledgeBase
from .report_generator import ReportGenerator

logger = logging.getLogger(__name__)


@dataclass
class LearningConfig:
    """Configuration for the learning scheduler."""
    enabled: bool = True
    cron_hour: int = 2
    cron_minute: int = 0
    analysis_days_back: int = 7
    github_token: str = ""
    semantic_scholar_api_key: str = ""


class LearningScheduler:
    """Orchestrates the self-learning cycle on a schedule."""

    def __init__(self, config: Optional[LearningConfig] = None):
        self._config = config or LearningConfig()
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._kb: Optional[KnowledgeBase] = None
        self._github: Optional[GitHubMonitor] = None
        self._academic: Optional[AcademicSearcher] = None
        self._analysis: Optional[AnalysisEngine] = None
        self._reporter: Optional[ReportGenerator] = None
        self._running = False
        self._last_run: Optional[datetime] = None
        self._last_error: Optional[str] = None
        self._callbacks: List[Callable[..., Coroutine]] = []

    async def initialize(self) -> None:
        """Initialize all sub-components and the scheduler."""
        # Knowledge base
        self._kb = KnowledgeBase("data/knowledge.db")
        await self._kb.initialize()

        # GitHub monitor
        gh_config = GitHubConfig(token=self._config.github_token)
        self._github = GitHubMonitor(self._kb, gh_config)
        await self._github.initialize()

        # Academic searcher
        ac_config = AcademicConfig(
            semantic_scholar_api_key=self._config.semantic_scholar_api_key,
        )
        self._academic = AcademicSearcher(self._kb, ac_config)
        await self._academic.initialize()

        # Analysis engine
        self._analysis = AnalysisEngine(self._kb)

        # Report generator
        self._reporter = ReportGenerator(self._kb, "reports")

        # APScheduler
        self._scheduler = AsyncIOScheduler()
        self._scheduler.add_job(
            self._run_cycle,
            CronTrigger(
                hour=self._config.cron_hour,
                minute=self._config.cron_minute,
            ),
            id="learning_cycle",
            name="Daily Learning Cycle",
            replace_existing=True,
        )

        logger.info(
            "Learning scheduler initialized (cron: %02d:%02d)",
            self._config.cron_hour,
            self._config.cron_minute,
        )

    def on_cycle_complete(self, callback: Callable[..., Coroutine]) -> None:
        """Register a callback for when a cycle completes."""
        self._callbacks.append(callback)

    async def start(self) -> None:
        """Start the scheduler."""
        if not self._config.enabled:
            logger.info("Learning scheduler disabled")
            return

        if self._scheduler:
            self._scheduler.start()
            self._running = True
            logger.info("Learning scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler and clean up resources."""
        if self._scheduler and self._running:
            self._scheduler.shutdown(wait=False)
            self._running = False

        if self._github:
            await self._github.close()
        if self._academic:
            await self._academic.close()
        if self._kb:
            await self._kb.close()

        logger.info("Learning scheduler stopped")

    async def run_now(self) -> Dict[str, Any]:
        """Trigger an immediate learning cycle (manual run)."""
        return await self._run_cycle()

    async def _run_cycle(self) -> Dict[str, Any]:
        """Execute one complete learning cycle: search → analyze → report."""
        logger.info("Starting learning cycle...")
        results = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "github_findings": 0,
            "papers": 0,
            "proposals": 0,
            "report_path": "",
            "error": None,
        }

        try:
            # Step 1: Search GitHub
            logger.info("Step 1/3: Searching GitHub...")
            findings = await self._github.run_search()
            results["github_findings"] = len(findings)

            # Step 2: Search academic databases
            logger.info("Step 2/3: Searching academic databases...")
            papers = await self._academic.run_search()
            results["papers"] = len(papers)

            # Step 3: Analyze and generate report
            logger.info("Step 3/3: Analyzing and generating report...")
            analysis = await self._analysis.analyze(self._config.analysis_days_back)
            results["proposals"] = len(analysis.proposals)

            report_path = await self._reporter.generate_daily_report(analysis)
            results["report_path"] = report_path

            self._last_run = datetime.now(timezone.utc)
            self._last_error = None

            logger.info(
                "Learning cycle complete: %d findings, %d papers, %d proposals",
                results["github_findings"],
                results["papers"],
                results["proposals"],
            )

        except Exception as e:
            self._last_error = str(e)
            results["error"] = str(e)
            logger.error("Learning cycle failed: %s", e)

        # Notify callbacks
        for cb in self._callbacks:
            try:
                await cb(results)
            except Exception as e:
                logger.error("Callback error: %s", e)

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            "enabled": self._config.enabled,
            "running": self._running,
            "cron": f"{self._config.cron_hour:02d}:{self._config.cron_minute:02d}",
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "last_error": self._last_error,
            "next_run": (
                self._scheduler.get_job("learning_cycle").next_run_time.isoformat()
                if self._scheduler and self._scheduler.get_job("learning_cycle")
                else None
            ),
        }

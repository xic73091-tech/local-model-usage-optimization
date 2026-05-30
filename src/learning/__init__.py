"""
Self-Learning System for Local Model Optimization

Monitors GitHub and academic databases daily for new techniques,
analyzes findings, and generates optimization proposals.
"""

from .knowledge_base import KnowledgeBase
from .github_monitor import GitHubMonitor
from .academic_searcher import AcademicSearcher
from .analysis_engine import AnalysisEngine
from .report_generator import ReportGenerator
from .scheduler import LearningScheduler

__all__ = [
    "KnowledgeBase",
    "GitHubMonitor",
    "AcademicSearcher",
    "AnalysisEngine",
    "ReportGenerator",
    "LearningScheduler",
]

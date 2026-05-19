"""Reporting module — top-journal-quality report generation.

This module transforms raw pipeline results into publication-quality reports
following general top-journal standards (Nature, Science, JACS, JPCL, etc.).
"""

from .figure_generator import FigureGenerator
from .composite_figure import CompositeFigure
from .figure_selector import FigureSelector
from .shap_analyzer import SHAPAnalyzer
from .molecular_plots import MolecularPlotter
from .narrative_engine import NarrativeEngine
from .top_journal_report import TopJournalReport
from .si_generator import SIGenerator
from .report_bundle import ReportBundle
from .image_embedder import embed_markdown_images
from .research_crew import ClaimAuditorAgent, ReviewerAgent

__all__ = [
    "FigureGenerator",
    "CompositeFigure",
    "FigureSelector",
    "SHAPAnalyzer",
    "MolecularPlotter",
    "NarrativeEngine",
    "TopJournalReport",
    "SIGenerator",
    "ReportBundle",
    "embed_markdown_images",
    "ClaimAuditorAgent",
    "ReviewerAgent",
]

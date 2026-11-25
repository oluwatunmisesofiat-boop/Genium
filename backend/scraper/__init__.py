"""Genie Scraper — forked lean acquisition layer from sami-mag07/searchify (v0.3.0, MIT).

Provides local-first web search and multi-stage content/PDF/JSON-LD extraction
with SSRF protection and failure reason codes. Local cross-encoder reranking,
CLI, MCP server, and user profile config have been removed to keep memory and
dependencies minimal.
"""
from __future__ import annotations

from .extract import (
    extract,
    extract_document,
    extract_main_text,
    extract_page,
    extract_report,
    read_pages,
    read_pages_ex,
)
from .search import SearchBackendError, search, search_images, search_report

__all__ = [
    "search",
    "search_report",
    "search_images",
    "extract",
    "extract_report",
    "read_pages",
    "read_pages_ex",
    "extract_document",
    "extract_main_text",
    "extract_page",
    "SearchBackendError",
]

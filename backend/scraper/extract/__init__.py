"""Content extraction: fetch a page and return its clean article text.

A load ladder: a fast static fetch first, then a real browser only when the
static result is too thin, with an SSRF-safe outbound proxy for the browser
stages. Exposes a small, stable API.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..reasons import code_for, status_for
from .htmltext import extract_document, extract_main_text, extract_page
from .reader import read_pages, read_pages_ex

__all__ = [
    "extract",
    "extract_report",
    "read_pages",
    "read_pages_ex",
    "extract_document",
    "extract_main_text",
    "extract_page",
]


def extract(urls: List[str], fmt: str = "text", timeout: Optional[float] = None) -> List[Dict]:
    """Fetch each address and return its clean content.

    Each result is a dict with url, title, published, author, description,
    content and links. `fmt` is "text" (default) or "markdown". Pages that fail
    are left out of the result.
    """
    docs, _failed = read_pages(list(urls), fmt, timeout)
    return docs


def extract_report(
    urls: List[str],
    fmt: str = "text",
    timeout: Optional[float] = None,
    *,
    max_stage: Optional[str] = None,
    budget: Optional[str] = None,
    cloud_fallback: Optional[bool] = None,
    structured: bool = False,
) -> Dict:
    """Like `extract`, but says what happened: ``{"results", "failed", "status"}``.

    ``results`` are the clean pages, exactly as `extract` returns them. ``failed``
    lists ``{url, error, reason}`` per page that could not be read: ``error`` is the
    human text, ``reason`` a fixed code. ``status`` is
    ``{read, failed, thin, reasons: {code: n}}``.

    ``max_stage`` (plain, tls, browser, stealth), ``budget`` (low, normal, high) and
    ``cloud_fallback`` limit the load for this call, always below the operator cap
    SEARCHIFY_MAX_STAGE. The raw ``structured`` block (JSON-LD, og:/twitter: meta)
    is left out unless ``structured=True``.
    """
    from .. import limits as limits_mod  # noqa: PLC0415

    kwargs = limits_mod.explicit(max_stage, budget, cloud_fallback, None)
    docs, failed, thin = read_pages_ex(list(urls), fmt, timeout, structured=structured, **kwargs)
    failed = [
        {
            "url": f.get("url"),
            "error": f.get("error"),
            "reason": f.get("reason") or code_for(f.get("error")),
        }
        for f in failed
    ]
    return {"results": docs, "failed": failed, "status": status_for(len(docs), failed, thin.values())}

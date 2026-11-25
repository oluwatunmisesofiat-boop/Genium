"""Multi-angle discovery runner combining local Greenbook, parallel search, image search, and extraction."""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from registry.greenbook import GreenbookStore, default_db_path
from scraper.config import get_settings
from scraper.extract import extract_report
from scraper.extract.structured import _is_junk_image
from scraper.search import SearchBackendError, search_images, search_report

from .models import DiscoveredSource, DiscoveryBundle, DiscoverySeed, SearchQuery
from .planner import (
    build_image_query,
    build_queries,
    build_trusted_image_queries,
    build_trusted_pharmacy_web_queries,
    enrich_seed_from_greenbook,
)
from .trust import canonicalize_url, classify_url, extract_domain, is_disqualified_image_source, rank_and_select

_RELEVANCE_STOPWORDS = frozenset(
    {
        "mg",
        "mcg",
        "ml",
        "gram",
        "grams",
        "tablet",
        "tablets",
        "capsule",
        "capsules",
        "syrup",
        "suspension",
        "injection",
        "cream",
        "ointment",
        "drops",
        "oral",
        "pack",
        "packs",
        "box",
        "strip",
        "bp",
        "usp",
        "plus",
        "extra",
        "forte",
        "ds",
        "bd",
        "xr",
        "sr",
    }
)


def _lookup_greenbook(
    seed: DiscoverySeed,
    *,
    greenbook: Optional[GreenbookStore],
    db_path: Optional[Path | str],
) -> List[Dict]:
    store = greenbook
    if store is None:
        target_path = Path(db_path) if db_path is not None else default_db_path()
        if not target_path.exists():
            return []
        store = GreenbookStore(target_path)

    matches: List[Dict] = []
    seen_ids: set = set()

    if seed.nafdac_number:
        for row in store.lookup_by_nafdac(seed.nafdac_number):
            pid = row.get("product_id")
            if pid not in seen_ids:
                seen_ids.add(pid)
                matches.append(row)

    if seed.product_name and len(matches) < 5:
        for row in store.search(seed.product_name, manufacturer=seed.manufacturer, limit=5):
            pid = row.get("product_id")
            if pid not in seen_ids:
                seen_ids.add(pid)
                matches.append(row)

    return matches[:5]


def _default_search_settings(
    search_fn: Optional[Callable],
    max_workers: Optional[int],
    stagger_s: Optional[float],
) -> tuple[int, float]:
    if max_workers is not None and stagger_s is not None:
        return max_workers, stagger_s

    if search_fn is not None:
        return (max_workers or 4), (0.0 if stagger_s is None else stagger_s)

    settings = get_settings()
    using_faster_backend = bool(settings.searxng_url or settings.serper_api_key)
    default_workers = 4 if using_faster_backend else 1
    default_stagger = 0.0 if using_faster_backend else 1.0
    return (
        max_workers if max_workers is not None else default_workers,
        stagger_s if stagger_s is not None else default_stagger,
    )


def _run_one_query(
    query_obj: SearchQuery,
    *,
    delay_s: float,
    max_results: int,
    country: Optional[str],
    search_fn: Callable,
) -> tuple[SearchQuery, List[Dict], Optional[Dict]]:
    if delay_s > 0:
        time.sleep(delay_s)
    kwargs: Dict = {"max_results": max_results}
    if query_obj.include_domains:
        kwargs["include_domains"] = query_obj.include_domains
    if query_obj.time_range:
        kwargs["time_range"] = query_obj.time_range
    if country and not query_obj.include_domains:
        kwargs["country"] = country

    try:
        res = search_fn(query_obj.query, **kwargs)
    except SearchBackendError as exc:
        return query_obj, [], {"angle": query_obj.angle, "query": query_obj.query, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return query_obj, [], {
            "angle": query_obj.angle,
            "query": query_obj.query,
            "error": f"{type(exc).__name__}: {exc}",
        }

    if isinstance(res, dict):
        hits = list(res.get("results") or [])
        err = res.get("error")
        err_entry = (
            {"angle": query_obj.angle, "query": query_obj.query, "error": str(err)}
            if err and not hits
            else None
        )
        return query_obj, hits, err_entry

    return query_obj, list(res or []), None


def _run_image_search(
    img_query: str,
    *,
    delay_s: float,
    max_results: int,
    country: Optional[str],
    manufacturer_hint: Optional[str],
    image_search_fn: Callable,
    trusted_store=None,
) -> tuple[List[Dict], Optional[Dict]]:
    if delay_s > 0:
        time.sleep(delay_s)
    kwargs: Dict = {"max_results": max_results}
    if country:
        kwargs["country"] = country
    try:
        raw_hits = list(image_search_fn(img_query, **kwargs) or [])
    except SearchBackendError as exc:
        return [], {"angle": "image_search", "query": img_query, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return [], {
            "angle": "image_search",
            "query": img_query,
            "error": f"{type(exc).__name__}: {exc}",
        }

    filtered: List[Dict] = []
    seen: set[str] = set()
    for hit in raw_hits:
        img_url = str(hit.get("url") or "").strip()
        if not img_url or img_url in seen:
            continue
        page_url = str(hit.get("source_page_url") or "").strip()
        tier, label = classify_url(page_url or img_url, manufacturer_hint=manufacturer_hint)
        if tier == 0:
            continue
        title = str(hit.get("title") or hit.get("alt") or "").strip()
        width = hit.get("width")
        height = hit.get("height")
        if _is_junk_image(img_url, alt=title, width=width, height=height):
            continue
        if is_disqualified_image_source(img_url, page_url, title):
            continue
        trusted_match = (
            trusted_store.is_verified_image_source(page_url or img_url)
            if trusted_store is not None
            else None
        )
        seen.add(img_url)
        filtered.append(
            {
                "url": img_url,
                "source_page_url": page_url,
                "title": title,
                "width": width,
                "height": height,
                "source": str(hit.get("source") or "image_search"),
                "source_trust_tier": tier,
                "source_trust_label": label,
                "captured_from_domain": trusted_match["domain"] if trusted_match else None,
                "captured_from_type": trusted_match["source_type"] if trusted_match else None,
            }
        )
    return filtered, None


def _brand_tokens(product_name: Optional[str]) -> List[str]:
    if not product_name:
        return []
    tokens = []
    for tok in re.findall(r"[A-Za-z0-9]+", product_name.lower()):
        if len(tok) < 3 or tok.isdigit() or tok in _RELEVANCE_STOPWORDS:
            continue
        if tok not in tokens:
            tokens.append(tok)
    return tokens


def _check_relevance(source: DiscoveredSource, seed: DiscoverySeed) -> Tuple[bool, str]:
    """Filter out off-topic articles and general pages with at most a passing mention."""
    if source.is_smpc_pdf:
        return True, "smpc_pdf"

    title_low = (source.title or "").lower()
    content_low = (source.content or "").lower()

    nafdac = (seed.nafdac_number or "").strip().lower()
    if nafdac:
        found_nafdacs = [n.lower() for n in source.identifiers.get("nafdac_numbers", [])]
        if nafdac in title_low or nafdac in content_low or nafdac in found_nafdacs:
            return True, "nafdac_match"

    gtin = (seed.gtin or "").strip().lower()
    if gtin:
        found_gtins = [g.lower() for g in source.identifiers.get("gtins", [])]
        if gtin in title_low or gtin in content_low or gtin in found_gtins:
            return True, "gtin_match"

    tokens = _brand_tokens(seed.product_name)
    if not tokens:
        if nafdac or gtin:
            return False, "off_topic_no_identifier_mention"
        return True, "no_filter_tokens"

    if any(re.search(rf"\b{re.escape(tok)}\b", title_low) for tok in tokens):
        return True, "title_match"

    max_mentions = max(
        len(re.findall(rf"\b{re.escape(tok)}\b", content_low)) for tok in tokens
    )
    if max_mentions == 0:
        return False, "off_topic_no_product_mention"
    if max_mentions == 1 and len(source.content) > 2500:
        return False, "passing_mention_only"
    return True, "content_match"


def discover(
    seed: DiscoverySeed,
    *,
    greenbook: Optional[GreenbookStore] = None,
    db_path: Optional[Path | str] = None,
    trusted_store=None,
    max_urls: int = 8,
    max_results_per_query: int = 6,
    max_stage: Optional[str] = None,
    budget: Optional[str] = None,
    include_smpc_pdf: bool = True,
    include_image_search: bool = True,
    max_image_hits: int = 10,
    max_workers: Optional[int] = None,
    stagger_s: Optional[float] = None,
    search_fn: Optional[Callable] = None,
    image_search_fn: Optional[Callable] = None,
    extract_fn: Optional[Callable] = None,
) -> DiscoveryBundle:
    """Run local Greenbook lookup, parallel multi-angle + image search, trust ranking, and extraction."""
    t0 = time.monotonic()

    # 1. Local Greenbook lookup & seed enrichment
    gb_matches = _lookup_greenbook(seed, greenbook=greenbook, db_path=db_path)
    enriched_seed = enrich_seed_from_greenbook(seed, gb_matches)

    smpc_url = ""
    if include_smpc_pdf and gb_matches:
        candidate_smpc = str(gb_matches[0].get("smpc_url") or "").strip()
        if candidate_smpc.startswith(("http://", "https://")):
            smpc_url = candidate_smpc

    queries = build_queries(enriched_seed, has_smpc_pdf=bool(smpc_url))
    trusted_domains: List[str] = []
    if trusted_store is not None:
        trusted_domains = trusted_store.get_verified_image_domains(
            country=enriched_seed.country or "NG",
            manufacturer_hint=enriched_seed.manufacturer,
        )
        queries.extend(
            build_trusted_pharmacy_web_queries(
                enriched_seed,
                trusted_domains=trusted_domains,
            )
        )

    # 2. Parallel multi-angle web search (+ optional direct packaging image search)
    effective_search = search_fn or search_report
    run_img_search = include_image_search and (image_search_fn is not None or search_fn is None)
    effective_img_search = image_search_fn or search_images
    workers, stagger = _default_search_settings(search_fn, max_workers, stagger_s)

    all_hits: List[Dict] = []
    search_errors: List[Dict] = []
    image_candidates: List[Dict] = []
    country = enriched_seed.country.lower() if enriched_seed.country else None

    img_queries: List[str] = []
    if run_img_search:
        gb_top = gb_matches[0] if gb_matches else None
        if trusted_store is not None:
            img_queries = build_trusted_image_queries(
                enriched_seed,
                trusted_domains=trusted_domains,
                greenbook_match=gb_top,
            )
        else:
            img_queries = [build_image_query(enriched_seed, greenbook_match=gb_top)]

    total_tasks = len(queries) + len(img_queries)
    if total_tasks > 0:
        with ThreadPoolExecutor(max_workers=min(workers, total_tasks)) as pool:
            text_futures = [
                pool.submit(
                    _run_one_query,
                    q,
                    delay_s=idx * stagger,
                    max_results=max_results_per_query,
                    country=country,
                    search_fn=effective_search,
                )
                for idx, q in enumerate(queries)
            ]
            img_futures = [
                pool.submit(
                    _run_image_search,
                    iq,
                    delay_s=(len(queries) + idx) * stagger,
                    max_results=max_image_hits,
                    country=country,
                    manufacturer_hint=enriched_seed.manufacturer,
                    image_search_fn=effective_img_search,
                    trusted_store=trusted_store,
                )
                for idx, iq in enumerate(img_queries)
            ]

            for fut in as_completed(text_futures):
                q_obj, hits, err_entry = fut.result()
                if err_entry:
                    search_errors.append(err_entry)
                for h in hits:
                    hit_copy = dict(h)
                    hit_copy["angle"] = q_obj.angle
                    all_hits.append(hit_copy)

            seen_img_urls: set[str] = set()
            for img_fut in img_futures:
                batch_imgs, img_err = img_fut.result()
                if img_err:
                    search_errors.append(img_err)
                for item in batch_imgs:
                    u = item.get("url")
                    if u and u not in seen_img_urls:
                        seen_img_urls.add(u)
                        image_candidates.append(item)

    # 3. Rank & select web candidates (+ official SmPC PDF + guaranteed trusted pharmacy product pages)
    web_budget = max(1, max_urls - 1) if smpc_url and max_urls > 1 else max_urls
    selected = rank_and_select(
        all_hits,
        manufacturer_hint=enriched_seed.manufacturer,
        max_urls=web_budget,
        max_per_domain=2,
    )

    if trusted_store is not None:
        trusted_hits = [h for h in all_hits if h.get("angle") == "trusted_pharmacy"]
        if trusted_hits:
            trusted_selected = rank_and_select(
                trusted_hits,
                manufacturer_hint=enriched_seed.manufacturer,
                max_urls=4,
                max_per_domain=1,
            )
            existing_canons = {c.canonical_url for c in selected}
            for tc in trusted_selected:
                if tc.canonical_url not in existing_canons:
                    selected.append(tc)
                    existing_canons.add(tc.canonical_url)

    meta_by_canon: Dict[str, Dict] = {}
    fetch_urls: List[str] = []

    if smpc_url:
        smpc_canon = canonicalize_url(smpc_url)
        tier, label = classify_url(smpc_url, manufacturer_hint=enriched_seed.manufacturer)
        fetch_urls.append(smpc_url)
        meta_by_canon[smpc_canon or smpc_url] = {
            "url": smpc_url,
            "canonical_url": smpc_canon or smpc_url,
            "domain": extract_domain(smpc_url),
            "trust_tier": tier,
            "trust_label": label,
            "angles": ["greenbook_smpc"],
            "is_smpc_pdf": True,
        }

    for cand in selected:
        if cand.canonical_url in meta_by_canon:
            continue
        fetch_urls.append(cand.url)
        meta_by_canon[cand.canonical_url] = {
            "url": cand.url,
            "canonical_url": cand.canonical_url,
            "domain": cand.domain,
            "trust_tier": cand.trust_tier,
            "trust_label": cand.trust_label,
            "angles": cand.angles,
            "is_smpc_pdf": False,
        }

    # 4. Batch parallel extraction via forked scraper
    extracted_docs: List[Dict] = []
    failed_sources: List[Dict] = []
    if fetch_urls:
        effective_extract = extract_fn or extract_report
        report = effective_extract(
            fetch_urls,
            fmt="text",
            max_stage=max_stage,
            budget=budget,
            structured=True,
        )
        extracted_docs = list(report.get("results") or [])
        failed_sources = list(report.get("failed") or [])

    sources: List[DiscoveredSource] = []
    dropped_sources: List[Dict] = []
    for doc in extracted_docs:
        doc_url = str(doc.get("url") or "")
        doc_canon = canonicalize_url(doc_url)
        meta = meta_by_canon.get(doc_canon) or meta_by_canon.get(doc_url)
        if meta is None:
            tier, label = classify_url(doc_url, manufacturer_hint=enriched_seed.manufacturer)
            meta = {
                "url": doc_url,
                "canonical_url": doc_canon or doc_url,
                "domain": extract_domain(doc_url),
                "trust_tier": tier,
                "trust_label": label,
                "angles": [],
                "is_smpc_pdf": bool(smpc_url and doc_url == smpc_url),
            }
        src = DiscoveredSource(
            url=doc_url or meta["url"],
            canonical_url=meta["canonical_url"],
            domain=meta["domain"],
            trust_tier=meta["trust_tier"],
            trust_label=meta["trust_label"],
            angles=list(meta["angles"]),
            title=str(doc.get("title") or ""),
            content=str(doc.get("content") or ""),
            stage=str(doc.get("stage") or "static"),
            product=doc.get("product"),
            images=list(doc.get("images") or []),
            identifiers=dict(doc.get("identifiers") or {}),
            is_smpc_pdf=bool(meta["is_smpc_pdf"]),
        )
        keep, reason = _check_relevance(src, enriched_seed)
        if keep:
            sources.append(src)
        else:
            dropped_sources.append(
                {
                    "url": src.url,
                    "domain": src.domain,
                    "trust_tier": src.trust_tier,
                    "title": src.title,
                    "reason": reason,
                }
            )

    sources.sort(key=lambda s: (s.trust_tier, not s.is_smpc_pdf))

    stats = {
        "greenbook_match_count": len(gb_matches),
        "queries_executed": len(queries),
        "raw_hits": len(all_hits),
        "urls_selected": len(fetch_urls),
        "pages_extracted": len(sources),
        "pages_dropped_off_topic": len(dropped_sources),
        "pages_failed": len(failed_sources),
        "image_candidates_found": len(image_candidates),
        "elapsed_s": round(time.monotonic() - t0, 3),
    }

    return DiscoveryBundle(
        seed=seed,
        enriched_seed=enriched_seed,
        greenbook_matches=gb_matches,
        queries_run=queries,
        search_errors=search_errors,
        sources=sources,
        dropped_sources=dropped_sources,
        failed_sources=failed_sources,
        image_candidates=image_candidates,
        stats=stats,
    )

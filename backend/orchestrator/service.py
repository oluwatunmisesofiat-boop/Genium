"""Concurrent GENIE AI Orchestrator: parallel CV + local verification & non-blocking async discovery."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from discovery.models import DiscoverySeed
from intelligence.synthesizer import compute_reference_level
from registry.greenbook import GreenbookStore, default_db_path
from registry.trusted_sources import TrustedSourceStore, default_genie_db_path
from verification.engine import (
    DEFAULT_PRODUCTS_DIR,
    VerificationEngine,
    _resolve_product,
    load_saved_dossiers,
)
from verification.schemas import ScanInput, VerificationResult
from verification.signals.visual import _approved_versions

from .jobs import DiscoveryJobRunner, JobStore
from .scans import ScanStore


class OrchestratedVerificationResponse(BaseModel):
    """Full orchestrated verification response including scan_id, timing, and optional async discovery job."""

    scan_id: Optional[str] = None
    scan_json_path: Optional[str] = None
    result: VerificationResult
    execution_time_ms: float
    discovery_job_id: Optional[str] = None
    discovery_job_status: Optional[str] = None
    discovery_deduplicated: bool = False


class GenieOrchestrator:
    """Coordinates simultaneous CV analysis, 6 local verification signals, and background discovery."""

    def __init__(
        self,
        *,
        products_dir: Path | str = DEFAULT_PRODUCTS_DIR,
        genie_db_path: Optional[Path | str] = None,
        greenbook: Any = None,
        load_default_greenbook: bool = True,
        dossiers: Optional[List[Dict[str, Any]]] = None,
        cv_adapter: Optional[Any] = None,
        discovery_worker_fn: Optional[Callable[[DiscoverySeed], Dict[str, Any]]] = None,
        max_discovery_workers: int = 2,
    ) -> None:
        self.products_dir = Path(products_dir)
        self.genie_db_path = (
            Path(genie_db_path) if genie_db_path is not None else default_genie_db_path()
        )
        if greenbook is not None:
            self.greenbook: Optional[GreenbookStore] = greenbook
        elif load_default_greenbook and default_db_path().is_file():
            self.greenbook = GreenbookStore(default_db_path())
        else:
            self.greenbook = None

        self.trusted_store = TrustedSourceStore(self.genie_db_path)
        self.trusted_store.ensure_seeded()

        self.job_store = JobStore(self.genie_db_path)
        self.scans_dir = self.genie_db_path.parent / "scans"
        self.scan_store = ScanStore(self.genie_db_path, scans_dir=self.scans_dir)
        self.job_runner = DiscoveryJobRunner(
            job_store=self.job_store,
            products_dir=self.products_dir,
            max_workers=max_discovery_workers,
            worker_fn=discovery_worker_fn,
        )
        self._explicit_dossiers = dossiers
        self._cv_adapter = cv_adapter
        self._engine = VerificationEngine(
            products_dir=self.products_dir,
            dossiers=dossiers,
            greenbook=self.greenbook,
            cv_adapter=cv_adapter,
        )
        self._cv_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="genie-cv")

    def _current_dossiers(self) -> List[Dict[str, Any]]:
        if self._explicit_dossiers is not None:
            return self._explicit_dossiers
        return load_saved_dossiers(self.products_dir)

    def verify_scan(
        self,
        scan: ScanInput,
        *,
        auto_discover: bool = True,
    ) -> OrchestratedVerificationResponse:
        """Run CV analysis + deterministic verification concurrently and enqueue async discovery if needed."""
        t0 = time.perf_counter()
        all_dossiers = self._current_dossiers()
        resolved, _rec = _resolve_product(scan, all_dossiers, self.greenbook)

        # If CV adapter is configured and approved references exist, run CV concurrently with local checks
        effective_scan = scan
        if (
            resolved is not None
            and scan.visual_match is None
            and self._cv_adapter is not None
        ):
            approved = _approved_versions(resolved)
            if approved:
                cv_future = self._cv_pool.submit(self._cv_adapter.analyze, scan, approved)
                # While CV runs in background thread, pre-warm Greenbook lookup if needed
                if self.greenbook is not None and scan.nafdac_number:
                    self.greenbook.lookup_by_nafdac(scan.nafdac_number)
                cv_match = cv_future.result()
                if cv_match is not None:
                    effective_scan = scan.model_copy(update={"visual_match": cv_match})

        result = self._engine.verify(effective_scan)

        job_id: Optional[str] = None
        job_status: Optional[str] = None
        deduped = False

        if auto_discover and result.recommend_discovery:
            seed = self._seed_from_scan(scan, result)
            if seed is not None:
                job, deduped = self.job_runner.submit(seed)
                job_id = job["job_id"]
                job_status = job["status"]

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        scan_id = ScanStore.new_scan_id()
        response = OrchestratedVerificationResponse(
            scan_id=scan_id,
            result=result,
            execution_time_ms=elapsed_ms,
            discovery_job_id=job_id,
            discovery_job_status=job_status,
            discovery_deduplicated=deduped,
        )
        mp = result.matched_product or {}
        json_path = self.scan_store.record_scan(
            scan_id=scan_id,
            source_endpoint="verify",
            product_name=mp.get("brand_name") or scan.product_name_hint,
            nafdac_number=mp.get("nafdac_number") or scan.nafdac_number,
            verdict=result.verdict,
            ml_infra_verdict=result.ml_infra_verdict,
            confidence=result.confidence,
            coverage=result.coverage,
            reference_level=result.reference_level,
            summary=result.summary,
            limitations=list(result.limitations),
            counterfeit_tell_hits=[h.model_dump() for h in result.counterfeit_tell_hits],
            discovery_job_id=job_id,
            execution_time_ms=elapsed_ms,
            payload=response.model_dump(mode="json"),
        )
        return response.model_copy(update={"scan_json_path": json_path})

    @staticmethod
    def _seed_from_scan(scan: ScanInput, result: VerificationResult) -> Optional[DiscoverySeed]:
        mp = result.matched_product or {}
        name = mp.get("brand_name") or scan.product_name_hint
        nafdac = mp.get("nafdac_number") or scan.nafdac_number
        mfr = mp.get("manufacturer")
        gtin = scan.gtin
        if not any((name, nafdac, gtin)):
            return None
        return DiscoverySeed(
            product_name=name,
            manufacturer=mfr,
            nafdac_number=nafdac,
            gtin=gtin,
        )

    def trigger_discovery(self, seed: DiscoverySeed) -> Dict[str, Any]:
        job, deduped = self.job_runner.submit(seed)
        return {"job": job, "deduplicated": deduped}

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.job_store.get_job(job_id)

    def list_jobs(self, *, limit: int = 20, status: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.job_store.list_jobs(limit=limit, status=status)

    def get_scan(self, scan_id: str) -> Optional[Dict[str, Any]]:
        return self.scan_store.get_scan(scan_id)

    def list_scans(
        self,
        *,
        limit: int = 20,
        verdict: Optional[str] = None,
        nafdac_number: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self.scan_store.list_scans(limit=limit, verdict=verdict, nafdac_number=nafdac_number)

    def list_products(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for d in self._current_dossiers():
            items.append(
                {
                    "slug": d.get("_slug"),
                    "brand_name": d.get("brand_name"),
                    "nafdac_number": d.get("nafdac_number"),
                    "manufacturer": d.get("manufacturer"),
                    "registration_status": d.get("registration_status"),
                    "dossier_status": d.get("dossier_status", "staging"),
                    "reference_level": compute_reference_level(d.get("packaging_versions") or []),
                    "packaging_versions_count": len(d.get("packaging_versions") or []),
                }
            )
        return items

    def get_product(self, slug: str) -> Optional[Dict[str, Any]]:
        safe_slug = Path(slug).name
        if safe_slug != slug or not safe_slug:
            return None
        target = self.products_dir / safe_slug / "product.json"
        if not target.is_file():
            return None
        return json.loads(target.read_text(encoding="utf-8"))

    def health(self) -> Dict[str, Any]:
        gb_count = self.greenbook.count() if self.greenbook is not None else 0
        trusted_all = self.trusted_store.list_sources(active_only=False)
        verified_trusted = sum(1 for r in trusted_all if r.get("trust_status") == "verified")
        return {
            "status": "ok",
            "greenbook_records": gb_count,
            "trusted_sources_verified": verified_trusted,
            "trusted_sources_total": len(trusted_all),
            "cached_products": len(self._current_dossiers()),
            "active_discovery_jobs": self.job_runner.active_count(),
        }

    def shutdown(self, *, wait: bool = True) -> None:
        self.job_runner.shutdown(wait=wait)
        self._cv_pool.shutdown(wait=wait)

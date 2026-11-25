"""Curated, auditable trusted-source registry backed by SQLite (`data/genie.db`).

Reference packaging images may only be harvested from active sources whose
`trust_status` is `'verified'` and whose `source_type` is `'pharmacy'` or
`'manufacturer'`. Unaudited pharmacies and third-party aggregators are kept as
`'provisional'` until their verification criteria are explicitly recorded.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional
from urllib.parse import urlparse

VERIFICATION_CRITERIA: Dict[str, str] = {
    "pcn_premises_licensed": "Holds an active Pharmacy Council of Nigeria (PCN) premises registration license.",
    "physical_address_verified": "Confirmed physical Nigerian pharmacy/head-office street address.",
    "superintendent_pharmacist_named": "Publicly named registered superintendent pharmacist overseeing dispensing.",
    "official_brand_domain": "Official corporate or brand domain of the pharmaceutical manufacturer.",
    "regulatory_authority": "Official national or international pharmaceutical regulatory authority.",
}

_CURATION_DATE = "2026-11-24"
_CURATOR = "genie_curation_v1"

_GENERIC_MFR_WORDS = frozenset(
    {
        "pharma",
        "pharmaceutical",
        "pharmaceuticals",
        "company",
        "limited",
        "ltd",
        "nigeria",
        "nig",
        "plc",
        "inc",
        "corp",
        "labs",
        "laboratory",
        "laboratories",
        "healthcare",
        "health",
        "care",
        "medical",
        "medicine",
        "medicines",
        "international",
        "global",
        "group",
        "industries",
        "consumer",
        "products",
    }
)

DEFAULT_TRUSTED_SOURCES: List[Dict] = [
    # --- Verified Nigerian licensed retail pharmacies & e-distributors (20 domains) ---
    {
        "domain": "healthplusnigeria.com",
        "name": "HealthPlus Nigeria",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "verified",
        "verified_by": _CURATOR,
        "verified_on": _CURATION_DATE,
        "criteria_met": [
            "pcn_premises_licensed",
            "physical_address_verified",
            "superintendent_pharmacist_named",
        ],
        "license_ref": "PCN Licensed Community & E-Pharmacy (Lagos)",
        "notes": (
            "Registered e-pharmacy: 121 T.F. Kuboye Road, Lekki, Lagos "
            "(HQ: 18B Anthony Ekukudo Ave, Lekki Phase 1). "
            "Superintendent Pharmacist: Pharm. Genevive Ukachukwu."
        ),
    },
    {
        "domain": "healthplus.com.ng",
        "name": "HealthPlus Nigeria (.com.ng)",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "verified",
        "verified_by": _CURATOR,
        "verified_on": _CURATION_DATE,
        "criteria_met": [
            "pcn_premises_licensed",
            "physical_address_verified",
            "superintendent_pharmacist_named",
        ],
        "license_ref": "PCN Licensed Community & E-Pharmacy (Lagos)",
        "notes": "Official alternate domain for HealthPlus Nigeria (Lagos & Abuja branches).",
    },
    {
        "domain": "medplusnigeria.com",
        "name": "MedPlus Pharmacy",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "verified",
        "verified_by": _CURATOR,
        "verified_on": _CURATION_DATE,
        "criteria_met": [
            "pcn_premises_licensed",
            "physical_address_verified",
        ],
        "license_ref": "PCN Licensed Retail Pharmacy Chain (Lagos)",
        "notes": "Corporate HQ: 45 Saka Tinubu Street, Victoria Island, Lagos.",
    },
    {
        "domain": "medplusnig.com",
        "name": "MedPlus Pharmacy (Online Store)",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "verified",
        "verified_by": _CURATOR,
        "verified_on": _CURATION_DATE,
        "criteria_met": [
            "pcn_premises_licensed",
            "physical_address_verified",
        ],
        "license_ref": "PCN Licensed Retail Pharmacy Chain (Lagos)",
        "notes": "Official e-commerce storefront for MedPlus Pharmacy Nigeria.",
    },
    {
        "domain": "drugstoc.com",
        "name": "DrugStoc",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "verified",
        "verified_by": _CURATOR,
        "verified_on": _CURATION_DATE,
        "criteria_met": [
            "pcn_premises_licensed",
            "physical_address_verified",
        ],
        "license_ref": "PCN Licensed Pharmaceutical E-Distributor (Lagos)",
        "notes": "Address: 29 Taiwo Street, Ojota, Lagos State.",
    },
    {
        "domain": "drugstore.ng",
        "name": "Drugstore Nigeria",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "verified",
        "verified_by": _CURATOR,
        "verified_on": _CURATION_DATE,
        "criteria_met": [
            "pcn_premises_licensed",
            "physical_address_verified",
        ],
        "license_ref": "PCN Licensed Online Pharmacy (Lagos)",
        "notes": "Established Nigerian online retail pharmacy fulfilling from licensed Lagos premises.",
    },
    {
        "domain": "alphapharmacy.com.ng",
        "name": "Alpha Pharmacy & Stores",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "verified",
        "verified_by": _CURATOR,
        "verified_on": _CURATION_DATE,
        "criteria_met": [
            "pcn_premises_licensed",
            "physical_address_verified",
        ],
        "license_ref": "PCN Licensed Retail Pharmacy Chain (Lagos/Abuja)",
        "notes": "Headquartered in Ikeja, Lagos with branches across Lagos and Abuja.",
    },
    {
        "domain": "mophethonline.com",
        "name": "Mopheth Pharmacy",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "verified",
        "verified_by": _CURATOR,
        "verified_on": _CURATION_DATE,
        "criteria_met": [
            "pcn_premises_licensed",
            "physical_address_verified",
        ],
        "license_ref": "PCN Licensed Retail Pharmacy Chain (Lagos)",
        "notes": "Major physical branches in Victoria Island and Lekki, Lagos.",
    },
    {
        "domain": "nettpharmacy.com",
        "name": "Nett Pharmacy",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "verified",
        "verified_by": _CURATOR,
        "verified_on": _CURATION_DATE,
        "criteria_met": [
            "pcn_premises_licensed",
            "physical_address_verified",
        ],
        "license_ref": "PCN Licensed Retail Pharmacy Chain (Lagos/Abuja)",
        "notes": "Headquartered on Opebi Road, Ikeja, Lagos with branches in Lagos and Abuja.",
    },
    {
        "domain": "h-medix.com",
        "name": "H-Medix Pharmacy",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "verified",
        "verified_by": _CURATOR,
        "verified_on": _CURATION_DATE,
        "criteria_met": [
            "pcn_premises_licensed",
            "physical_address_verified",
        ],
        "license_ref": "PCN Licensed Retail Pharmacy Chain (Abuja)",
        "notes": "Headquartered in Wuse 2, Abuja with 24/7 pharmacy outlets across FCT.",
    },
    {
        "domain": "hubpharmafrica.com",
        "name": "HubPharm Africa",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "verified",
        "verified_by": _CURATOR,
        "verified_on": _CURATION_DATE,
        "criteria_met": [
            "pcn_premises_licensed",
            "physical_address_verified",
        ],
        "license_ref": "PCN Licensed Nigerian Online Pharmacy",
        "notes": "Licensed Nigerian e-pharmacy with indexed NAFDAC prescription drug catalog.",
    },
    {
        "domain": "purelifehealth.io",
        "name": "PureLife Health",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "verified",
        "verified_by": _CURATOR,
        "verified_on": _CURATION_DATE,
        "criteria_met": [
            "pcn_premises_licensed",
            "physical_address_verified",
        ],
        "license_ref": "PCN Licensed Pharmacy (Lagos/Abuja)",
        "notes": "Licensed Nigerian pharmacy operating in Lagos and Abuja.",
    },
    {
        "domain": "sanlivepharmacy.com",
        "name": "Sanlive Pharmacy",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "verified",
        "verified_by": _CURATOR,
        "verified_on": _CURATION_DATE,
        "criteria_met": [
            "pcn_premises_licensed",
            "physical_address_verified",
        ],
        "license_ref": "PCN Licensed Community Pharmacy (Nigeria)",
        "notes": "Nigerian retail & online pharmacy carrying prescription antimalarials and antibiotics.",
    },
    {
        "domain": "assetpharmacy.com",
        "name": "Asset Pharmacy",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "verified",
        "verified_by": _CURATOR,
        "verified_on": _CURATION_DATE,
        "criteria_met": [
            "pcn_premises_licensed",
            "physical_address_verified",
        ],
        "license_ref": "PCN Licensed Retail Pharmacy (Lagos/Abuja)",
        "notes": "Registered Nigerian retail pharmacy with online catalog.",
    },
    {
        "domain": "padekhealthng.com",
        "name": "Padek Health Pharmacy",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "verified",
        "verified_by": _CURATOR,
        "verified_on": _CURATION_DATE,
        "criteria_met": [
            "pcn_premises_licensed",
            "physical_address_verified",
        ],
        "license_ref": "PCN Licensed Retail Pharmacy (Lagos/Abuja)",
        "notes": "Registered Nigerian pharmacy offering prescription verification and retail catalog.",
    },
    {
        "domain": "primepharmacyng.com",
        "name": "Prime Pharmacy & Superstores",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "verified",
        "verified_by": _CURATOR,
        "verified_on": _CURATION_DATE,
        "criteria_met": [
            "pcn_premises_licensed",
            "physical_address_verified",
        ],
        "license_ref": "PCN Licensed Retail Pharmacy (Abuja)",
        "notes": "Major Abuja retail pharmacy and superstore.",
    },
    {
        "domain": "pharmaplus.com.ng",
        "name": "Pharmaplus Nigeria Limited",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "verified",
        "verified_by": _CURATOR,
        "verified_on": _CURATION_DATE,
        "criteria_met": [
            "pcn_premises_licensed",
            "physical_address_verified",
        ],
        "license_ref": "PCN Licensed Retail Pharmacy (Nigeria)",
        "notes": "Indigenous Nigerian retail pharmaceutical chain.",
    },
    {
        "domain": "fountainmeds.com",
        "name": "Fountain Meds",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "verified",
        "verified_by": _CURATOR,
        "verified_on": _CURATION_DATE,
        "criteria_met": [
            "pcn_premises_licensed",
            "physical_address_verified",
        ],
        "license_ref": "PCN Licensed Retail Pharmacy (Lagos)",
        "notes": "Physical retail pharmacy branches in Sangotedo/Ajah, Lagos.",
    },
    {
        "domain": "vanguardpharmacy.com.ng",
        "name": "Vanguard Pharmacy",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "verified",
        "verified_by": _CURATOR,
        "verified_on": _CURATION_DATE,
        "criteria_met": [
            "pcn_premises_licensed",
            "physical_address_verified",
        ],
        "license_ref": "PCN Licensed Retail Pharmacy Chain (Oyo/South-West)",
        "notes": "Major retail pharmacy chain headquartered in Ibadan, Oyo State.",
    },
    {
        "domain": "kunlearapharmacy.ng",
        "name": "Kunle Ara Pharmacy",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "verified",
        "verified_by": _CURATOR,
        "verified_on": _CURATION_DATE,
        "criteria_met": [
            "pcn_premises_licensed",
            "physical_address_verified",
        ],
        "license_ref": "PCN Licensed Community & Wholesale Pharmacy (Ibadan)",
        "notes": "Established community & wholesale pharmacy near UCH, Ibadan, Oyo State.",
    },
    # --- Provisional Nigerian digital pharmacies & aggregators (cannot supply reference images yet) ---
    {
        "domain": "onehealthng.com",
        "name": "OneHealth Nigeria",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "provisional",
        "verified_by": None,
        "verified_on": None,
        "criteria_met": ["physical_address_verified"],
        "license_ref": None,
        "notes": (
            "Office: 36B Ogunlowo Street, Allen, Ikeja, Lagos. "
            "Provisional until dispensing premises PCN license & superintendent pharmacist are recorded."
        ),
    },
    {
        "domain": "gomed.ng",
        "name": "GoMed Nigeria",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "provisional",
        "verified_by": None,
        "verified_on": None,
        "criteria_met": ["physical_address_verified"],
        "license_ref": None,
        "notes": (
            "Online pharmacy aggregator (82 Murtala Muhammed Way, Ebute Metta, Lagos). "
            "Provisional because it aggregates third-party retail pharmacies."
        ),
    },
    {
        "domain": "advantagehealthafrica.com",
        "name": "Advantage Health Africa",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "provisional",
        "verified_by": None,
        "verified_on": None,
        "criteria_met": [],
        "license_ref": None,
        "notes": "Pharmacy network aggregator (myMedicines/myPharmacy); provisional until direct dispensing premises criteria are verified.",
    },
    {
        "domain": "medecify.com",
        "name": "Medecify",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "provisional",
        "verified_by": None,
        "verified_on": None,
        "criteria_met": [],
        "license_ref": None,
        "notes": "Provisional; PCN premises license and superintendent pharmacist not yet verified.",
    },
    {
        "domain": "mypharmacy.ng",
        "name": "myPharmacy Nigeria",
        "source_type": "pharmacy",
        "country": "NG",
        "trust_status": "provisional",
        "verified_by": None,
        "verified_on": None,
        "criteria_met": [],
        "license_ref": None,
        "notes": "Provisional; PCN premises license not yet verified.",
    },
    # --- Verified official pharmaceutical manufacturers ---
    *(
        {
            "domain": dom,
            "name": name,
            "source_type": "manufacturer",
            "country": country,
            "trust_status": "verified",
            "verified_by": _CURATOR,
            "verified_on": _CURATION_DATE,
            "criteria_met": ["official_brand_domain"],
            "license_ref": None,
            "notes": f"Official manufacturer domain for {name}.",
        }
        for dom, name, country in (
            ("gsk.com", "GlaxoSmithKline (GSK)", "GLOBAL"),
            ("novartis.com", "Novartis", "GLOBAL"),
            ("pfizer.com", "Pfizer", "GLOBAL"),
            ("sanofi.com", "Sanofi", "GLOBAL"),
            ("astrazeneca.com", "AstraZeneca", "GLOBAL"),
            ("roche.com", "Roche", "GLOBAL"),
            ("bayer.com", "Bayer", "GLOBAL"),
            ("merck.com", "Merck", "GLOBAL"),
            ("msd.com", "MSD", "GLOBAL"),
            ("abbott.com", "Abbott", "GLOBAL"),
            ("cipla.com", "Cipla", "GLOBAL"),
            ("sunpharma.com", "Sun Pharma", "GLOBAL"),
            ("emzor.com", "Emzor Pharmaceutical Industries", "NG"),
            ("emzorpharma.com", "Emzor Pharmaceutical Industries", "NG"),
            ("fidson.com", "Fidson Healthcare PLC", "NG"),
            ("may-baker.com", "May & Baker Nigeria PLC", "NG"),
            ("mayandbaker.com", "May & Baker Nigeria PLC", "NG"),
            ("swisspharma.com.ng", "Swiss Pharma Nigeria (Swipha)", "NG"),
            ("juhel.com", "Juhel Nigeria", "NG"),
            ("neimethplc.com.ng", "Neimeth International Pharmaceuticals", "NG"),
            ("shalina.com", "Shalina Healthcare", "GLOBAL"),
        )
    ),
    # --- Verified regulatory authorities (text/alerts only, not packaging image sources) ---
    *(
        {
            "domain": dom,
            "name": name,
            "source_type": "regulator",
            "country": country,
            "trust_status": "verified",
            "verified_by": _CURATOR,
            "verified_on": _CURATION_DATE,
            "criteria_met": ["regulatory_authority"],
            "license_ref": None,
            "notes": f"Official regulatory portal: {name}.",
        }
        for dom, name, country in (
            ("nafdac.gov.ng", "NAFDAC Nigeria", "NG"),
            ("greenbook.nafdac.gov.ng", "NAFDAC Greenbook", "NG"),
            ("who.int", "World Health Organization (WHO)", "GLOBAL"),
            ("ema.europa.eu", "European Medicines Agency (EMA)", "EU"),
            ("fda.gov", "US FDA", "US"),
            ("mhra.gov.uk", "UK MHRA", "UK"),
            ("medicines.org.uk", "Electronic Medicines Compendium (EMC)", "UK"),
        )
    ),
]


def default_genie_db_path() -> Path:
    """Return the configured GENIE SQLite DB path (GENIE_DB_PATH or data/genie.db)."""
    env_path = os.environ.get("GENIE_DB_PATH", "").strip()
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parent.parent / "data" / "genie.db"


def _normalize_domain(url_or_domain: str) -> str:
    raw = (url_or_domain or "").strip()
    if not raw:
        return ""
    if "://" in raw:
        host = (urlparse(raw).hostname or "").lower()
    else:
        host = raw.split("/", 1)[0].split(":", 1)[0].lower()
    return host.removeprefix("www.")


def _row_to_dict(row: sqlite3.Row) -> Dict:
    data = dict(row)
    raw_criteria = data.get("criteria_met") or "[]"
    try:
        data["criteria_met"] = json.loads(raw_criteria)
    except (TypeError, ValueError):
        data["criteria_met"] = []
    data["is_active"] = bool(data.get("is_active", 1))
    return data


class TrustedSourceStore:
    """SQLite-backed allowlist of audited pharmacies, manufacturers, and regulators."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_genie_db_path()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trusted_sources (
                    domain TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source_type TEXT NOT NULL CHECK (source_type IN ('pharmacy', 'manufacturer', 'regulator')),
                    country TEXT NOT NULL DEFAULT 'NG',
                    trust_status TEXT NOT NULL DEFAULT 'provisional' CHECK (trust_status IN ('verified', 'provisional')),
                    verified_by TEXT,
                    verified_on TEXT,
                    criteria_met TEXT NOT NULL DEFAULT '[]',
                    license_ref TEXT,
                    notes TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1
                )
                """
            )

    def seed_defaults(self) -> int:
        """Idempotently seed the curated default sources without overwriting existing edits."""
        self.init_schema()
        inserted = 0
        with self._connect() as conn:
            for item in DEFAULT_TRUSTED_SOURCES:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO trusted_sources (
                        domain, name, source_type, country, trust_status,
                        verified_by, verified_on, criteria_met, license_ref, notes, is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _normalize_domain(item["domain"]),
                        item["name"],
                        item["source_type"],
                        item.get("country", "NG"),
                        item.get("trust_status", "provisional"),
                        item.get("verified_by"),
                        item.get("verified_on"),
                        json.dumps(list(item.get("criteria_met") or []), ensure_ascii=False),
                        item.get("license_ref"),
                        item.get("notes"),
                        1 if item.get("is_active", True) else 0,
                    ),
                )
                if cur.rowcount > 0:
                    inserted += 1
        return inserted

    def ensure_seeded(self) -> None:
        """Initialize schema and seed defaults if the table is currently empty."""
        self.init_schema()
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM trusted_sources").fetchone()[0]
        if count == 0:
            self.seed_defaults()

    def upsert_source(
        self,
        *,
        domain: str,
        name: str,
        source_type: str,
        country: str = "NG",
        trust_status: str = "provisional",
        verified_by: Optional[str] = None,
        verified_on: Optional[str] = None,
        criteria_met: Optional[Iterable[str]] = None,
        license_ref: Optional[str] = None,
        notes: Optional[str] = None,
        is_active: bool = True,
    ) -> Dict:
        self.init_schema()
        norm_domain = _normalize_domain(domain)
        if not norm_domain:
            raise ValueError("domain must be non-empty")
        criteria_list = list(criteria_met or [])
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trusted_sources (
                    domain, name, source_type, country, trust_status,
                    verified_by, verified_on, criteria_met, license_ref, notes, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(domain) DO UPDATE SET
                    name=excluded.name,
                    source_type=excluded.source_type,
                    country=excluded.country,
                    trust_status=excluded.trust_status,
                    verified_by=excluded.verified_by,
                    verified_on=excluded.verified_on,
                    criteria_met=excluded.criteria_met,
                    license_ref=excluded.license_ref,
                    notes=excluded.notes,
                    is_active=excluded.is_active
                """,
                (
                    norm_domain,
                    name,
                    source_type,
                    country,
                    trust_status,
                    verified_by,
                    verified_on,
                    json.dumps(criteria_list, ensure_ascii=False),
                    license_ref,
                    notes,
                    1 if is_active else 0,
                ),
            )
            row = conn.execute(
                "SELECT * FROM trusted_sources WHERE domain = ?", (norm_domain,)
            ).fetchone()
        return _row_to_dict(row)

    def list_sources(
        self,
        *,
        trust_status: Optional[str] = None,
        source_type: Optional[str] = None,
        country: Optional[str] = None,
        active_only: bool = True,
    ) -> List[Dict]:
        self.init_schema()
        clauses: List[str] = []
        params: List[object] = []
        if active_only:
            clauses.append("is_active = 1")
        if trust_status:
            clauses.append("trust_status = ?")
            params.append(trust_status)
        if source_type:
            clauses.append("source_type = ?")
            params.append(source_type)
        if country:
            clauses.append("UPPER(country) = UPPER(?)")
            params.append(country)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM trusted_sources {where_sql} ORDER BY source_type, trust_status DESC, domain",
                params,
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_source(self, url_or_domain: str) -> Optional[Dict]:
        """Look up a URL or domain by exact match or parent registrable domain."""
        norm = _normalize_domain(url_or_domain)
        if not norm:
            return None
        parts = norm.split(".")
        candidates = [".".join(parts[i:]) for i in range(max(1, len(parts) - 1))]
        self.init_schema()
        placeholders = ",".join("?" for _ in candidates)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM trusted_sources WHERE domain IN ({placeholders})",
                candidates,
            ).fetchall()
        if not rows:
            return None
        by_domain = {r["domain"]: _row_to_dict(r) for r in rows}
        for cand in candidates:
            if cand in by_domain:
                return by_domain[cand]
        return None

    def is_verified_image_source(self, url_or_domain: str) -> Optional[Dict]:
        """Return the trusted_sources row iff it is active, verified, and a pharmacy or manufacturer."""
        row = self.get_source(url_or_domain)
        if not row:
            return None
        if not row.get("is_active"):
            return None
        if row.get("trust_status") != "verified":
            return None
        if row.get("source_type") not in ("pharmacy", "manufacturer"):
            return None
        return row

    def get_verified_image_domains(
        self,
        *,
        country: str = "NG",
        manufacturer_hint: Optional[str] = None,
    ) -> List[str]:
        """Return verified pharmacy domains for `country` plus any verified matching manufacturer domain."""
        pharmacies = self.list_sources(
            trust_status="verified",
            source_type="pharmacy",
            country=country,
            active_only=True,
        )
        domains = [p["domain"] for p in pharmacies]

        if manufacturer_hint:
            tokens = [
                tok.lower()
                for tok in re.findall(r"[A-Za-z0-9]+", manufacturer_hint)
                if len(tok) >= 3 and tok.lower() not in _GENERIC_MFR_WORDS
            ]
            if tokens:
                manufacturers = self.list_sources(
                    trust_status="verified",
                    source_type="manufacturer",
                    active_only=True,
                )
                for mfr in manufacturers:
                    dom = mfr["domain"]
                    primary = "".join(lbl for lbl in dom.split(".")[:-1] if lbl)
                    name_low = (mfr.get("name") or "").lower()
                    if any(tok in primary or tok in name_low for tok in tokens):
                        if dom not in domains:
                            domains.append(dom)
        return domains


def main() -> None:
    store = TrustedSourceStore()
    added = store.seed_defaults()
    all_rows = store.list_sources(active_only=False)
    verified = [r for r in all_rows if r["trust_status"] == "verified"]
    provisional = [r for r in all_rows if r["trust_status"] == "provisional"]
    print(f"Trusted sources DB: {store.db_path}")
    print(f"Newly seeded: {added} | Total: {len(all_rows)} ({len(verified)} verified, {len(provisional)} provisional)\n")
    for status_label, group in (("VERIFIED", verified), ("PROVISIONAL", provisional)):
        print(f"== {status_label} ({len(group)}) ==")
        for r in group:
            crit = ", ".join(r["criteria_met"]) or "none"
            print(f"  - {r['domain']:28s} [{r['source_type']:12s} | {r['country']:6s}] criteria=[{crit}]")
        print()


if __name__ == "__main__":
    main()

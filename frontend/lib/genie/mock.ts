import type {
  Pharmacy,
  RecentCheck,
  Scenario,
  VerificationResult,
} from "./types";

/**
 * Mock data for the frontend prototype. Everything here mirrors the shape of
 * GENIE AI's real responses so wiring the API later is a drop-in swap.
 */

export const PHARMACIES: Pharmacy[] = [
  {
    id: "ph_1",
    name: "GreenLife Pharmacy",
    address: "12 Adeola Odeku St",
    area: "Victoria Island",
    distanceKm: 0.8,
    licenseId: "PCL-1182",
    verified: true,
    stockConfirmed: true,
  },
  {
    id: "ph_2",
    name: "MedPlus Yaba",
    address: "44 Herbert Macaulay Wy",
    area: "Yaba",
    distanceKm: 1.6,
    licenseId: "PCL-0447",
    verified: true,
    stockConfirmed: false,
  },
  {
    id: "ph_3",
    name: "HealthPlus Lekki",
    address: "Admiralty Way",
    area: "Lekki Phase 1",
    distanceKm: 2.4,
    licenseId: "PCL-0912",
    verified: true,
    stockConfirmed: true,
  },
  {
    id: "ph_4",
    name: "Family Care Chemists",
    address: "9 Ojuelegba Rd",
    area: "Surulere",
    distanceKm: 3.1,
    licenseId: "PCL-2210",
    verified: false,
    stockConfirmed: false,
  },
  {
    id: "ph_5",
    name: "Emzor Care Hub",
    address: "27 Ikorodu Rd",
    area: "Somolu",
    distanceKm: 4.2,
    licenseId: "PCL-3051",
    verified: true,
    stockConfirmed: true,
  },
];

export const SCENARIO_VALUES: Scenario[] = [
  "authentic",
  "suspicious",
  "counterfeit",
  "unknown",
];

export const SCENARIO_OPTIONS: { id: Scenario; label: string; hint: string }[] = [
  { id: "authentic", label: "Genuine", hint: "Augmentin 625 mg" },
  { id: "suspicious", label: "Suspicious", hint: "Coartem 20/120" },
  { id: "counterfeit", label: "Counterfeit", hint: "Cloned Panadol" },
  { id: "unknown", label: "Unknown", hint: "Unregistered pack" },
];

const RESULTS: Record<Scenario, VerificationResult> = {
  authentic: {
    id: "chk_augmentin",
    scenario: "authentic",
    verdict: "authentic",
    confidence: 0.97,
    scannedAt: "2026-11-24T09:41:00Z",
    matchedProduct: {
      brandName: "Augmentin 625 mg",
      genericName: "Amoxicillin / Clavulanic Acid",
      strength: "500 mg + 125 mg",
      form: "Film-coated tablet",
      manufacturer: "GlaxoSmithKline Consumer Nigeria PLC",
      nafdacNumber: "04-1928",
      registrationStatus: "active",
    },
    matchedVersion: {
      label: "Pack v3 · 2024 redesign",
      effectiveFrom: "Mar 2024",
      isCurrent: true,
    },
    signals: [
      {
        key: "visual",
        label: "Visual match",
        status: "pass",
        detail: "96% similarity against 7 verified packaging photos.",
      },
      {
        key: "nafdac",
        label: "NAFDAC registration",
        status: "pass",
        detail: "04-1928 is active in the Greenbook and matches this manufacturer.",
      },
      {
        key: "barcode",
        label: "Barcode / GTIN",
        status: "pass",
        detail: "GTIN check digit is valid and maps to this exact product.",
      },
      {
        key: "batch_consistency",
        label: "Batch & expiry",
        status: "pass",
        detail: "Batch format, expiry window and dosage are all consistent.",
      },
      {
        key: "version_match",
        label: "Packaging version",
        status: "pass",
        detail: "Matches the current 2024 pack design.",
      },
    ],
    alternatives: [],
    note: "All independent signals agree. Safe to use as directed.",
  },

  suspicious: {
    id: "chk_coartem",
    scenario: "suspicious",
    verdict: "suspicious",
    confidence: 0.41,
    scannedAt: "2026-11-23T15:12:00Z",
    matchedProduct: {
      brandName: "Coartem 20/120",
      genericName: "Artemether / Lumefantrine",
      strength: "20 mg / 120 mg",
      form: "Tablet",
      manufacturer: "Novartis Pharma AG",
      nafdacNumber: "04-4471",
      registrationStatus: "active",
    },
    matchedVersion: {
      label: "Pack v5 · 2023",
      effectiveFrom: "Aug 2023",
      isCurrent: true,
    },
    signals: [
      {
        key: "visual",
        label: "Visual match",
        status: "warn",
        detail: "Print density is lower in the dosage panel than reference photos.",
      },
      {
        key: "nafdac",
        label: "NAFDAC registration",
        status: "pass",
        detail: "04-4471 is active and registered to Novartis.",
      },
      {
        key: "barcode",
        label: "Barcode / GTIN",
        status: "pass",
        detail: "GTIN is valid for Coartem 20/120.",
      },
      {
        key: "batch_consistency",
        label: "Batch & expiry",
        status: "warn",
        detail: "Expiry month is unusually close to a known repack pattern.",
      },
      {
        key: "version_match",
        label: "Packaging version",
        status: "warn",
        detail: "Layout resembles the 2021 pack, not the current 2023 design.",
      },
    ],
    alternatives: PHARMACIES.filter((p) => p.verified),
    note: "Three weak signals. Confirm at a verified pharmacy before use.",
  },

  counterfeit: {
    id: "chk_panadol",
    scenario: "counterfeit",
    verdict: "likely_counterfeit",
    confidence: 0.93,
    scannedAt: "2026-11-22T11:05:00Z",
    matchedProduct: {
      brandName: "Panadol Extra",
      genericName: "Paracetamol / Caffeine",
      strength: "500 mg / 65 mg",
      form: "Caplet",
      manufacturer: "GSK Consumer Healthcare (claimed)",
      nafdacNumber: "04-0098",
      registrationStatus: "revoked",
    },
    matchedVersion: {
      label: "Pack v2 · 2022",
      effectiveFrom: "Jan 2022",
      isCurrent: false,
    },
    signals: [
      {
        key: "visual",
        label: "Visual match",
        status: "fail",
        detail: "Security hologram missing and the logo kerning is off by ~2 px.",
      },
      {
        key: "nafdac",
        label: "NAFDAC registration",
        status: "fail",
        detail: "04-0098 was revoked in 2024 and is not valid for this pack.",
      },
      {
        key: "barcode",
        label: "Barcode / GTIN",
        status: "fail",
        detail: "GTIN check digit is invalid for an EAN-13 barcode.",
      },
      {
        key: "batch_consistency",
        label: "Batch & expiry",
        status: "fail",
        detail: "Expiry printed as 13/2026 — an impossible month.",
      },
      {
        key: "version_match",
        label: "Packaging version",
        status: "fail",
        detail: "Print layout matches no known genuine version on file.",
      },
    ],
    alternatives: PHARMACIES.filter((p) => p.verified),
    note: "Do not use this product. It matches known counterfeit patterns.",
  },

  unknown: {
    id: "chk_unknown",
    scenario: "unknown",
    verdict: "unknown",
    confidence: 0.12,
    scannedAt: "2026-11-21T08:30:00Z",
    matchedProduct: null,
    matchedVersion: null,
    signals: [
      {
        key: "visual",
        label: "Visual match",
        status: "unknown",
        detail: "No verified reference images for this product yet.",
      },
      {
        key: "nafdac",
        label: "NAFDAC registration",
        status: "unknown",
        detail: "No registration match found in the Greenbook mirror.",
      },
      {
        key: "barcode",
        label: "Barcode / GTIN",
        status: "unknown",
        detail: "GTIN is not linked to any verified product record.",
      },
      {
        key: "batch_consistency",
        label: "Batch & expiry",
        status: "unknown",
        detail: "Skipped until a product record exists.",
      },
      {
        key: "version_match",
        label: "Packaging version",
        status: "unknown",
        detail: "No packaging history on file.",
      },
    ],
    alternatives: [],
    discoveryJobId: "dsc_8f21c",
    note: "Genie is crawling official sources now. Check back in a few minutes.",
  },
};

export function getResult(scenario: Scenario): VerificationResult {
  return RESULTS[scenario] ?? RESULTS.unknown;
}

export const RECENT_CHECKS: RecentCheck[] = [
  {
    id: "chk_augmentin",
    scenario: "authentic",
    productName: "Augmentin 625 mg",
    verdict: "authentic",
    when: "Today, 09:41",
  },
  {
    id: "chk_coartem",
    scenario: "suspicious",
    productName: "Coartem 20/120",
    verdict: "suspicious",
    when: "Yesterday",
  },
  {
    id: "chk_panadol",
    scenario: "counterfeit",
    productName: "Panadol Extra",
    verdict: "likely_counterfeit",
    when: "Mon, 11:05",
  },
  {
    id: "chk_unknown",
    scenario: "unknown",
    productName: "Rovamycin 3 MIU",
    verdict: "unknown",
    when: "Sun, 08:30",
  },
];

export const STATS = [
  { value: "9,065", label: "NAFDAC records" },
  { value: "5", label: "signals per check" },
  { value: "12", label: "pack versions" },
];

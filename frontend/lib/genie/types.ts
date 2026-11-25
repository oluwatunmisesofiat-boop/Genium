// Shapes mirror the Genie <-> Genium verification contract (BUILD_PLAN §5B).
// Everything here is mocked on the client until the API is wired up.

export type Verdict =
  | "authentic"
  | "likely_authentic"
  | "suspicious"
  | "likely_counterfeit"
  | "unknown";

/** Demo scenarios drive the simulated scan flow. */
export type Scenario = "authentic" | "suspicious" | "counterfeit" | "unknown";

export type SignalKey =
  | "visual"
  | "nafdac"
  | "barcode"
  | "batch_consistency"
  | "version_match";

export type SignalStatus = "pass" | "warn" | "fail" | "unknown";

export interface VerificationSignal {
  key: SignalKey;
  label: string;
  status: SignalStatus;
  detail: string;
}

export interface MatchedProduct {
  brandName: string;
  genericName: string;
  strength: string;
  form: string;
  manufacturer: string;
  nafdacNumber: string;
  registrationStatus: "active" | "expired" | "revoked" | "unregistered";
}

export interface MatchedVersion {
  label: string;
  effectiveFrom: string;
  isCurrent: boolean;
}

export interface Pharmacy {
  id: string;
  name: string;
  address: string;
  area: string;
  distanceKm: number;
  licenseId: string;
  /** Pharmacy itself is licensed + verified. */
  verified: boolean;
  /** Product availability has been confirmed separately. */
  stockConfirmed: boolean;
  latitude?: number;
  longitude?: number;
  googleMapsUrl?: string;
  directionsUrl?: string;
  phone?: string | null;
  rating?: number | null;
  ratingCount?: number | null;
  pcnMatchedChain?: string | null;
  source?: string;
}

export interface NearbyPharmaciesResponse {
  queryLocation: string;
  latitude: number;
  longitude: number;
  radiusKm: number;
  total: number;
  verifiedCount: number;
  source: string;
  pharmacies: Pharmacy[];
}

export interface VerificationResult {
  id: string;
  scenario: Scenario;
  verdict: Verdict;
  confidence: number;
  scannedAt: string;
  matchedProduct: MatchedProduct | null;
  matchedVersion: MatchedVersion | null;
  signals: VerificationSignal[];
  alternatives: Pharmacy[];
  discoveryJobId?: string;
  note?: string;
}

export interface RecentCheck {
  id: string;
  scenario: Scenario;
  productName: string;
  verdict: Verdict;
  when: string;
}

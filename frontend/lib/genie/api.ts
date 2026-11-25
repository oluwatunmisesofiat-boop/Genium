import type { NearbyPharmaciesResponse, Pharmacy } from "./types";

export const GENIE_AI_API_BASE = (
  process.env.NEXT_PUBLIC_GENIE_AI_URL || "http://102.209.46.116:8001"
).replace(/\/+$/, "");

export const ML_INFRA_API_BASE = (
  process.env.NEXT_PUBLIC_ML_INFRA_URL || "http://102.209.46.116:8000"
).replace(/\/+$/, "");

export const ABEOKUTA_DEFAULT_COORDS = {
  lat: 7.1475,
  lng: 3.3619,
  label: "Abeokuta, Ogun State",
};

/** Offline Abeokuta PCN-verified fallback if the API server is unreachable. */
export const ABEOKUTA_FALLBACK_PHARMACIES: Pharmacy[] = [
  {
    id: "ph_abk_vanguard",
    name: "Vanguard Pharmacy — Oke-Ilewo",
    address: "38 Lalubu Street (Beside Zenith Bank), Oke-Ilewo, Abeokuta",
    area: "Oke-Ilewo, Abeokuta",
    distanceKm: 2.8,
    licenseId: "PCN-OGN-VP-001",
    verified: true,
    stockConfirmed: false,
    latitude: 7.1419,
    longitude: 3.3372,
    googleMapsUrl:
      "https://www.google.com/maps/search/?api=1&query=Vanguard+Pharmacy+38+Lalubu+Street+Oke-Ilewo+Abeokuta",
    directionsUrl: "https://www.google.com/maps/dir/?api=1&destination=7.141900,3.337200",
  },
  {
    id: "ph_abk_medplus",
    name: "MedPlus Pharmacy — OOPL",
    address: "Opposite Olusegun Obasanjo Presidential Library, IBB Boulevard, Abeokuta",
    area: "IBB Boulevard, Abeokuta",
    distanceKm: 1.5,
    licenseId: "PCN-OGN-MP-002",
    verified: true,
    stockConfirmed: false,
    latitude: 7.1346,
    longitude: 3.3583,
    googleMapsUrl:
      "https://www.google.com/maps/search/?api=1&query=MedPlus+Pharmacy+OOPL+IBB+Boulevard+Abeokuta",
    directionsUrl: "https://www.google.com/maps/dir/?api=1&destination=7.134600,3.358300",
  },
  {
    id: "ph_abk_healthplus",
    name: "HealthPlus Pharmacy — IBB Boulevard",
    address: "Delta Mall (Shoprite), Ibrahim Babangida Boulevard, Abeokuta",
    area: "IBB Boulevard, Abeokuta",
    distanceKm: 1.4,
    licenseId: "PCN-OGN-HP-003",
    verified: true,
    stockConfirmed: false,
    latitude: 7.1361,
    longitude: 3.3562,
    googleMapsUrl:
      "https://www.google.com/maps/search/?api=1&query=HealthPlus+Pharmacy+Delta+Mall+Abeokuta",
    directionsUrl: "https://www.google.com/maps/dir/?api=1&destination=7.136100,3.356200",
  },
  {
    id: "ph_abk_goldcross",
    name: "Gold Cross Pharmacy — Idi-Aba",
    address: "15 Idi-Aba Road (Opposite Federal Medical Centre), Idi-Aba, Abeokuta",
    area: "Idi-Aba, Abeokuta",
    distanceKm: 2.0,
    licenseId: "PCN-OGN-GC-008",
    verified: true,
    stockConfirmed: false,
    latitude: 7.1472,
    longitude: 3.3801,
    googleMapsUrl:
      "https://www.google.com/maps/search/?api=1&query=Gold+Cross+Pharmacy+15+Idi-Aba+Road+Abeokuta",
    directionsUrl: "https://www.google.com/maps/dir/?api=1&destination=7.147200,3.380100",
  },
];

export async function fetchNearbyPharmacies(params: {
  lat?: number;
  lng?: number;
  location?: string;
  radiusKm?: number;
  limit?: number;
  verifiedOnly?: boolean;
}): Promise<NearbyPharmaciesResponse> {
  const qs = new URLSearchParams();
  if (params.lat !== undefined && params.lng !== undefined) {
    qs.set("lat", String(params.lat));
    qs.set("lng", String(params.lng));
  }
  if (params.location) qs.set("location", params.location);
  if (params.radiusKm) qs.set("radius_km", String(params.radiusKm));
  if (params.limit) qs.set("limit", String(params.limit));
  if (params.verifiedOnly) qs.set("verified_only", "true");

  const resp = await fetch(`/api/pharmacies/nearby?${qs.toString()}`, {
    cache: "no-store",
  });
  if (!resp.ok) {
    throw new Error(`Nearby pharmacies API failed (${resp.status})`);
  }
  return (await resp.json()) as NearbyPharmaciesResponse;
}

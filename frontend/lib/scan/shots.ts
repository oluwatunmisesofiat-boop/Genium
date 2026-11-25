import type { ScanShots } from "./frame";

/**
 * Carries the frames the user just captured from /scan over to /result.
 * Lives in memory on purpose: it survives client navigation, and a hard reload
 * of the result route simply shows no evidence strip.
 */
let current: ScanShots | null = null;
const listeners = new Set<() => void>();

export function rememberShots(shots: ScanShots) {
  current = shots;
  listeners.forEach((listener) => listener());
}

export function subscribeShots(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function getShots() {
  return current;
}

export function getServerShots(): ScanShots | null {
  return null;
}

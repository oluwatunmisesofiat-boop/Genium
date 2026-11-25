import type { SignalStatus, Verdict } from "./types";

export type Tone = "ok" | "warn" | "bad" | "info" | "muted" | "accent";

/**
 * Explicit class strings (never built dynamically) so Tailwind can see them.
 */
export const TONE_CLASS: Record<
  Tone,
  { chip: string; dot: string; text: string; fill: string }
> = {
  ok: {
    chip: "border-ok/30 bg-ok/10 text-ok",
    dot: "bg-ok",
    text: "text-ok",
    fill: "bg-ok text-ink",
  },
  warn: {
    chip: "border-warn/30 bg-warn/10 text-warn",
    dot: "bg-warn",
    text: "text-warn",
    fill: "bg-warn text-ink",
  },
  bad: {
    chip: "border-bad/30 bg-bad/10 text-bad",
    dot: "bg-bad",
    text: "text-bad",
    fill: "bg-bad text-white",
  },
  info: {
    chip: "border-info/30 bg-info/10 text-info",
    dot: "bg-info",
    text: "text-info",
    fill: "bg-info text-ink",
  },
  muted: {
    chip: "border-line bg-card text-muted",
    dot: "bg-muted",
    text: "text-muted",
    fill: "bg-card text-fg",
  },
  accent: {
    chip: "border-accent/30 bg-accent/10 text-accent",
    dot: "bg-accent",
    text: "text-accent",
    fill: "bg-accent text-accent-ink",
  },
};

export const VERDICT_META: Record<
  Verdict,
  { label: string; short: string; tone: Tone; blurb: string }
> = {
  authentic: {
    label: "Authentic",
    short: "Real",
    tone: "ok",
    blurb: "Every independent signal agrees with the verified product record.",
  },
  likely_authentic: {
    label: "Likely authentic",
    short: "Likely real",
    tone: "ok",
    blurb: "Most signals match, with one minor gap worth a second look.",
  },
  suspicious: {
    label: "Suspicious",
    short: "Check it",
    tone: "warn",
    blurb: "Some signals disagree. Confirm at a verified pharmacy before use.",
  },
  likely_counterfeit: {
    label: "Likely counterfeit",
    short: "Fake",
    tone: "bad",
    blurb: "Multiple core signals failed. Do not use this product.",
  },
  unknown: {
    label: "Unknown product",
    short: "Unknown",
    tone: "info",
    blurb: "Not in our verified records yet. Genie has started an investigation.",
  },
};

export const STATUS_META: Record<SignalStatus, { label: string; tone: Tone }> = {
  pass: { label: "Match", tone: "ok" },
  warn: { label: "Review", tone: "warn" },
  fail: { label: "Failed", tone: "bad" },
  unknown: { label: "Unknown", tone: "muted" },
};

export function verdictTone(verdict: Verdict): Tone {
  return VERDICT_META[verdict].tone;
}

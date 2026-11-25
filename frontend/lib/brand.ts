/**
 * Brand geometry — the single source of truth for the Genium mark.
 *
 * Both the `LogoMark` component and `scripts/generate-icons.mjs` read from here
 * so the on-screen mark and the installed app icons can never drift apart.
 * Units are the 32-unit grid the mark is drawn on.
 */
export const LOGO_GRID = 32;

export const LOGO_CARD = {
  width: 11,
  height: 13.5,
  radius: 3.9,
  tilt: 20,
} as const;

/** The card behind: the back of the pack. */
export const LOGO_BACK = { x: 12.6, y: 10.8 } as const;

/** The card in front: the face of the pack. */
export const LOGO_FRONT = { x: 19.4, y: 19.2 } as const;

export const LOGO_INK = "#0b0e13";
export const LOGO_ACCENT = "#c8f65a";
export const LOGO_ACCENT_DIM = "#566a2f";

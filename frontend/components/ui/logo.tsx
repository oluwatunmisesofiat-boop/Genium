import { LOGO_BACK, LOGO_CARD, LOGO_FRONT, LOGO_GRID } from "@/lib/brand";
import { cn } from "@/lib/cn";

const CARDS = [
  { center: LOGO_BACK, opacity: 0.42 },
  { center: LOGO_FRONT, opacity: 1 },
];

/**
 * The Genium mark: the front and the back of a pack, fanned like the pair of
 * frames the scan flow captures and the result screen shows.
 *
 * Geometry comes from `lib/brand.ts` — the same source the app icons are
 * generated from — so the mark stays identical from a 16px favicon upward.
 */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox={`0 0 ${LOGO_GRID} ${LOGO_GRID}`}
      aria-hidden
      focusable="false"
      className={cn("shrink-0", className)}
    >
      {CARDS.map(({ center, opacity }) => (
        <rect
          key={`${center.x}-${center.y}`}
          x={center.x - LOGO_CARD.width / 2}
          y={center.y - LOGO_CARD.height / 2}
          width={LOGO_CARD.width}
          height={LOGO_CARD.height}
          rx={LOGO_CARD.radius}
          fill="currentColor"
          fillOpacity={opacity}
          transform={`rotate(${-LOGO_CARD.tilt} ${center.x} ${center.y})`}
        />
      ))}
    </svg>
  );
}

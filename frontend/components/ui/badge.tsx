import type { ReactNode } from "react";
import { cn } from "@/lib/cn";
import { TONE_CLASS, type Tone } from "@/lib/genie/verdict";

export function Badge({
  tone = "muted",
  dot = false,
  className,
  children,
}: {
  tone?: Tone;
  dot?: boolean;
  className?: string;
  children: ReactNode;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-pill border px-2.5 py-1 text-xs font-semibold",
        TONE_CLASS[tone].chip,
        className,
      )}
    >
      {dot && (
        <span className={cn("h-1.5 w-1.5 rounded-pill", TONE_CLASS[tone].dot)} />
      )}
      {children}
    </span>
  );
}

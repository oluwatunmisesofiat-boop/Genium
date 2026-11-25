"use client";

import { motion } from "framer-motion";

/**
 * Three segments, one per step. `progress` is a continuous 0–3 value so the
 * active segment can creep forward while the scan runs.
 */
export function StepBar({
  progress,
  slow = false,
}: {
  progress: number;
  slow?: boolean;
}) {
  return (
    <div className="flex w-full items-center gap-1.5">
      {[0, 1, 2].map((index) => {
        const fill = Math.min(1, Math.max(0, progress - index));

        return (
          <div
            key={index}
            className="relative h-1.5 flex-1 overflow-hidden rounded-pill bg-line"
          >
            <motion.span
              className="absolute inset-0 origin-left rounded-pill bg-accent"
              initial={false}
              animate={{ scaleX: fill }}
              transition={{ duration: slow ? 2.4 : 0.4, ease: "easeOut" }}
            />
          </div>
        );
      })}
    </div>
  );
}

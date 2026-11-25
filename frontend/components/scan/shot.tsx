/* eslint-disable @next/next/no-img-element */
// Captured frames only exist at runtime as data URLs, so the optimised image
// pipeline can't be used here.

import { cn } from "@/lib/cn";

export function Shot({ src, className }: { src: string; className?: string }) {
  return (
    <img
      src={src}
      alt=""
      decoding="async"
      className={cn("h-full w-full object-cover", className)}
    />
  );
}

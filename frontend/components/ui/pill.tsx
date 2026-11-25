import Link from "next/link";
import type { ComponentProps } from "react";
import { cn } from "@/lib/cn";

export type PillSize = "sm" | "md" | "lg";
export type PillVariant =
  | "primary"
  | "secondary"
  | "ghost"
  | "danger"
  | "outline";

const SIZES: Record<PillSize, string> = {
  sm: "h-9 gap-1.5 px-4 text-sm",
  md: "h-11 gap-2 px-5 text-[15px]",
  lg: "h-14 gap-2.5 px-6 text-base",
};

const VARIANTS: Record<PillVariant, string> = {
  primary: "bg-accent text-accent-ink hover:brightness-95",
  secondary: "border border-line bg-card text-fg hover:bg-surface",
  ghost: "text-fg hover:bg-card",
  danger: "bg-bad text-white",
  outline: "border border-accent/60 text-accent hover:bg-accent/10",
};

export function pillStyles(
  variant: PillVariant = "primary",
  size: PillSize = "md",
  className?: string,
) {
  return cn(
    "inline-flex select-none items-center justify-center rounded-pill font-semibold transition-[transform,filter,background-color] duration-150 active:scale-[0.97] disabled:pointer-events-none disabled:opacity-50",
    SIZES[size],
    VARIANTS[variant],
    className,
  );
}

type PillBase = { variant?: PillVariant; size?: PillSize; className?: string };

export function PillButton({
  variant,
  size,
  className,
  type = "button",
  ...props
}: ComponentProps<"button"> & PillBase) {
  return (
    <button
      type={type}
      className={pillStyles(variant, size, className)}
      {...props}
    />
  );
}

export function PillLink({
  variant,
  size,
  className,
  ...props
}: ComponentProps<typeof Link> & PillBase) {
  return <Link className={pillStyles(variant, size, className)} {...props} />;
}

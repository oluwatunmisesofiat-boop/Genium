"use client";

import Link from "next/link";
import { Button } from "@ui-sdk/react/button";
import { LogoMark } from "@/components/ui/logo";

export function SiteNav() {
  return (
    <header className="pointer-events-none fixed inset-x-0 top-0 z-30">
      <nav className="nav-safe-top mx-auto flex w-full max-w-[480px] items-center justify-between px-4">
        <Link
          href="/"
          className="pointer-events-auto flex h-11 items-center gap-2 rounded-pill border border-line bg-surface pl-2 pr-4 transition-transform duration-200 ease-out active:scale-95"
        >
          <LogoMark className="h-8 w-8 text-accent" />
          <span className="font-display text-sm font-bold leading-none tracking-tight">
            genium
          </span>
        </Link>



        <Button
          aria-label="Open menu"
          className="pointer-events-auto flex h-11 w-11 items-center justify-center rounded-pill border border-line bg-surface text-fg transition-transform duration-200 ease-out active:scale-95"
        >
          <svg
            width="18"
            height="18"
            viewBox="0 0 18 18"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            aria-hidden
          >
            <path d="M2.5 6.25h13" />
            <path d="M6.5 11.75h9" />
          </svg>
        </Button>

      </nav>
    </header>
  );
}

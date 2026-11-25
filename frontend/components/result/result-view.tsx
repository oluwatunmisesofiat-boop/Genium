"use client";

import Link from "next/link";
import { useEffect, useState, useSyncExternalStore } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { BackButton } from "@/components/app/back-button";
import { Shot } from "@/components/scan/shot";
import { Badge } from "@/components/ui/badge";
import {
  AlertIcon,
  ArrowRightIcon,
  CheckIcon,
  QuestionIcon,
  StoreIcon,
  XIcon,
} from "@/components/ui/icons";
import { PillLink } from "@/components/ui/pill";
import { VerdictGlyph } from "@/components/ui/verdict-glyph";
import { cn } from "@/lib/cn";
import type { SignalStatus, VerificationResult } from "@/lib/genie/types";
import { STATUS_META, TONE_CLASS, VERDICT_META } from "@/lib/genie/verdict";
import { getServerShots, getShots, subscribeShots } from "@/lib/scan/shots";

function StatusGlyph({ status }: { status: SignalStatus }) {
  if (status === "pass") return <CheckIcon className="h-4 w-4" />;
  if (status === "warn") return <AlertIcon className="h-4 w-4" />;
  if (status === "fail") return <XIcon className="h-4 w-4" />;
  return <QuestionIcon className="h-4 w-4" />;
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-4 py-2.5 first:pt-0 last:pb-0">
      <dt className="shrink-0 text-xs text-muted">{label}</dt>
      <dd className="text-right text-xs font-semibold">{value}</dd>
    </div>
  );
}

export function ResultView({ result }: { result: VerificationResult }) {
  const [open, setOpen] = useState(false);
  const shots = useSyncExternalStore(subscribeShots, getShots, getServerShots);

  const meta = VERDICT_META[result.verdict];
  const product = result.matchedProduct;
  const percent = Math.round(result.confidence * 100);
  const hasShots = Boolean(shots?.front || shots?.back);

  useEffect(() => {
    if (!open) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, [open]);

  return (
    <div className="flex min-h-dvh flex-col gap-5 px-4 pb-[max(1.5rem,env(safe-area-inset-bottom))] pt-[max(0.75rem,env(safe-area-inset-top))]">
      <header className="flex items-center justify-between">
        <BackButton />
      </header>

      {hasShots && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, ease: "easeOut" }}
          className="flex items-center justify-center gap-3"
        >
          {shots?.front && (
            <span className="h-24 w-[72px] -rotate-6 overflow-hidden rounded-[18px] border border-line">
              <Shot src={shots.front} />
            </span>
          )}
          {shots?.back && (
            <span className="h-24 w-[72px] rotate-6 overflow-hidden rounded-[18px] border border-line">
              <Shot src={shots.back} />
            </span>
          )}
        </motion.div>
      )}

      <motion.section
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.04, ease: "easeOut" }}
        className={cn("rounded-card p-5", TONE_CLASS[meta.tone].fill)}
      >
        <div className="flex items-start justify-between">
          <span className="flex h-11 w-11 items-center justify-center rounded-pill bg-black/15">
            <VerdictGlyph verdict={result.verdict} className="h-6 w-6" />
          </span>
          <span className="font-display text-2xl font-extrabold tabular-nums">
            {percent}%
          </span>
        </div>

        {product && (
          <p className="mt-4 truncate text-sm font-semibold opacity-70">
            {product.brandName}
          </p>
        )}

        <h1 className="mt-1 text-balance font-display text-[2.1rem] font-extrabold leading-[0.95] tracking-[-0.03em]">
          {meta.label}
        </h1>

        <div className="mt-4 h-2 overflow-hidden rounded-pill bg-black/15">
          <motion.div
            initial={{ scaleX: 0 }}
            animate={{ scaleX: 1 }}
            transition={{ duration: 0.7, delay: 0.2, ease: "easeOut" }}
            style={{ width: `${percent}%` }}
            className="h-full origin-left rounded-pill bg-current"
          />
        </div>
      </motion.section>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.08, ease: "easeOut" }}
        className="grid grid-cols-5 gap-2"
      >
        {result.signals.map((signal) => {
          const status = STATUS_META[signal.status];
          return (
            <span
              key={signal.key}
              className={cn(
                "flex h-12 items-center justify-center rounded-pill border",
                TONE_CLASS[status.tone].chip,
              )}
            >
              <StatusGlyph status={signal.status} />
            </span>
          );
        })}
      </motion.div>

      <div className="mt-auto flex flex-col gap-2 pt-2">
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="flex h-14 w-full items-center justify-center gap-2 rounded-pill bg-accent text-base font-semibold text-accent-ink transition-transform duration-150 ease-out active:scale-[0.98]"
        >
          View more
          <ArrowRightIcon className="h-4 w-4" />
        </button>

        <div className="grid grid-cols-2 gap-2">
          <PillLink href="/scan" size="md" variant="secondary" className="w-full">
            Scan another
          </PillLink>
          <PillLink href="/" size="md" variant="ghost" className="w-full">
            Done
          </PillLink>
        </div>
      </div>

      <DetailsSheet
        result={result}
        open={open}
        shots={shots}
        onClose={() => setOpen(false)}
      />
    </div>
  );
}

function DetailsSheet({
  result,
  open,
  shots,
  onClose,
}: {
  result: VerificationResult;
  open: boolean;
  shots: ReturnType<typeof getShots>;
  onClose: () => void;
}) {
  const product = result.matchedProduct;

  return (
    <AnimatePresence>
      {open && (
        <div className="fixed inset-0 z-50 flex flex-col justify-end">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={onClose}
            className="absolute inset-0 bg-ink/70 backdrop-blur-sm"
          />

          <motion.div
            initial={{ y: "100%" }}
            animate={{ y: 0 }}
            exit={{ y: "100%" }}
            transition={{ type: "spring", stiffness: 340, damping: 34 }}
            className="relative max-h-[88svh] overflow-y-auto overscroll-contain rounded-t-[32px] border-t border-line bg-surface pb-[max(1.5rem,env(safe-area-inset-bottom))]"
          >
            <div className="sticky top-0 z-10 bg-surface px-5 pb-2 pt-3">
              <div className="flex justify-center">
                <span className="h-1.5 w-10 rounded-pill bg-line" />
              </div>
              <button
                type="button"
                onClick={onClose}
                aria-label="Close"
                className="absolute right-4 top-3 flex h-9 w-9 items-center justify-center rounded-pill border border-line bg-card text-fg transition-transform duration-150 ease-out active:scale-90"
              >
                <XIcon className="h-4 w-4" />
              </button>
            </div>

            <div className="flex flex-col gap-4 px-5 pt-1">
              {shots?.front || shots?.back ? (
                <div className="flex items-center justify-center gap-3">
                  {shots?.front && (
                    <span className="h-28 w-20 -rotate-3 overflow-hidden rounded-[16px] border border-line">
                      <Shot src={shots.front} />
                    </span>
                  )}
                  {shots?.back && (
                    <span className="h-28 w-20 rotate-3 overflow-hidden rounded-[16px] border border-line">
                      <Shot src={shots.back} />
                    </span>
                  )}
                </div>
              ) : null}

              {product ? (
                <div className="rounded-card border border-line bg-card p-5">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <h2 className="font-display text-xl font-bold leading-tight">
                        {product.brandName}
                      </h2>
                      <p className="truncate text-sm text-muted">
                        {product.genericName}
                      </p>
                    </div>
                    <Badge
                      tone={
                        product.registrationStatus === "active" ? "ok" : "bad"
                      }
                      dot
                    >
                      {product.registrationStatus}
                    </Badge>
                  </div>

                  <dl className="mt-4 flex flex-col divide-y divide-line">
                    <InfoRow label="NAFDAC no." value={product.nafdacNumber} />
                    <InfoRow label="Manufacturer" value={product.manufacturer} />
                    <InfoRow label="Strength" value={product.strength} />
                    <InfoRow label="Dosage form" value={product.form} />
                    {result.matchedVersion && (
                      <InfoRow
                        label="Pack version"
                        value={`${result.matchedVersion.label}${
                          result.matchedVersion.isCurrent
                            ? " · current"
                            : " · historical"
                        }`}
                      />
                    )}
                  </dl>
                </div>
              ) : (
                <div className="rounded-card border border-line bg-card p-5">
                  <p className="text-sm leading-relaxed text-muted">
                    No verified product record matched this scan yet. Genie is
                    investigating it against official sources.
                  </p>
                  {result.discoveryJobId && (
                    <div className="mt-3 flex items-center justify-between rounded-pill border border-info/30 bg-info/10 px-4 py-2.5">
                      <span className="text-xs font-semibold text-info">
                        Investigation
                      </span>
                      <span className="font-mono text-xs text-info">
                        {result.discoveryJobId}
                      </span>
                    </div>
                  )}
                </div>
              )}

              <div className="flex flex-col gap-2">
                {result.signals.map((signal) => {
                  const status = STATUS_META[signal.status];
                  return (
                    <div
                      key={signal.key}
                      className="rounded-card border border-line bg-card p-4"
                    >
                      <div className="flex items-center justify-between gap-3">
                        <div className="flex min-w-0 items-center gap-2.5">
                          <span
                            className={cn(
                              "flex h-8 w-8 shrink-0 items-center justify-center rounded-pill border",
                              TONE_CLASS[status.tone].chip,
                            )}
                          >
                            <StatusGlyph status={signal.status} />
                          </span>
                          <span className="truncate text-sm font-semibold">
                            {signal.label}
                          </span>
                        </div>
                        <Badge tone={status.tone}>{status.label}</Badge>
                      </div>
                      <p className="mt-2.5 text-xs leading-relaxed text-muted">
                        {signal.detail}
                      </p>
                    </div>
                  );
                })}
              </div>

              {result.alternatives.length > 0 && (
                <Link href="/pharmacies" className="block">
                  <div className="flex items-center gap-3 rounded-card border border-accent/30 bg-accent/10 p-4">
                    <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-pill bg-accent text-accent-ink">
                      <StoreIcon className="h-5 w-5" />
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-semibold">
                        Verified pharmacies nearby
                      </p>
                    </div>
                    <ArrowRightIcon className="h-5 w-5 shrink-0 text-accent" />
                  </div>
                </Link>
              )}

              {result.note && (
                <p className="rounded-card border border-line bg-card p-4 text-xs leading-relaxed text-muted">
                  {result.note}
                </p>
              )}
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}

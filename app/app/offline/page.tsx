import type { Metadata } from "next";
import { Screen } from "@/components/app/screen";
import { SparkleIcon } from "@/components/ui/icons";
import { PillLink } from "@/components/ui/pill";

export const metadata: Metadata = { title: "Offline" };

export default function OfflinePage() {
  return (
    <Screen className="items-center justify-center text-center">
      <span className="flex h-16 w-16 items-center justify-center rounded-pill bg-accent text-accent-ink">
        <SparkleIcon className="h-8 w-8" />
      </span>
      <div>
        <h1 className="text-2xl font-bold tracking-tight">You&apos;re offline</h1>
        <p className="mt-2 max-w-[32ch] text-sm leading-relaxed text-muted">
          Genie needs a connection to cross-check product records. Reconnect and
          we&apos;ll pick up right where you left off.
        </p>
      </div>
      <PillLink href="/" variant="secondary">
        Try again
      </PillLink>
    </Screen>
  );
}

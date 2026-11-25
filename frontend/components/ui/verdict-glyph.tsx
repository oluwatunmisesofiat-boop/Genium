import type { Verdict } from "@/lib/genie/types";
import { AlertIcon, QuestionIcon, ShieldCheckIcon, XIcon } from "./icons";

export function VerdictGlyph({
  verdict,
  className = "h-5 w-5",
}: {
  verdict: Verdict;
  className?: string;
}) {
  switch (verdict) {
    case "authentic":
    case "likely_authentic":
      return <ShieldCheckIcon className={className} />;
    case "suspicious":
      return <AlertIcon className={className} />;
    case "likely_counterfeit":
      return <XIcon className={className} />;
    default:
      return <QuestionIcon className={className} />;
  }
}

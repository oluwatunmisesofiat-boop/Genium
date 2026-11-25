import { Fragment } from "react";
import { cn } from "@/lib/cn";

type BlurTextProps = {
  text: string;
  /** Word rendered in the accent colour. */
  highlight?: string;
  /** Extra delay before the first word, in milliseconds. */
  delay?: number;
  className?: string;
};

/**
 * Word-by-word blur reveal. Pure CSS (no JS, no layout work): each word is an
 * inline-block that fades and unblurs with a staggered delay, so it stays
 * cheap on low-end phones.
 */
export function BlurText({
  text,
  highlight,
  delay = 0,
  className,
}: BlurTextProps) {
  const words = text.split(" ");

  return (
    <span className={className}>
      {words.map((word, index) => (
        <Fragment key={`${word}-${index}`}>
          {index > 0 ? " " : null}
          <span
            className={cn(
              "blur-word",
              highlight && word === highlight && "text-accent",
            )}
            style={{ animationDelay: `${delay + index * 80}ms` }}
          >
            {word}
          </span>
        </Fragment>
      ))}
    </span>
  );
}

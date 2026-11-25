"use client";

import { useEffect, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { Button } from "@ui-sdk/react/button";
import { Shot } from "@/components/scan/shot";
import { AlertIcon, CameraIcon, ImageIcon, ScanIcon } from "@/components/ui/icons";
import { cn } from "@/lib/cn";
import { captureFrame, type ScanShots, type ScanSide } from "@/lib/scan/frame";

const CORNERS = [
  "left-0 top-0 rounded-tl-[22px] border-l-2 border-t-2",
  "right-0 top-0 rounded-tr-[22px] border-r-2 border-t-2",
  "bottom-0 left-0 rounded-bl-[22px] border-b-2 border-l-2",
  "bottom-0 right-0 rounded-br-[22px] border-b-2 border-r-2",
];

const ROUND_BUTTON =
  "flex h-12 w-12 items-center justify-center rounded-pill border border-line bg-surface text-fg transition-transform duration-150 ease-out active:scale-90";

export function ScanCamera({
  side,
  shots,
  onFrame,
}: {
  side: ScanSide;
  shots: ScanShots;
  onFrame: (dataUrl: string) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const nativeRef = useRef<HTMLInputElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<"starting" | "live" | "blocked">("starting");
  const [flash, setFlash] = useState(0);
  const reduced = useReducedMotion();

  // The stream lives for the whole capture phase, so stepping from front to
  // back never restarts the camera.
  useEffect(() => {
    let cancelled = false;
    let stream: MediaStream | null = null;

    async function start() {
      if (!navigator.mediaDevices?.getUserMedia) {
        setStatus("blocked");
        return;
      }

      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: { ideal: "environment" },
            width: { ideal: 1440 },
            height: { ideal: 1440 },
          },
        });
      } catch {
        if (!cancelled) setStatus("blocked");
        return;
      }

      const video = videoRef.current;
      if (cancelled || !video) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      video.srcObject = stream;
      await video.play().catch(() => undefined);
      if (!cancelled) setStatus("live");
    }

    void start();

    return () => {
      cancelled = true;
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  function shoot() {
    const video = videoRef.current;

    if (!video || status !== "live") {
      // No live feed (still starting, or blocked) — hand off to the camera app.
      nativeRef.current?.click();
      return;
    }

    // First frames can arrive after playback starts; capturing here would grab
    // a black frame and wrongly look like a bad photo.
    if (video.readyState < 2) return;

    const frame = captureFrame(video);
    if (!frame) return;

    setFlash((count) => count + 1);
    onFrame(frame);
  }

  function onPick(event: ChangeEvent<HTMLInputElement>) {
    const input = event.currentTarget;
    const file = input.files?.[0];
    input.value = "";
    if (!file || !file.type.startsWith("image/")) return;

    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") onFrame(reader.result);
    };
    reader.readAsDataURL(file);
  }

  return (
    <>
      <section className="relative min-h-[46svh] flex-1 overflow-hidden rounded-card border border-line bg-surface">
        {status !== "blocked" && (
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            className="absolute inset-0 h-full w-full object-cover"
          />
        )}

        {status === "blocked" ? (
          <div className="flex h-full flex-col items-center justify-center gap-3">
            <span className="flex h-14 w-14 items-center justify-center rounded-pill bg-card text-warn">
              <AlertIcon className="h-6 w-6" />
            </span>
            <p className="text-sm font-semibold text-muted">Camera off</p>
          </div>
        ) : (
          <>
            <div className="pointer-events-none absolute inset-0 bg-ink/20" />

            <div className="pointer-events-none absolute inset-0 flex items-center justify-center p-6">
              <div className="relative h-full w-full max-w-[300px]">
                {CORNERS.map((corner) => (
                  <span
                    key={corner}
                    className={cn("absolute h-9 w-9 border-accent", corner)}
                  />
                ))}

                {status === "starting" && (
                  <span className="absolute inset-0 flex items-center justify-center">
                    <ScanIcon className="h-8 w-8 animate-pulse text-muted" />
                  </span>
                )}
              </div>
            </div>

            <div className="pointer-events-none absolute inset-x-0 bottom-4 flex justify-center">
              <motion.span
                key={side}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.24, ease: "easeOut" }}
                className="rounded-pill bg-ink/75 px-4 py-2 text-xs font-semibold backdrop-blur-md"
              >
                {side === "front" ? "Front of pack" : "Back of pack"}
              </motion.span>
            </div>
          </>
        )}

        {flash > 0 && !reduced && (
          <motion.span
            key={flash}
            initial={{ opacity: 0.7 }}
            animate={{ opacity: 0 }}
            transition={{ duration: 0.45, ease: "easeOut" }}
            className="pointer-events-none absolute inset-0 bg-fg"
          />
        )}
      </section>

      <footer className="flex flex-col items-center gap-4 pb-[max(1.25rem,env(safe-area-inset-bottom))]">
        <div className="flex items-center gap-2">
          {(["front", "back"] as ScanSide[]).map((slot) => (
            <ShotSlot
              key={slot}
              src={shots[slot]}
              active={side === slot}
              filled={Boolean(shots[slot])}
            />
          ))}
        </div>

        <div className="mx-auto flex w-full max-w-[300px] items-center justify-between">
          <Button
            aria-label="Choose a photo"
            onClick={() => fileRef.current?.click()}
            className={ROUND_BUTTON}
          >
            <ImageIcon className="h-5 w-5" />
          </Button>

          <Button
            aria-label="Take the photo"
            onClick={shoot}
            className="flex h-[76px] w-[76px] items-center justify-center rounded-pill border-[3px] border-fg/70 transition-transform duration-150 ease-out active:scale-90 disabled:opacity-40"
          >
            <span className="h-[58px] w-[58px] rounded-pill bg-accent" />
          </Button>

          <Button
            aria-label="Open the camera app"
            onClick={() => nativeRef.current?.click()}
            className={ROUND_BUTTON}
          >
            <CameraIcon className="h-5 w-5" />
          </Button>
        </div>

        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          tabIndex={-1}
          onChange={onPick}
          className="sr-only"
        />
        <input
          ref={nativeRef}
          type="file"
          accept="image/*"
          capture="environment"
          tabIndex={-1}
          onChange={onPick}
          className="sr-only"
        />
      </footer>
    </>
  );
}

function ShotSlot({
  src,
  active,
  filled,
}: {
  src: string | null;
  active: boolean;
  filled: boolean;
}) {
  return (
    <span
      className={cn(
        "h-11 w-11 overflow-hidden rounded-[14px] border",
        filled
          ? "border-accent/70 bg-card"
          : active
            ? "border-dashed border-accent/50 bg-card/40"
            : "border-dashed border-line bg-card/40",
      )}
    >
      {src && <Shot src={src} />}
    </span>
  );
}

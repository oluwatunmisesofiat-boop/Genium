/**
 * Camera frame helpers. Everything here runs in the browser on the frames the
 * user just captured — no upload, no server round-trip.
 */

export type ScanSide = "front" | "back";

export type ScanShots = Record<ScanSide, string | null>;

/** Frames are downscaled before they hit state so low-end phones stay smooth. */
const MAX_EDGE = 960;

/** Sample grid for the lighting check. */
const SAMPLE = 48;

/**
 * Mean luminance (0–255) below this reads as "too dark to check". Kept lenient
 * on purpose: a false rejection is worse than a false pass in a demo.
 */
const MIN_LUMA = 26;

export function captureFrame(video: HTMLVideoElement): string | null {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) return null;

  const scale = Math.min(1, MAX_EDGE / Math.max(width, height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);

  const context = canvas.getContext("2d");
  if (!context) return null;

  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.82);
}

function loadImage(src: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => resolve(null);
    image.src = src;
  });
}

/**
 * Returns false when a frame can't be decoded or is too dark to read.
 * Simulations aside, this is a real check on the pixels that were captured.
 */
export async function inspectFrame(src: string): Promise<boolean> {
  const image = await loadImage(src);
  if (!image) return false;

  const canvas = document.createElement("canvas");
  canvas.width = SAMPLE;
  canvas.height = SAMPLE;

  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return true;

  context.drawImage(image, 0, 0, SAMPLE, SAMPLE);

  let total = 0;
  try {
    const { data } = context.getImageData(0, 0, SAMPLE, SAMPLE);
    for (let i = 0; i < data.length; i += 4) {
      total += 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
    }
  } catch {
    return true;
  }

  return total / (SAMPLE * SAMPLE) >= MIN_LUMA;
}

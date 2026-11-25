/**
 * Generates every Genium icon from one geometric definition.
 *
 *   pnpm -C frontend icons
 *
 * Runs with zero dependencies (node:zlib + node:fs), so the icons can be
 * regenerated on any machine without pulling a rasteriser into the project.
 */

import { deflateSync } from "node:zlib";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  LOGO_ACCENT,
  LOGO_ACCENT_DIM,
  LOGO_BACK,
  LOGO_CARD,
  LOGO_FRONT,
  LOGO_GRID,
  LOGO_INK,
} from "../lib/brand.ts";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

const INK = LOGO_INK;
const ACCENT = LOGO_ACCENT;
const ACCENT_DIM = LOGO_ACCENT_DIM;

/**
 * The mark: two portrait cards fanned apart — the front and the back of a pack,
 * drawn exactly the way the scan flow and the result screen tilt their frames.
 * Geometry lives in `lib/brand.ts` so the icons and `LogoMark` stay in sync.
 */
const GRID = LOGO_GRID;
const CARD = LOGO_CARD;
const BACK = LOGO_BACK;
const FRONT = LOGO_FRONT;

/** Rounded plate used for the non-maskable icons (23% radius). */
const PLATE_RADIUS = 118;

/** Keeps the mark inside the maskable safe circle (centred 80% diameter). */
export const MASKABLE_SCALE = 0.84;

/* ------------------------------------------------------------------ shapes */

function hexToRgb(hex) {
  const value = parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

const CARD_RGB = hexToRgb(ACCENT);
const DIM_RGB = hexToRgb(ACCENT_DIM);
const INK_RGB = hexToRgb(INK);

const scaled = (point, factor) => ({
  x: GRID / 2 + (point.x - GRID / 2) * factor,
  y: GRID / 2 + (point.y - GRID / 2) * factor,
});

function shapes(factor = 1) {
  const back = scaled(BACK, factor);
  const front = scaled(FRONT, factor);

  const card = (point, tilt, rgb) => ({
    cx: point.x,
    cy: point.y,
    width: CARD.width * factor,
    height: CARD.height * factor,
    radius: CARD.radius * factor,
    rotate: tilt,
    rgb,
  });

  return [
    card(back, -CARD.tilt, DIM_RGB),
    card(front, -CARD.tilt, CARD_RGB),
  ];
}

/* -------------------------------------------------------------------- raster */

/**
 * Paints one rotated rounded rectangle into an RGBA buffer, sampling 4×4 points
 * per pixel for edge anti-aliasing (cheap, and good enough at icon sizes).
 */
function paint(buffer, size, shape) {
  const k = size / GRID;
  const cx = shape.cx * k;
  const cy = shape.cy * k;
  const width = shape.width * k;
  const height = shape.height * k;
  const radius = Math.min(shape.radius * k, width / 2, height / 2);
  const halfW = width / 2;
  const halfH = height / 2;

  const theta = (shape.rotate * Math.PI) / 180;
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);
  const reach = Math.hypot(width, height) / 2 + 1;

  const x0 = Math.max(0, Math.floor(cx - reach));
  const x1 = Math.min(size - 1, Math.ceil(cx + reach));
  const y0 = Math.max(0, Math.floor(cy - reach));
  const y1 = Math.min(size - 1, Math.ceil(cy + reach));

  const STEPS = 4;
  const unit = 1 / (STEPS * STEPS);
  const innerX = halfW - radius;
  const innerY = halfH - radius;

  for (let y = y0; y <= y1; y += 1) {
    for (let x = x0; x <= x1; x += 1) {
      let hits = 0;

      for (let sy = 0; sy < STEPS; sy += 1) {
        for (let sx = 0; sx < STEPS; sx += 1) {
          const px = x + (sx + 0.5) / STEPS - cx;
          const py = y + (sy + 0.5) / STEPS - cy;

          // Rotate the sample into the card's own axes.
          const lx = px * cos + py * sin;
          const ly = -px * sin + py * cos;

          if (Math.abs(lx) > halfW || Math.abs(ly) > halfH) continue;

          const dx = Math.max(Math.abs(lx) - innerX, 0);
          const dy = Math.max(Math.abs(ly) - innerY, 0);
          if (dx * dx + dy * dy <= radius * radius) hits += 1;
        }
      }

      if (hits === 0) continue;
      blend(buffer, (y * size + x) * 4, shape.rgb, hits * unit);
    }
  }
}

function blend(buffer, i, rgb, alpha) {
  const under = buffer[i + 3] / 255;
  const out = alpha + under * (1 - alpha);
  if (out <= 0) return;

  for (let c = 0; c < 3; c += 1) {
    buffer[i + c] = Math.round(
      (rgb[c] * alpha + buffer[i + c] * under * (1 - alpha)) / out,
    );
  }
  buffer[i + 3] = Math.round(out * 255);
}

function plate(buffer, size, { radius, rgb }) {
  paint(buffer, size, {
    cx: GRID / 2,
    cy: GRID / 2,
    width: GRID,
    height: GRID,
    radius,
    rotate: 0,
    rgb,
  });
}

export function render(size, { bleed = false, markScale = 1 } = {}) {
  const buffer = new Uint8Array(size * size * 4);

  // Full bleed for maskable / iOS, rounded plate for everything else.
  plate(buffer, size, {
    rgb: INK_RGB,
    radius: bleed ? 0 : (PLATE_RADIUS / 512) * GRID,
  });

  for (const shape of shapes(markScale)) paint(buffer, size, shape);

  return buffer;
}

/* ---------------------------------------------------------------------- png */

const CRC_TABLE = (() => {
  const table = new Int32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[n] = c;
  }
  return table;
})();

function crc32(bytes) {
  let c = -1;
  for (let i = 0; i < bytes.length; i += 1) {
    c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ -1) >>> 0;
}

function chunk(type, data) {
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length, 0);

  const body = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body), 0);

  return Buffer.concat([length, body, crc]);
}

function encodePng(size, rgba) {
  const header = Buffer.alloc(13);
  header.writeUInt32BE(size, 0);
  header.writeUInt32BE(size, 4);
  header[8] = 8; // bit depth
  header[9] = 6; // truecolour + alpha
  header[10] = 0;
  header[11] = 0;
  header[12] = 0;

  const stride = size * 4;
  const raw = Buffer.alloc((stride + 1) * size);
  for (let y = 0; y < size; y += 1) {
    raw[y * (stride + 1)] = 0; // filter: none
    Buffer.from(rgba.buffer, y * stride, stride).copy(raw, y * (stride + 1) + 1);
  }

  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", header),
    chunk("IDAT", deflateSync(raw, { level: 9 })),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

function png(size, options) {
  return encodePng(size, render(size, options));
}

/* ---------------------------------------------------------------------- ico */

/** ICO container with PNG entries — supported by every browser we target. */
function ico(sizes, options) {
  const images = sizes.map((size) => ({ size, data: png(size, options) }));

  const directory = Buffer.alloc(6);
  directory.writeUInt16LE(0, 0);
  directory.writeUInt16LE(1, 2);
  directory.writeUInt16LE(images.length, 4);

  let offset = 6 + images.length * 16;
  const entries = images.map(({ size, data }) => {
    const entry = Buffer.alloc(16);
    entry[0] = size >= 256 ? 0 : size;
    entry[1] = size >= 256 ? 0 : size;
    entry.writeUInt16LE(1, 4);
    entry.writeUInt16LE(32, 6);
    entry.writeUInt32LE(data.length, 8);
    entry.writeUInt32LE(offset, 12);
    offset += data.length;
    return entry;
  });

  return Buffer.concat([directory, ...entries, ...images.map((i) => i.data)]);
}

/* ---------------------------------------------------------------------- svg */

export function svg({ bleed = false } = {}) {
  const k = 512 / GRID;
  const factor = bleed ? MASKABLE_SCALE : 1;
  const round = (value) => Number(value.toFixed(2));

  const back = scaled(BACK, factor);
  const front = scaled(FRONT, factor);
  const width = CARD.width * factor * k;
  const height = CARD.height * factor * k;
  const radius = CARD.radius * factor * k;

  const card = (point, tilt, fill) =>
    `<rect x="${round(point.x * k - width / 2)}" y="${round(
      point.y * k - height / 2,
    )}" width="${round(width)}" height="${round(
      height,
    )}" rx="${round(radius)}" fill="${fill}" transform="rotate(${tilt} ${round(
      point.x * k,
    )} ${round(point.y * k)})"/>`;

  const base = bleed
    ? `<rect width="512" height="512" fill="${INK}"/>`
    : `<rect width="512" height="512" rx="${PLATE_RADIUS}" fill="${INK}"/>`;

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512" role="img" aria-label="Genium">
${base}
${card(back, -CARD.tilt, ACCENT_DIM)}
${card(front, -CARD.tilt, ACCENT)}
</svg>
`;
}

/* -------------------------------------------------------------------- write */

export function writeAll() {
  const icons = join(ROOT, "public", "icons");
  mkdirSync(icons, { recursive: true });

  writeFileSync(join(ROOT, "app", "icon.svg"), svg());
  writeFileSync(join(icons, "icon.svg"), svg());
  writeFileSync(join(icons, "icon-maskable.svg"), svg({ bleed: true }));

  writeFileSync(join(icons, "icon-192.png"), png(192));
  writeFileSync(join(icons, "icon-512.png"), png(512));
  writeFileSync(
    join(icons, "icon-maskable-192.png"),
    png(192, { bleed: true, markScale: MASKABLE_SCALE }),
  );
  writeFileSync(
    join(icons, "icon-maskable-512.png"),
    png(512, { bleed: true, markScale: MASKABLE_SCALE }),
  );
  writeFileSync(
    join(icons, "apple-touch-icon.png"),
    png(180, { bleed: true, markScale: 0.9 }),
  );

  writeFileSync(
    join(ROOT, "app", "favicon.ico"),
    ico([16, 32, 48], { bleed: true, markScale: 0.9 }),
  );
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  writeAll();
  console.log("genium icons written");
}

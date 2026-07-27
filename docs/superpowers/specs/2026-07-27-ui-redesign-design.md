# UI redesign — modern, safe-area-aware shell

Date: 2026-07-27 · Status: approved (card layout, system theme)

## Problem

The shell declares `viewport-fit=cover` and a translucent status bar, so installed
(standalone) the OS extends the page under the cutout and home indicator — but no
element pads for those zones:

- The `Plate Reader` title sits in the status-bar strip; the Dynamic Island / notch
  covers it and the clock overlaps the text.
- The action bar's last 16px sit under the iOS home indicator / Android gesture bar.
- Roughly half the screen is dead space in every state; emoji icons (📷 🖼) and flat
  gray panels read as a wireframe.
- Browsers hide all of this behind their own chrome, which is why it survived testing.

## Decisions (user-approved)

1. **Card layout** — single-screen flow kept; three safe zones:
   header / content / action bar. No immersive full-bleed photo (no live viewfinder
   exists; boxes must never hide under a cutout).
2. **Cutout strip stays empty** — no UI in the status-bar strip, ever. The header
   begins below `env(safe-area-inset-top)`.
3. **Follow system theme** — light + dark palettes via `prefers-color-scheme`.
   The status strip + header keep a dark surface in both themes because iOS
   standalone always renders status text white over `black-translucent`;
   a light strip would make the clock invisible. Android gets per-theme
   `theme-color` metas.

## Layout spec

- `#shell` — `padding-left/right: max(16px, env(safe-area-inset-left/right))`
  (landscape: island/notch moves to the side).
- `#topbar` — `padding-top: calc(env(safe-area-inset-top) + 8px)`; dark surface in
  both themes; title left; green **On-device** badge (lock icon) right — replaces
  the buried "nothing is uploaded" sentence.
- `#photo-panel` — rounded card; unchanged tap-testing contract
  (`photo-view.ts` untouched; canvas keeps intrinsic-ratio sizing).
- `#plates-panel` — per candidate, a **meta row** above the card:
  index chip (matches the box number on the photo) · confidence % · country
  right-aligned. Replaces the floating yellow badge and the separate
  `Country:` line. Non-DE reads keep the neutral chip and get
  "check manually" instead of a country.
- `#action-bar` — `padding-bottom: calc(env(safe-area-inset-bottom) + 12px)`;
  primary pill "Take photo" with inline SVG camera icon; square gallery icon
  button; scroll fade above the bar. Emoji removed.
- Empty state — centered camera glyph + "Scan a license plate" +
  "Photos are analyzed on your phone — nothing leaves it."
- Landscape media query — photo card and results side by side.

## Out of scope

Pipeline, models, eval, plate-card anatomy (FE-Schrift face, EU band, seals —
kept as-is), in-place editing behavior, service worker.

## Verification

- Existing vitest suite passes; `npm run build` clean.
- Playwright screenshot harness re-run at iPhone standalone (393×852),
  iPhone browser (393×665), Android standalone (412×915) in empty + results
  states; visually inspected against the approved mockups
  (artifact `4efe5d22`, label `initial-proposal`).

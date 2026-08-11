# Third-party assets

## German license-plate font — GL-Nummernschild-Mtl

- **File:** `src/web/fonts/GL-Nummernschild-Mtl.ttf`
- **What it is:** a digitization of FE-Schrift ("Fälschungserschwerende Schrift"),
  the mandatory typeface for German vehicle registration plates (designed by
  Karlgeorg Hoefer, introduced 1994/2000). This is the "Mittelschrift"
  (medium/standard width) variant.
- **Source:** https://github.com/Gutenberg-Labo/GL-Nummernschild
  (raw file: `fonts/ttf/GL-Nummernschild-Mtl.ttf`, downloaded 2026-07-16)
- **License:** per the repository's `LICENSE.txt` (Copyright (C) 2009-2026
  Gutenberg Labo):

  > These fonts are free softwares.
  > Unlimited permission is granted to use, copy, and distribute it, with or
  > without modification, either commercially and noncommercially.
  > THESE FONTS ARE PROVIDED "AS IS" WITHOUT WARRANTY.

  This permits redistribution (including bundling in this repository and
  serving it same-origin from the app), which is why it was chosen over
  commercial FE-Schrift digitizations (e.g. Lineto's LL FE Schrift) that do
  not grant redistribution rights.
- **Used for:** `.plate-card` text (`src/web/ui.css`, `@font-face` block),
  self-hosted — no CDN, no external font request at runtime.

## UI font — Inter Variable

- **Files:** `src/web/fonts/inter-latin-wght-normal.woff2` (variable weights
  100–900, latin subset), license text in `src/web/fonts/Inter-LICENSE-OFL.txt`
- **What it is:** Inter by Rasmus Andersson — the mandatory typeface of the
  FLEXOPTIX brand system (see `flexoptix-ai-brand-system/DESIGN.md`); the
  subsetted woff2 is bundled with the brand package (Fontsource build).
- **License:** SIL Open Font License 1.1 (permits bundling and self-hosted
  redistribution; full text alongside the font file).
- **Used for:** all UI chrome typography (`src/web/ui.css`, `@font-face`
  block), self-hosted — no CDN, no external font request at runtime.

## Brand assets — FLEXOPTIX

- **Files:** inline eyecon SVG in `index.html`, `public/favicon.svg`,
  `public/icon-*.png` (launcher icons composed from the official eyecon plus
  a license-plate element by `scripts/make-app-icons.mjs`)
- **Source:** FLEXOPTIX brand package (`flexoptix-ai-brand-system/logos/`),
  used verbatim / exported at size — never redrawn.
- **License:** © FLEXOPTIX GmbH, internal use.

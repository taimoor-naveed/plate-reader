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

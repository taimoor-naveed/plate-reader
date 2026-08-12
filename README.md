# ANPR — plate extraction (stage 1) + owner matching (stage 2)

Reads German license plates from phone photos, fully on-device (PWA, ONNX in WebAssembly).
Photos never leave the device. Spec: docs/superpowers/specs/2026-07-15-plate-extraction-design.md

Stage 2 adds owner matching: the app can fetch a plates→owner list from a
user-configured URL (settings gear), cache it in localStorage, and show the
owner name(s) under a matched read — a plate may be listed for several
people, all are shown. That fetch is the app's ONLY cross-origin request,
it is optional (the app works fully without a list), and photos still
never leave the device. (Messaging the owner via Element is planned but
not wired up yet; the schema's matrixId field is reserved for it.)

## Setup

    npm install
    npm run fetch-models   # downloads ONNX models + copies ORT wasm (once)
    npm test               # unit tests
    npm run eval           # accuracy eval over attachments/ (needs eval/expected.json)
    npm run dev            # dev server (LAN-exposed; open on phone via Mac's IP)

## Install & offline

The deployed app lives at https://taimoor-naveed.github.io/plate-reader/
(GitHub Pages — pushes to `main` build and deploy automatically via
`.github/workflows/deploy.yml`; one-time prerequisite: the repo's
Settings → Pages must have **Source: GitHub Actions**). It's a PWA: open
that URL once with a network connection, then:

- **Install it** so it feels like a real app, not a browser tab:
  - **iOS (Safari):** Share → "Add to Home Screen". Launches full-screen,
    no browser chrome (via the `apple-mobile-web-app-capable` /
    `apple-touch-icon` tags in `index.html`).
  - **Android (Chrome):** you'll get an install prompt ("Add to Home
    screen" / "Install app"), driven by `public/manifest.webmanifest`
    (`display: standalone`).
- **Offline after the first visit:** a service worker (`public/sw.js`,
  stamped at build time by `scripts/stamp-sw.mjs` with the complete asset
  list of that exact build) precaches everything the app needs — shell,
  JS/CSS/wasm, both ONNX models (~11 MB), manifest, icons — during the
  first visit. From then on the app works fully offline; a server is only
  needed to pick up updates.
- **Updates:** a new deploy changes `sw.js` (its cache name embeds a hash
  of the build's contents). The update is fetched on the next load *with
  network* and becomes active on the launch after that; old caches are
  cleaned up automatically. Offline launches keep using the cached build
  indefinitely — there is no expiry.
- **iOS cache-eviction caveat:** iOS may evict an installed PWA's cache if
  it goes unused for an extended period (roughly a couple of weeks). If
  offline launch ever fails to load photos/models after a long gap, open
  it once online to restore the cache.

### HTTPS for local install/offline testing

Service workers (and the install prompt) require a secure context — plain
`http://<lan-ip>` on your phone won't do. To test install/offline from
your own machine before relying on the Pages deployment:

    brew install mkcert
    mkcert -install                  # once, trusts a local root CA on this Mac
    mkcert <your-lan-ip>             # e.g. mkcert 192.168.1.23 — writes a cert+key pair

Point Vite's dev/preview server at the generated cert (e.g. via
`server.https` / `preview.https` in `vite.config.ts` with the mkcert
`.pem`/`-key.pem` paths), then on your phone: open `chrome://flags` (or
Safari settings) once to trust mkcert's root CA profile — usually done by
visiting `http://<lan-ip>:<port>/` isn't enough; install the mkcert root
CA on the phone (AirDrop/email the `rootCA.pem` from
`mkcert -CAROOT`, then Settings → General → VPN & Device Management →
install profile → enable full trust for the mkcert cert in Certificate
Trust Settings). After that, `https://<lan-ip>:<port>` on the phone is a
secure context and behaves exactly like the deployed Pages URL.

For a team deployment instead of GitHub Pages: serve `dist/` (static
files only) from any company-internal HTTPS host — no server-side logic
required.

## Plates list (owner matching)

The list is served as a single `plates.json` from a URL the user enters in
the app's settings — the URL is never hardcoded here. Schema (fake data):

```json
{
  "version": 1,
  "people": [
    { "name": "Jane Doe", "matrixId": "@jane:matrix.example.com", "plates": ["BN CR 788", "TÖL AB 123"] },
    { "name": "John NoChat", "plates": ["M X 1"] }
  ]
}
```

`version` must be 1; `matrixId` is optional and currently unused (reserved
for the later Element-messaging stage); plates may be written in any human
format — the app normalizes (case, separators, umlaut folding) before
matching, and the E/H registration suffix (electric/historic) is ignored on
both sides, since list entries often omit it. A plate listed for several
people matches all of them. On a match the card is re-rendered with the
plate as written in the list — the list's spelling is authoritative, which
also resolves ambiguous district splits (DA-T 295 vs D-AT 295). A read
with no list entry shows a quiet "not in the plates list" note.
The app auto-refreshes
the list when it is missing or older than 7 days and has a force-update
button in the settings panel; while the server is unreachable it keeps
using the cached copy and says so.

Serving constraints: the response needs `Access-Control-Allow-Origin`
(cross-origin fetch), and the server must be **HTTPS** — the deployed PWA
is HTTPS and browsers block mixed content, so a plain `http://<lan-ip>`
server is only reachable from the `npm run dev` page, not from the
installed app. The reference dev server (plus its Tailscale-based HTTPS
exposure) lives outside this repo, next to the real list.

## Privacy

attachments/ (colleague car photos) and eval/expected.json (their plate numbers)
are gitignored — never commit them. The same applies to the plates→owner
list: real names, plate numbers, Matrix IDs, and the list server's URL live
only on the serving machine (outside this repo) and in the phone's
localStorage — never in this repository. The iOS eviction caveat above
extends to localStorage: an evicted PWA loses the saved URL and list
together and returns to its first-launch state.

## Status (MVP, 2026-07-16)

Verified on the user's phone over LAN (dev server): model preload, live camera
reads, multi-plate photos, tap-to-select, in-place editing. On-device latency
well within budget. UI iterated live to the minimal app-shell design
(photo + rectangles, authentic plate cards, no page scroll). Remaining known
misses (4/34 labeled plates) are documented in docs/eval-results.md; retake
is the designed recourse.

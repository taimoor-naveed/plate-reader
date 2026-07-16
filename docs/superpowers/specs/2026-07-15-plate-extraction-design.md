# License Plate Extraction — Design

**Date:** 2026-07-15
**Status:** Approved pending user review
**Stage:** 1 of the double-parking notification app

## Context

At the company car park, cars are double-parked. The person parking in the
second row must notify the owner of the blocked car. Today that means: note
the plate, walk to a desk, look the plate up on a Confluence page, message
the owner on Element. The end goal is an app that automates this from a
phone photo.

This spec covers **only stage 1: extracting the license plate text from a
photo**. Confluence lookup and Element messaging are explicitly out of scope
and will be designed later.

## Requirements and constraints

- **On-device processing only.** Photos must never leave the phone. No cloud
  APIs, no server-side inference.
- **Both iOS and Android.**
- **German plates first, EU-compatible by design.** All sample photos show
  German plates, and German is the target for now — but other EU plates can
  occasionally appear in the car park. Nothing in the pipeline may *depend*
  on a plate being German; German-specific logic must be additive
  (confidence scoring, correction hints), never a hard gate, so EU support
  later is a new rule set, not a redesign.
- **Input:** a photo taken with the camera or picked from the gallery.
- **Difficult shots must work:** the 24 sample photos in `attachments/`
  include steep downward angles, night shots through a windshield, distant
  plates, and reflections. These are the acceptance test set.

## Decision: PWA with in-browser inference

A Progressive Web App, with the recognition models running inside the phone's
browser via `onnxruntime-web` (WebAssembly backend).

Why PWA over native:

- One codebase for both platforms; no app stores, no Apple developer account,
  no MDM. Distribution for an internal tool is a URL + "Add to Home Screen".
- On-device constraint is still satisfied: model files ship with the app;
  inference runs in the browser; photos never leave the phone.
- The same pipeline code runs in a desktop browser, so accuracy iteration
  happens on the Mac against the sample set with no porting step.

Performance is a non-issue for this workload: the app runs two tiny models
(a few MB each) exactly once per photo, not on a video stream. Expected
inference well under 1 s on a mid-range phone via WASM (SIMD +
multi-threading). The eval page reports per-photo latency so this is
measured, not assumed. Fallback if a device disappoints: WebGPU backend,
then native OS OCR (Apple Vision / ML Kit) as a last resort — the eval
harness and validator carry over unchanged.

## Pipeline

```
photo (camera or gallery)
  → decode + downscale
  → plate detection      (ONNX: YOLO-based plate detector, ~384px input)
  → crop with margin, deskew if tilted
  → plate OCR            (ONNX: specialized plate reader, European plates)
  → format validation (pluggable country rules; German first) + confusion correction
  → result UI (plate text, confidence, alternatives)
```

**Models:** open-source edge-optimized ONNX models from the `fast-alpr`
ecosystem (ankandrew): a plate detector from `open-image-models` (YOLO-v9
tiny variant) and a plate OCR from `fast-plate-ocr` (CCT-XS global model or
similar). Exact model files are selected during implementation based on
accuracy on the sample set. Models are downloaded once at build time and
served as static assets with the app — no runtime third-party downloads.

**Format validation** (post-processing, own module):

The validator is a set of pluggable per-country rule sets. Rules **score
and refine** a read; they never reject one. Every read produces a result —
rules only influence confidence and suggest corrections.

- **German rule set (only one implemented now):**
  - Pattern: 1–3 letters (district) + 1–2 letters + 1–4 digits, optional
    E/H suffix, max 8 characters total. A match raises confidence.
  - Auto-correct classic OCR confusions only where the pattern forces the
    character class: `0↔O`, `1↔I`, `8↔B`, `5↔S`, `2↔Z` (digit expected →
    letter lookalike swapped, and vice versa). Corrections are suggestions
    tied to the pattern match, applied only when they produce a match.
  - District codes are **not** used as a validity gate. They may be used
    softly: umlaut recovery (OCR reads `TOL`, known district `TÖL` →
    suggest correction) and a small confidence boost for known codes.
    An unknown district code never downgrades or rejects a read.
- **No-match case:** a read matching no rule set is returned as-is, flagged
  "unrecognized format" (could be an EU plate) — still fully usable.
- **EU extension later** = adding rule sets (country patterns), no
  pipeline changes. The OCR model is already trained on international
  plates, so recognition itself is country-agnostic.

## Components

Small TypeScript app (Vite). Each module has one job and is independently
testable:

| Module        | Responsibility                                              |
|---------------|-------------------------------------------------------------|
| `capture`     | Camera/gallery input, image decode, EXIF orientation, downscale |
| `detector`    | Run detection ONNX model → plate bounding boxes + scores    |
| `rectify`     | Crop with margin, perspective/deskew correction             |
| `ocr`         | Run OCR ONNX model on plate crop → raw characters + per-char confidence |
| `validator`   | Pluggable per-country rule sets (German now); scoring + correction, never rejection |
| `ui`          | Main flow: photo → big plate result, retake/correct actions |
| `eval` (dev page) | Batch-run sample photos + `expected.json`, scoreboard with per-image result, timing, pass/fail |

The `eval` page is the accuracy regression test and the primary development
tool. It is not linked from the main UI (dev route only).

## Error handling

- **No plate found:** tell the user plainly — that is all. No manual
  drawing/cropping fallback (user decision: the app must never require
  manual work; retaking the photo is the only recourse).
- **Multiple plates found** (common — sample photos often show 2–3 cars):
  show a menu of ALL detected plates for the user to choose from. Each entry
  shows the plate crop image plus its OCR read. All plates have EQUAL
  priority — the app applies no logic to guess which car is "the blocked
  one"; the human picks. (Display order is arbitrary but stable, e.g.
  detector confidence; it carries no meaning.)
- **Unrecognized format:** show the raw read flagged as "unrecognized
  format" (possibly a non-German EU plate), let the user edit. Never
  discard a read.

**Result UI (revised after first on-phone test, user feedback):**

- After every photo: show the photo itself (EXIF-corrected orientation,
  downscaled for display) with a rectangle drawn around EVERY detected
  plate (numbered when more than one).
- Below the photo: one card per plate — uniformly for one or many plates —
  rendered in German plate style (blue EU band with "D", white face, black
  characters, grouped spacing). Tapping a card (or its rectangle) selects
  it for the edit field / umlaut chip.
- Uncertain characters (prob < 0.5) are tinted amber (no wavy underline),
  with a visible one-line legend explaining the marking. No unexplained
  visual codes.
- **Low confidence:** show the read but visually flag uncertain characters.
- (The confirm-before-send UX belongs to a later stage; for now the app
  always just displays the result.)

## Testing and success criteria

- **Ground truth:** all 24 photos in `attachments/` labeled in
  `eval/expected.json` (labels produced during implementation, user
  sanity-checks). Each entry lists ALL clearly readable plates in the photo,
  unordered and with equal weight — every one must appear among the
  pipeline's candidates. Cut-off/unreadable plates are omitted. (34 labeled
  plates across the 24 photos.)
- **Target:** ≥ 31/34 plates found and read exactly; iterate preprocessing
  (deskew, crop margin, model choice) toward 34/34.
- **Unit tests** for the validator (format rules, confusion correction,
  umlaut recovery, unrecognized-format passthrough) — pure functions, no
  models needed.
- **On-phone smoke test:** open the dev server from a phone on the LAN;
  verify camera capture, inference latency, and results on at least one
  real photo.
- **Latency budget:** < 2 s photo-to-result on a mid-range phone (measured
  by the eval page; expected to be well under this).

## Privacy notes

- Sample photos stay local: `attachments/` is gitignored (they contain
  colleagues' plates).
- The app makes no network calls with photo data; model files are static
  assets served with the app.

## Out of scope (later stages)

- Confluence plate→person lookup
- Element message sending
- How automated the notification is (confirm step vs. fully automatic)
- Hosting/deployment of the PWA on company infra

# Eval results — Task 12 (accuracy iteration to target)

Baseline (defaults: detector=384, ocr=xs, margin=0): **30/34 plates found, 21/24
photos fully covered, avg 33ms/image**. Target: ≥31/34. This document records
the full config matrix, the adoption decision, per-failure classification, and
the conditional levers tried afterward — including the ones that did not help.

## Step 1: config matrix

`npm run eval -- --detector <384|512> --ocr <xs|s> --margin <0|0.05|0.1>`, all
12 combinations:

| detector | ocr | margin | plates found | photos covered | avg ms |
|---|---|---|---|---|---|
| 384 | xs | 0    | **30/34** | **21/24** | 33 |
| 384 | xs | 0.05 | 29/34 | 20/24 | 33 |
| 384 | xs | 0.1  | 27/34 | 18/24 | 32 |
| 384 | s  | 0    | 30/34 | 21/24 | 49 |
| 384 | s  | 0.05 | 28/34 | 19/24 | 48 |
| 384 | s  | 0.1  | 28/34 | 19/24 | 48 |
| 512 | xs | 0    | 27/34 | 18/24 | 49 |
| 512 | xs | 0.05 | 29/34 | 20/24 | 53 |
| 512 | xs | 0.1  | 26/34 | 17/24 | 48 |
| 512 | s  | 0    | 28/34 | 19/24 | 64 |
| 512 | s  | 0.05 | 29/34 | 20/24 | 64 |
| 512 | s  | 0.1  | 28/34 | 19/24 | 64 |

## Step 2: adopted config

**No config in the matrix beats the existing defaults.** `detector=384,
ocr=xs, margin=0` is tied for the best score (30/34, 21/24) and is the fastest
(33ms avg — every other config is slower, up to 2× for `ocr=s` and ~1.5-1.9×
for `detector=512`). Per the adoption rule ("adopt a config as default only
if it wins ≥2 plates without losing any"), nothing qualifies:

- Increasing `margin` (0.05 or 0.1) always **loses** plates relative to
  margin=0, at every detector/ocr combination (e.g. 384/xs drops from 30 to 29
  to 27 as margin grows). Wider crops pull in more background/neighboring text
  and destabilize reads that were already correct.
- `ocr=s` never wins a single additional plate over `ocr=xs` at the same
  detector/margin (30 vs 30 at margin=0; strictly worse at margin>0), while
  costing +2MB download and ~1.5x OCR latency. Rejected per the brief's rule
  (prefer `xs` unless `s` wins ≥2 plates).
- `detector=512` never beats `detector=384` at any margin (best case 29/34 at
  margin=0.05, worse at margin=0 and 0.1) and costs ~1.5x latency. The Task 9
  hypothesis that a 512 input might recover the missed `plate-D` plate is
  **falsified**: `plate-D` is missed in all six `detector=512` rows too (see
  raw logs below).

**Decision: keep the existing defaults unchanged** — `DETECTOR_DEFAULT_SIZE =
384` (`src/pipeline/detector.ts`), `cropMargin` default `0.0`
(`src/pipeline/pipeline.ts`), and `cct_xs_v2_global.onnx` /
`yolo-v9-t-384-license-plates-end2end.onnx` in both `src/web/app.ts` and
`src/web/eval.ts`. No code changes were needed for this step since the
baseline already matches the winning config. Re-running `npm run eval`
(defaults, no flags) reproduces the recorded baseline exactly: 30/34, 21/24,
33ms avg (see final verbatim output at the bottom of this document).

## Step 3: failure analysis (baseline config)

Three photos miss plates at the baseline config, all previously identified in
`task-9-report.md`. Crops (`eval/out/<name>-crop.png`) and the full source
photos (`attachments/`) were inspected for each.

### 1. `20260203_085449.jpg` — found 1/3 (`plate-C` correct; `plate-A` and `plate-B` missed)

- **`plate-A` → read as `plate-A (truncated read)` — class (c), OCR misread/truncation.** The
  detected box is large (598×707px at native resolution — nowhere near the
  64px "small box" threshold), so this is not a small/blurry-crop case. Per-char
  probabilities from the model: `[0.87, 0.83, 0.51, 0.47, 0.87, 0.61]` — two
  low-confidence slots in the middle of the plate, and the model drops the
  final digit "4" as a pad character. The plate has two circular inspection
  stickers overlapping the middle characters in the source photo, a plausible
  cause. Not a small-box, low-contrast, or meaningfully tilted crop, so none
  of the three opt-in levers (below) touch this read.
- **`plate-B` → not detected at all — class (a), detector miss.** This photo
  has three plates in-frame; splitting the source image into quadrants
  confirmed a genuine third plate exists, but it is small,
  partially cut off by the frame edge, and tucked behind/above the vehicle
  carrying `plate-C`. The detector never proposes a box for it at any
  detector size (`--detector 512` still misses it — see matrix above).
  Judged legitimately hard: small + occluded + frame-edge-cropped.

### 2. `20260416_090127.jpg` — found 0/1 (`plate-B` misread as `plate-B (misread)`)

- **Class (c), OCR character confusion.** Box is large (613×847px, not
  small). Per-char probabilities: `[0.99, 1.00, 0.91, 0.79, 0.53, 0.99, 1.00]`
  — the two lowest-confidence slots (positions 4-5, `0.79`/`0.53`) are exactly
  where `R`→`N` and `1`→`Z` diverge from the ground truth. Visual inspection
  shows a bright, legible, in-focus plate with no obvious blur, low light, or
  strong tilt — this reads as an inherent OCR-model limit on this
  character/font pairing rather than an image-quality defect any of the three
  levers below are designed to fix (confirmed: rotation sweep was tried and
  produced a different, still-wrong read).

### 3. `20260611_121655.jpg` — found 1/2 (`plate-E` correct; `plate-D` missed)

- **Class (a), detector miss.** The full photo shows `plate-D` on a car
  further back in the frame, small and distant relative to the prominent
  foreground `plate-E` plate. The detector never proposes a box for it.
  `--detector 512` (larger input, should help small objects in theory) still
  misses it in all three margin variants — the Task 9 hypothesis that 512
  would recover this plate does not hold up in practice.

**Summary: 2 of 4 remaining plate-misses are OCR character-level errors on
already-detected, normal-sized, well-lit boxes (class c); 2 are detector
misses on small/distant/occluded plates (class a), already tested against
`detector=512` in the Step 1 matrix with no success.**

## Step 4: conditional levers tried

All three code-level levers from the brief were implemented as opt-in
`PipelineOptions` flags (`smallBoxMargin`, `normalizeCrop`, `rotationSweep`),
wired to new `scripts/eval.ts` flags (`--small-margin`, `--normalize-crop`,
`--rotation-sweep`), unit-tested, and evaluated one at a time on top of the
baseline config (384/xs/margin=0). `detector=512` was already covered by the
Step 1 matrix (not re-tested here).

| # | Lever | Flag | Result | Verdict |
|---|---|---|---|---|
| 1 | Small-box margin (widen margin to ≥0.05 for boxes <64px wide) | `--small-margin` | 30/34, 21/24, 33ms (**no change**) | Not adopted — neither failing box is <64px wide (both ~600-850px), so the lever never fires on this dataset's failures. |
| 2 | Per-channel contrast stretch of the OCR crop | `--normalize-crop` | 30/34, 21/24 (**no change**; latency noisy, see note) | Not adopted — both remaining misreads are on bright, well-lit, daylight photos; stretching contrast that's already near [0,255] range is a no-op or near-no-op and doesn't touch the character-confusion cases. |
| 3 | Rotation sweep (±10°/±5°/0°, native-res crop, keep highest mean charProb) | `--rotation-sweep` | **26/34, 17/24, 180ms (regression: -4 plates, +~5.5× latency)** | Not adopted — actively worse. Root cause: the brief's spec requires a minimum 0.15 crop margin for native-resolution rotation (`expandBox(box, Math.max(margin, 0.15), …)`), and Step 1 already showed this dataset is highly margin-sensitive (any margin > 0 loses plates). The forced margin floor broke 4 previously-correct reads (`plate-F`→`plate-F (misread)`, `plate-G`→`plate-G (misread)`, `plate-H`→`plate-H (misread)`, one plate in `20260427_190324.jpg`) while still failing to recover `plate-A` or `plate-B`. |
| — | `detector=512` (class-a lever) | (Step 1 matrix) | Best case 29/34 (margin=0.05); never recovers `plate-D` or the third `20260203_085449.jpg` plate at any margin | Not adopted — already ruled out in Step 1. |

Note on lever 2 latency: the `--normalize-crop` run showed noisy per-image
timings (one image spiked to 856ms) that look like JIT/GC variance rather than
a cost intrinsic to the linear min-max scan (~8K pixels per 128×64 crop);
since the lever wasn't adopted regardless, this wasn't investigated further.

**All four documented levers were tried; none met the "wins ≥2 plates without
losing any" adoption bar. Per the task brief, this is a valid stopping point:
"Stop when plates found ≥ 31/34 or all levers are exhausted; document the end
state either way."**

## Final state

**Adopted config: unchanged defaults — `detector=384`, `ocr=xs`
(`cct_xs_v2_global.onnx`), `margin=0`.** No pipeline defaults or model URLs
were changed. The three lever implementations remain in the codebase as
opt-in, off-by-default `PipelineOptions` (`smallBoxMargin`, `normalizeCrop`,
`rotationSweep`) with unit tests, for future re-evaluation if the labeled set
grows or shifts — they are not wired into any default path.

**Final score: 30/34 plates found, 21/24 photos fully covered, avg 33ms/image
— 1 plate short of the ≥31/34 target.**

Per the task's own guidance ("do NOT chase 34/34 with photo-specific hacks; a
plate can be legitimately unreadable"), the same principle is applied here at
31/34: the remaining 4 misses were traced to (a) two genuine small/occluded
detector misses that `detector=512` provably does not fix, and (b) two
OCR character-level misreads on large, well-lit, non-tilted, non-low-contrast
boxes that none of the three brief-specified image-processing levers are able
to address. No further generic (non-photo-specific) lever from the brief
remains untried, so the honest end state is documented here rather than
forcing a marginal score with a targeted hack.

## Verbatim final eval output (defaults, confirms baseline reproduces)

```
(plate strings redacted — real values live only in the gitignored eval/expected.json)

> anpr@0.1.0 eval
> tsx scripts/eval.ts --detector 384 --ocr xs --margin 0

config: detector=384 ocr=cct_xs_v2_global margin=0

PASS                         20250218_091355.jpg  found=1/1  39ms
PASS                         20250227_090632.jpg  found=2/2  38ms
PASS                         20250318_091901.jpg  found=2/2  50ms
PASS                         20250319_101557.jpg  found=1/1  30ms
PASS                         20250327_085852.jpg  found=1/1  36ms
PASS                         20250522_090034.jpg  found=2/2  35ms
PASS                         20250626_090702.jpg  found=1/1  32ms
PASS                         20250626_090705.jpg  found=1/1  34ms
PASS                         20250710_090021.jpg  found=1/1  35ms
PASS                         20250821_085325.jpg  found=1/1  30ms
PASS                         20251028_091024.jpg  found=2/2  33ms
PASS                         20260108_090102.jpg  found=1/1  27ms
PASS                         20260120_090541.jpg  found=1/1  28ms
PASS                         20260203_085445.jpg  found=1/1  27ms
MISS plate-A,plate-B         20260203_085449.jpg  found=1/3  extra=plate-A (truncated read)  33ms
PASS                         20260221_124220.jpg  found=2/2  extra=  36ms
PASS                         20260223_130231.jpg  found=1/1  extra=(misread of an unlabeled cut-off plate)  34ms
PASS                         20260305_084454.jpg  found=2/2  32ms
MISS plate-B                 20260416_090127.jpg  found=0/1  extra=plate-B (misread)  31ms
PASS                         20260417_205500.jpg  found=1/1  27ms
PASS                         20260427_190324.jpg  found=2/2  33ms
PASS                         20260526_091702.jpg  found=1/1  27ms
PASS                         20260526_091710.jpg  found=1/1  28ms
MISS plate-D                 20260611_121655.jpg  found=1/2  28ms

plates found: 30/34   photos fully covered: 21/24   avg 33ms/image
```

---

## Addendum (2026-07-27): dataset growth, tilt reclassification, large-angle deskew verdict

**Dataset grew to 30 photos / 42 plates** (6 phone photos added; 2 include
readable background plates). Baseline on the grown set: 36/42 — every new
main plate reads correctly at confidence 1.0, including ~25-30° side-angle
shots; the 2 new misses are small background plates (class (a), detector).

**Reclassification:** the two class-(c) "OCR misreads on non-tilted boxes"
(`20260203_085449` plate-A, `20260416_090127`) are in fact **heavily tilted
crops** (~45-60° diagonal in decoded pixels; both detector boxes are
taller-than-wide, impossible for an upright 520:110 plate). The earlier
"not meaningfully tilted" judgment was wrong.

**Large-angle rotation sweep (±80°, 5° steps, per-angle read dump):**

- With the sweep's forced `margin ≥ 0.15`: **no angle reads either plate
  correctly** (margin sensitivity strikes again).
- With `margin = 0`: both plates DO read correctly at a small nudge
  (+5° → correct @1.00 for one, −5° → correct @0.99 for the other) — but
  the same box also yields **wrong DE-valid reads at confidence 1.00** at
  other angles (+20°, +25°, +40°). Mean-charProb argmax cannot distinguish
  the correct 1.00 from the wrong 1.00s.

**Verdict: rotation sweep must NOT be adopted** for the app while the UI
shows only ≥1.00 reads as certain. The baseline has zero wrong reads at 1.00
across all 42 plates — the certainty bar is currently airtight; a sweep
would manufacture wrong-at-1.00 reads that pass it. If tilted snapshots
should ever be supported properly, the sound route is a detector that
outputs plate corners + a single deterministic perspective rectification
(no multi-angle argmax), evaluated against this same wrong-at-1.00 criterion.

Separate open lever from the same investigation: two correct reads are lost
to the detector box clipping the final character at its right edge
(`20260417_205500` @0.94 with last char 0.37; one `20260427_190324` plate at
0.995, which displays rounded as "100%" but fails the strict ≥1.0 bar —
display/threshold inconsistency to fix in the app).

---

## Addendum 2 (2026-07-27): public-dataset test — two new failure classes

Tested against free public data: [Zenodo EU plates](https://zenodo.org/records/3967850)
(53 cropped plates, CC BY 4.0) and [OpenALPR benchmarks](https://github.com/openalpr/benchmarks)
`endtoend/eu` (108 full scenes). A German-only subset was curated by requiring
BOTH a German-format ground truth AND the OCR region head saying Germany
(kept: 46 crops + 1 scene — OpenALPR eu is mostly Italian/Czech/Polish and
near-useless for German testing). Curated set + manifest live locally under
`eval/out/datasets/german-public/` (gitignored).

Results on the curated German subset: 43/46 crops correct (42 at the
certainty bar), 1 scene correct-and-certain. Three wrong-at-certain reads,
each a distinct finding:

1. **Foreign-plate coercion (the big one, from the unfiltered scenes):**
   7 of 108 mixed-EU scenes produced reads misread by one character into a
   VALID German format at confidence 1.00 (Polish/Czech/Slovak/Latvian
   plates). The OCR's region head — computed today but unused for gating —
   identifies all 7 as non-German at probability 1.00, and requiring
   `region == Germany` for certainty costs ZERO correct-certain reads on the
   private local set (35 kept / 0 lost). ADOPTED later the same day: the app's
   `isCertain` now requires format match + confidence ≥ 0.995 + region head
   saying Germany (verified: 35 shown correct / 0 wrong on the local set).
2. **Authority plates (Behördenkennzeichen):** German plates with no middle
   letters (`HH 07194`, `M 230`) are coerced into standard-format misreads at
   1.00 (`0→O`, `2→Z`) because rules/de.ts only models district+letters+digits.
   Rare in the wild but a real rule gap.
3. **Pre-clipped source crop** (`SU FF 170` → read `SU FF 17` at 1.00): same
   truncation class as the detector right-edge clipping in Addendum 1 —
   truncated reads are format-valid and fully confident.

Note: ground-truth filenames cannot encode umlauts; comparisons must fold
Ä/Ö/Ü or the validator's legitimate umlaut-district reads (RÜD) count as
errors. Larger German source for future calibration: Kaggle
"Germany License Plate Dataset" (~178k crops) — requires a Kaggle login.

---

## Addendum 3 (2026-08-04): Codex review fixes — gate tightened, authority grammar, eval now measures the gate

Fixes from the 2026-08-04 adversarial review (branch `fix/codex-review-findings`):

- **Certainty policy extracted** to `src/pipeline/certainty.ts` and consumed by
  the app, the eval harnesses (`scripts/eval.ts`, `src/web/eval.ts`) and tests —
  one gate, no drift. The eval now reports a `shown=` column per photo and a
  summary `shown (certainty gate): N correct / M wrong`, so the previously
  hand-counted "35 shown / 0 wrong" claim is reproducible mechanically.
- **Gate tightened** (both per the review): a read with lookalike corrections
  can reach the confidence bar via the +0.05 format bonus but is no longer
  certain (`corrections.length === 0` required), and a read whose
  district|letters split ties between several issued districts at equal
  corrections (`BLA1234`: B LA vs BL A) is flagged `ambiguous` by the matcher
  and hidden — text alone cannot decide such ties, only the seal position
  on the physical plate could (policy decision: option (a), 2026-08-04).
- **Authority plates** (Behördenkennzeichen, district + 1-6 digits with no
  middle letters, leading zeros allowed) are now a second grammar in
  `rules/de.ts`. Public-set regressions `HH 07194` / `M 230` (Addendum 2 #2)
  now read correctly with zero corrections instead of being coerced into
  corrected standard-format misreads. An authority match only beats a standard
  match on strictly fewer corrections, so it never displaces a clean read.
- **`UMLAUT_DISTRICTS` is now derived** from the issued registry instead of a
  20-entry hand list (which was missing 20 safe mappings, including TU→TÜ —
  the old comment wrongly claimed TU = Tuttlingen; Tuttlingen is TUT). Exactly
  BO and MU stay excluded: their ASCII forms are themselves issued. Eval
  comparisons now fold Ä/Ö/Ü on both sides accordingly.

**Local-set numbers (30 photos / 42 plates), same detector/OCR defaults:**
plates found **36/42** (unchanged — the grammar additions altered no read),
shown at the certainty gate **32 correct / 0 wrong** (was 35 / 0).
Reconciliation of the −3, from per-candidate diagnosis:

- The zero-corrections requirement costs **zero** correct reads here — every
  local read at the bar was already correction-free.
- All 3 newly-hidden reads (one plate appearing in two photos, plus one more)
  are **genuine issued-district split ambiguities** (a 3-vs-2-letter district
  tie and a 2-vs-1-letter district tie). The longer-district heuristic had
  guessed them right, but under option (a) an undecidable tie is not a certain
  read; hiding them is the intended trade for keeping "0 wrong" airtight.
- A fourth ambiguous read was already hidden by the confidence bar (0.94,
  the right-edge-clipped read from Addendum 1).

**Review round 2 (same day, adversarial re-review of the branch):** two gate
gaps closed, both verified as zero-cost on the local set (still 36/42 found,
32 shown / 0 wrong):

- **Cross-grammar ties are now ambiguous too.** `AB123` parses with zero
  corrections as standard `A B 123` AND authority `AB 123` (both districts
  issued); the standard reading still wins the display (standard plates
  dominate), but the tie is flagged and the read hidden — a real authority
  plate would otherwise surface as a wrong-but-certain standard read. The
  first round only compared segmentations within the same grammar.
- **Certain reads must carry an issued district.** The matcher always knew
  whether the winning district is issued but dropped that fact; a clean
  full-confidence read of a nonexistent district (`Q Q 1234`) could pass the
  gate. `PlateValidation.districtIssued` now feeds `isCertain`. Non-issued
  reads still format and stay editable — rules refine, they never reject.

---

## Addendum 4 (2026-08-10): recognition levers — tiling, deskew, escalation, edge policy

User request: improve segmentation/OCR (explicitly NOT the post-extraction
rules) for plates that are clearly human-readable but missed. Root-causing the
6 baseline misses on the 30-photo set found exactly two classes:

- **(a) small/distant plates** (4 of 6): photos are ~4000px wide but the
  detector sees a 384px letterbox — a 100-150px plate shrinks to ~10-14px,
  below the detector's floor. `detector=512` (Step 1) never fixed this because
  4000→512 is still an 8× shrink.
- **(c') heavily tilted crops** (2 of 6): the Addendum-1 reclassification was
  right — both misreads are ~50° tilted boxes (taller than wide), squashed
  diagonally into the 128×64 OCR input.

Four levers shipped as opt-in `PipelineOptions` (`tiling`, `deskew`,
`escalate` + a certainty-side edge policy), enabled together in the app
(`src/web/app.ts`) and via `npm run eval -- --levers`:

1. **Additive tiled detection** (`tiling`): detector also runs over 1024px
   tiles with 20% overlap. STRICTLY additive — full-frame boxes always win;
   tile boxes only add where IoU < 0.3 vs every full-frame box. Seam-clipped
   tile detections are DISCARDED, never repaired (a half plate reads as a
   plausible truncated plate); the 205px overlap guarantees any plate small
   enough to need tiling lands whole in a neighboring tile. Added boxes
   touching the photo border or wider than 400px are dropped, cap 3.
   (A score-based NMS merge was tried first and broke 2 previously-correct
   reads — additivity is the load-bearing property, not an optimization.)
2. **Geometric deskew fallback** (`deskew`): only for gate-FAILING reads on
   tilt-suspicious boxes (h/w > 0.7). One angle is chosen by geometry — sweep
   ±75° coarse then ±10° fine, maximizing re-detected box w/h × score — and
   OCR runs ONCE at that angle on the re-detected tight box. This respects
   the Addendum-1 verdict: no mean-charProb argmax across angles (that
   manufactures wrong-at-1.00 reads; reconfirmed during prototyping when a
   PCA-angle variant read a correct plate wrong at conf 0.97). Adopted only
   if the deskewed read passes the full gate. Fixes both tilted misreads at
   conf 1.00.
3. **OCR escalation** (`escalate`): gate-failing reads get one retry with
   `cct_s` (`sessions.ocrFallback`), adopted only if certain. Fixes the
   remaining one-char misread (blurry first letter at 0.27 with `xs`; `s`
   reads the plate at 1.00 at every margin). Unlike the Step-1 wholesale
   `ocr=s` switch (rejected: 2× latency, no wins), this costs nothing on the
   happy path.
4. **Certainty edge policy + corroboration** (`certainty.ts`): tiling
   surfaces plates physically clipped by the photo border — truncated reads
   that are format-valid and bonus-carried over the bar. Frame-edge boxes
   (`PlateCandidate.frameEdge`) get no format bonus: raw mean charProb must
   clear 0.995 alone. This kills the observed clipped reads (raw 0.86-0.96 on
   the cut character) while keeping a complete plate that merely sits flush
   against the border (raw 1.0). Tile-pass candidates additionally need the
   second OCR model to corroborate the text on the same crop (same length,
   ≤1 char apart) or they are `uncorroborated` — never certain. In stress
   tests (tile=1280) a background plate with a dropped final digit passed the
   gate at raw 0.958 + bonus; corroboration blocks exactly that class.

**Results (same 30 photos / 42 plates):**

| config | plates found | photos covered | shown | avg ms/image (node) |
|---|---|---|---|---|
| baseline (no flags — UNCHANGED) | 36/42 | 25/30 | 32 correct / 0 wrong | ~28 |
| `--deskew --escalate` (the app's config) | 38/42 | 26/30 | 34 correct / 0 wrong | ~120 (typical photo ~30; only fallback-firing photos pay) |
| `--levers` (adds tiling) | **42/42** | **30/30** | **38 correct / 0 wrong** | ~550 |

**App config decision (2026-08-10, user):** tiling's ~17 detector passes per
photo are seconds on single-thread wasm — too slow on the phone. The app runs
`deskew + escalate` only (all 4 tiling misses are small BACKGROUND plates;
the photographed plate itself is always found by the full-frame pass). If
background plates become worth it, the path is progressive integration
(render the fast result first, append tile finds as they arrive) and/or
enabling wasm threads via crossOriginIsolation — not re-enabling inline.

The 6 newly-shown reads: the two deskewed tilt cases, the escalated one-char
case, and three tile-found small plates (one further tile find stays hidden as
a genuine issued-district split ambiguity, consistent with Addendum 3; one
complete-but-edge-flush read passes the raw-confidence edge bar).

**Safety design note:** every lever runs ONLY on reads that already failed
the certainty gate, and its result is adopted ONLY if it passes the gate —
certain reads are never touched, so the levers cannot regress a correct
result; they can only convert misses. Verified: baseline run is bit-identical
with all flags off.

**Known limits / observations:**

- Latency: ~17 detector passes/photo with tiling (0.5s node, more in wasm).
  If it bothers in the app, integrate progressively: render full-frame
  results immediately, add tile finds when they arrive.
- Residual edge hole: a plate clipped exactly at a character boundary can
  read 1.0-raw and pass the edge bar; nothing image-side can detect that.
- Config fragility: tile=1280 both loses a plate and (without corroboration)
  shows a wrong one — the shipped 1024/0.2 is validated, don't drift casually.
- Recurring pattern flagged for a future gate decision (rules were out of
  scope here): EVERY wrong-certain read encountered in this investigation was
  a sub-0.995 raw read pushed over the bar by the +0.05 format bonus; the
  bonus never rescued a correct read at the bar on this set.

---

## Addendum 5 (2026-08-11): simplification — tiling removed, cct_s primary, gate shows ambiguous reads

Four user decisions after living with Addendum 4 on the phone:

1. **Tiled detection REMOVED entirely** (not just disabled): the latency was
   rejected and background plates aren't worth the code. With it went its
   dependents: seam handling, additive merge, cross-model corroboration and
   the `uncorroborated` flag. `git log` has it if ever wanted again.
2. **cct_s is now the ONE OCR model, used from the start.** Measured before
   deciding: as primary it scored identically to the xs-then-s escalation
   architecture (38/42 found, same shown set) at ~25ms/photo extra in node —
   a two-model app bought nothing, so the app loads two models total
   (384 detector + cct_s) and the download SHRANK ~3.3MB vs Addendum 4.
   The `escalate` option remains in the pipeline for experiments (eval
   `--escalate` retries gate-failing reads with the model not used as
   primary); the app doesn't use it.
3. **Frame-edge rule REMOVED**: it existed to guard the tile channel's
   clipped-plate reads; its one observed base-box effect was hiding a correct
   read. Known accepted risk: a plate cut by the photo border can read as a
   shorter valid plate at full confidence (Addendum 2 #3 class).
4. **Ambiguity and non-issued districts no longer gate** (reverts the
   2026-08-04 hiding policy): a full-confidence, zero-correction read whose
   district SPLIT is undecidable (DA·H vs D·AH) is still the right character
   string — only the displayed grouping embodies a guess, and the user chose
   "show it, a wrong city guess is fine". Same for districts missing from
   the registry. `ambiguous` / `districtIssued` stay in the validation as
   information. Trigger case: a perfectly-read e-plate (8 chars, district
   tie) hidden by the old policy despite every charProb at 1.00.

**Results (30 photos / 42 plates), app config = cct_s + deskew:**
38/42 found, 26/30 photos covered, **38 shown correct / 0 wrong**, ~93ms
avg/image in node (typical photo ~35ms; only deskew-firing photos pay more).
Every found plate is now shown — the gate and the reader agree on this set.
The 4 not-found plates are the small background plates that only tiling
reached. Known trade vs Addendum 4: "0 wrong shown" now rests on
format+confidence+region alone; the public-set foreign-plate guard (region
head) and the corrections guard both remain.

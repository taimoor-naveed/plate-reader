import Panzoom, { type PanzoomObject } from '@panzoom/panzoom'
import type { ImageDataLike, PlateCandidate } from '../pipeline/types'
import { extractPlates, type PipelineSessions, type PipelineResult } from '../pipeline/pipeline'
import { loadWebSession } from './ort-web'
import { fileToImageData } from './decode'
import { renderPhotoView, type PhotoView } from './photo-view'

const $ = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T

let sessions: PipelineSessions | null = null
let sessionsPromise: Promise<PipelineSessions> | null = null
let currentImage: ImageDataLike | null = null
let lastResult: PipelineResult | null = null
let view: PhotoView | null = null

function loadSessions(): Promise<PipelineSessions> {
  return (async () => {
    // BASE_URL-prefixed (not root-absolute): Vite rewrites asset URLs in HTML
    // for `base`, but never inside TS string literals — a bare '/models/...'
    // would 404 under the GitHub Pages subpath (/plate-reader/).
    const base = import.meta.env.BASE_URL
    const [detector, ocr] = await Promise.all([
      loadWebSession(`${base}models/yolo-v9-t-384-license-plates-end2end.onnx`),
      loadWebSession(`${base}models/cct_xs_v2_global.onnx`),
    ])
    return { detector, ocr }
  })()
}

/**
 * Single-flight session loader. Concurrent callers share one in-flight promise;
 * a rejection is never cached — the next call starts a fresh attempt.
 */
function ensureSessions(): Promise<PipelineSessions> {
  if (sessions) return Promise.resolve(sessions)
  if (!sessionsPromise) {
    sessionsPromise = loadSessions().then(
      (s) => {
        sessions = s
        return s
      },
      (err) => {
        sessionsPromise = null
        throw err
      },
    )
  }
  return sessionsPromise
}

// Preload silently at startup; a failure here is swallowed — handleFile calls
// ensureSessions() again on first use and surfaces any error there.
void ensureSessions().catch(() => {})

function setStatus(msg: string) {
  $('#status').textContent = msg
}

/**
 * Keep edits uppercase and within the plate charset; spaces are kept (grouping,
 * best-effort). ÄÖÜ are allowed — the validator auto-applies umlaut districts
 * (e.g. TÖL), so stripping them would wipe the umlaut on the first keystroke.
 */
function normalizeEditableValue(v: string): string {
  return v.toUpperCase().replace(/[^A-ZÄÖÜ0-9 ]/g, '')
}

/**
 * Static plate face with real-plate anatomy: district group, then a vertical
 * stack of two generic seal circles (green inspection disc above, silver/grey
 * state seal below), then the remaining groups. Re-rendered from the input's
 * value after each in-place edit. Seals need at least two groups to have a
 * place to sit; a single-token value renders without them.
 */
function renderPlateStatic(el: HTMLElement, display: string) {
  el.innerHTML = ''
  const tokens = display.split(' ').filter(Boolean)
  const grp = (t: string) => {
    const s = document.createElement('span')
    s.className = 'plate-grp'
    s.textContent = t
    return s
  }
  el.appendChild(grp(tokens[0] ?? ''))
  if (tokens.length > 1) {
    const seals = document.createElement('span')
    seals.className = 'seal-stack'
    seals.innerHTML = '<span class="seal seal-inspection"></span><span class="seal seal-state"></span>'
    el.appendChild(seals)
    for (const t of tokens.slice(1)) el.appendChild(grp(t))
  }
  // 8-glyph plates (MAX_TOTAL_LEN, e.g. TÖL AB 123) cannot hold 66%-height glyphs
  // plus seals inside 520:110 — real plates switch to the narrower Engschrift for
  // these; we ship one font, so shrink the tier instead.
  const glyphs = tokens.join('').length
  el.style.setProperty('--plate-fs', glyphs >= 8 ? '105cqh' : '120cqh')
}

/** Tiny inline SVG: 12 stars in a circle (EU flag), sized to sit above the "D". */
function euStarsSvg(): string {
  const stars = Array.from({ length: 12 }, (_, i) => `<text x="12" y="5" transform="rotate(${i * 30} 12 12)">★</text>`).join('')
  return `<svg class="eu-stars" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><g fill="#ffcc00" font-size="6" text-anchor="middle">${stars}</g></svg>`
}

function selectCandidate(index: number) {
  const c = lastResult?.candidates[index]
  if (!c) return
  const cards = $('#cards')
  cards.querySelectorAll('.candidate').forEach((wrap, i) => {
    wrap.classList.toggle('selected', i === index)
    wrap.querySelector('.plate-card, .plain-card')?.classList.toggle('selected', i === index)
    const input = wrap.querySelector('.plate-input') as HTMLInputElement | null
    if (input) input.readOnly = i !== index
  })
  view?.select(index)
  const selected = cards.children[index] as HTMLElement | undefined
  selected?.scrollIntoView({ block: 'nearest' })
}

/**
 * German-format reads (rule 'DE') get the authentic plate costume: blue EU band
 * (stars + "D") + white FE-Schrift face. Anything else renders as a plain neutral
 * chip — no EU band, no plate styling (the read isn't a German plate, so dressing
 * it as one would be a lie). Both variants are tappable and editable in place.
 * Each card ships inside a .candidate wrapper with a meta row on top: index chip
 * (matching the box number on the photo), read confidence, country / nudge.
 */
function renderCard(c: PlateCandidate, index: number, showIndex: boolean): HTMLDivElement {
  const isGerman = c.validation.rule === 'DE'
  const card = document.createElement('div')
  card.className = isGerman ? 'plate-card' : 'plain-card'
  if (isGerman) {
    const band = document.createElement('span')
    band.className = 'eu-band'
    band.innerHTML = `${euStarsSvg()}<span class="eu-d">D</span>`
    card.appendChild(band)
  }

  const face = document.createElement('span')
  face.className = 'plate-face'
  let staticLayer: HTMLSpanElement | null = null
  if (isGerman) {
    // static anatomy layer (district | seals | groups); the input only spans the
    // face while editing (second tap on the selected card), and returns on blur
    staticLayer = document.createElement('span')
    staticLayer.className = 'plate-static'
    renderPlateStatic(staticLayer, c.validation.display)
    face.appendChild(staticLayer)
  }
  const input = document.createElement('input')
  input.className = 'plate-input'
  input.type = 'text'
  input.autocapitalize = 'characters'
  input.autocomplete = 'off'
  input.spellcheck = false
  input.value = c.validation.display
  input.readOnly = true
  input.addEventListener('input', () => {
    const before = input.value
    const pos = input.selectionStart ?? before.length
    const after = normalizeEditableValue(before)
    if (after !== before) {
      const delta = before.length - after.length
      input.value = after
      const newPos = Math.max(0, pos - delta)
      input.setSelectionRange(newPos, newPos)
    }
  })
  if (staticLayer) {
    const layer = staticLayer
    input.addEventListener('blur', () => {
      card.classList.remove('editing')
      renderPlateStatic(layer, input.value)
    })
  }
  face.appendChild(input)
  card.appendChild(face)

  card.addEventListener('click', () => {
    const wasSelected = card.classList.contains('selected')
    selectCandidate(index)
    // second tap on an already-selected German card enters in-place edit mode
    if (staticLayer && wasSelected && !card.classList.contains('editing')) {
      card.classList.add('editing') // before focus(): display:none inputs can't take focus
      input.focus()
      input.setSelectionRange(input.value.length, input.value.length)
    }
  })

  const wrap = document.createElement('div')
  wrap.className = 'candidate'
  // meta row: just the index chip matching the box number on the photo.
  // No country label — the app only shows German plates, so naming the
  // country is redundant (reintroduce if more countries are ever supported).
  if (showIndex) {
    const meta = document.createElement('div')
    meta.className = 'meta'
    const chip = document.createElement('span')
    chip.className = 'mchip'
    chip.textContent = String(index + 1)
    meta.appendChild(chip)
    wrap.appendChild(meta)
  }
  wrap.appendChild(card)
  return wrap
}

/**
 * Render the cards below the (already-visible) photo, and drop the boxes onto its
 * overlay. Owns lastResult so selection always matches what is on screen.
 * Exported for headless UI verification (driven with synthetic results).
 */
/**
 * Automation bar (user decision 2026-07-27): only German-format reads at full
 * confidence are shown at all — boxes and cards alike. A plate the pipeline
 * can't read with certainty is treated as not read; the user retakes the photo
 * instead of second-guessing a maybe-read.
 *
 * The bar is 0.995, not 1.0: confidence is a mean of per-char probabilities
 * (+0.05 format bonus, clamped), so 0.995 = "rounds to 100%" — a strict 1.0
 * arbitrarily hid a correct 8-char read with a single 94% character. Verified
 * on the local eval set (30 photos / 42 plates): 35 correct reads shown, zero
 * wrong reads pass at this bar.
 */
function isCertain(c: PlateCandidate): boolean {
  return c.validation.rule === 'DE' && c.validation.confidence >= 0.995
}

export function renderResult(result: PipelineResult) {
  result = { ...result, candidates: result.candidates.filter(isCertain) }
  lastResult = result
  const cards = $('#cards')
  cards.innerHTML = ''

  view?.setBoxes(result.candidates.map((c) => c.box))

  $('#no-plate').hidden = result.candidates.length > 0
  const showIndex = result.candidates.length > 1
  result.candidates.forEach((c, i) => cards.appendChild(renderCard(c, i, showIndex)))

  if (result.candidates.length === 1) selectCandidate(0)
}

let panzoom: PanzoomObject | null = null

/**
 * Pinch/pan/wheel zoom for the photo, via Panzoom (transform-based, so
 * photo-view's rect-relative tap math keeps working — the bounding rect scales
 * with the content). Bound once to the persistent canvas; every new photo
 * resets to fit. A click that follows a real pan is swallowed in the capture
 * phase so dragging the photo never counts as tapping a plate box.
 */
function ensurePanzoom(canvas: HTMLCanvasElement) {
  if (panzoom) return
  // contain: 'outside' is Panzoom's built-in containment: panning is clamped so
  // the photo always covers the panel — no gaps, no manual snap-back. It assumes
  // the element fills its container at rest, which the CSS guarantees (the panel
  // hugs the canvas exactly; the size cap lives on the canvas itself).
  const pz = (panzoom = Panzoom(canvas, { maxScale: 6, minScale: 1, contain: 'outside', panOnlyWhenZoomed: true }))
  const panel = canvas.parentElement!
  panel.addEventListener('wheel', pz.zoomWithWheel)
  canvas.addEventListener('dblclick', () => pz.reset())
  let downX = 0
  let downY = 0
  panel.addEventListener('pointerdown', (e) => { downX = e.clientX; downY = e.clientY }, true)
  panel.addEventListener(
    'click',
    (e) => {
      if (Math.hypot(e.clientX - downX, e.clientY - downY) > 8) e.stopPropagation()
    },
    true,
  )
}

/**
 * Show the decoded photo immediately (rectangle-free) so it's visible in every
 * outcome. Exported for headless UI verification (paired with renderResult).
 */
export function showPhoto(image: ImageDataLike) {
  const canvas = $('#photo-canvas') as HTMLCanvasElement
  $('#photo-placeholder').hidden = true
  canvas.hidden = false
  view = renderPhotoView(canvas, image, [], (i) => selectCandidate(i))
  ensurePanzoom(canvas)
  panzoom?.reset({ animate: false })
}

async function handleFile(file: File) {
  // reset stale state up front so a failure never shows a previous photo's read
  $('#cards').innerHTML = ''
  $('#no-plate').hidden = true
  lastResult = null
  try {
    setStatus('Reading photo…')
    const [s, image] = await Promise.all([ensureSessions(), fileToImageData(file)])
    currentImage = image
    showPhoto(image)
    setStatus('Looking for plates…')
    const result = await extractPlates(currentImage, s)
    setStatus('')
    renderResult(result)
  } catch (err) {
    setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`)
  }
}

export function getCurrentImage() {
  return currentImage
}
export function getLastResult() {
  return lastResult
}
export function getSessions() {
  return sessions
}

// In-browser iOS Safari ignores user-scalable=no (installed apps honor it) —
// suppress its native page-pinch so only the photo zooms there too.
document.addEventListener('gesturestart', (e) => e.preventDefault())

for (const id of ['camera-input', 'gallery-input']) {
  $(`#${id}`).addEventListener('change', (e) => {
    const file = (e.target as HTMLInputElement).files?.[0]
    if (file) void handleFile(file)
    ;(e.target as HTMLInputElement).value = ''
  })
}

import { extractPlates } from '../pipeline/pipeline'
import { normalizePlateText } from '../pipeline/validate'
import { isCertain } from '../pipeline/certainty'
import { foldUmlauts } from '../pipeline/rules/de'
import { loadWebSession } from './ort-web'
import { fileToImageData, cropToDataUrl } from './decode'

const summary = document.getElementById('summary')!
const cards = document.getElementById('cards')!

const [detector, ocr, ocrFallback] = await Promise.all([
  loadWebSession(`${import.meta.env.BASE_URL}models/yolo-v9-t-384-license-plates-end2end.onnx`),
  loadWebSession(`${import.meta.env.BASE_URL}models/cct_xs_v2_global.onnx`),
  loadWebSession(`${import.meta.env.BASE_URL}models/cct_s_v2_global.onnx`),
])
// value: one plate or an unordered array of ALL readable plates (equal priority) — mirrors scripts/eval.ts
const expected: Record<string, string | string[]> = await (await fetch('/eval-data/expected.json')).json()

let done = 0
let platesFound = 0
let platesExpected = 0
let shownCorrect = 0
let shownWrong = 0
let totalMs = 0
const n = Object.keys(expected).length

for (const [file, want] of Object.entries(expected)) {
  // fold umlauts on both sides — mirrors scripts/eval.ts
  const plates = (Array.isArray(want) ? want : [want]).map((p) => normalizePlateText(foldUmlauts(p)))
  const blob = await (await fetch(`/attachments/${file}`)).blob()
  const image = await fileToImageData(blob)
  // same lever config as the app (src/web/app.ts) — this harness measures what the app does
  const res = await extractPlates(image, { detector, ocr, ocrFallback }, { deskew: true, escalate: true })
  const got = res.candidates.map((c) => foldUmlauts(c.validation.plate))
  const found = plates.filter((p) => got.includes(p))
  const pass = found.length === plates.length
  const shown = res.candidates.filter(isCertain).map((c) => foldUmlauts(c.validation.plate))
  const wrongShown = shown.filter((p) => !plates.includes(p))

  done++
  platesExpected += plates.length
  platesFound += found.length
  shownCorrect += shown.length - wrongShown.length
  shownWrong += wrongShown.length
  totalMs += res.timings.totalMs

  const card = document.createElement('div')
  card.className = 'card'
  const thumb = res.candidates[0] ? `<img src="${cropToDataUrl(image, res.candidates[0].box)}" alt="">` : ''
  const gotText = got.length ? got.join(', ') : '—'
  card.innerHTML = `${thumb}<span class="${pass ? 'pass' : 'fail'}">${pass ? 'PASS' : 'FAIL'}</span>
    <span class="mono">${file}</span> <span class="mono">want ${plates.join(', ')} · got ${gotText} · shown ${shown.length ? shown.join(', ') : '—'}</span>
    <span>${Math.round(res.timings.totalMs)}ms</span>`
  cards.appendChild(card)
  summary.textContent = `plates found ${platesFound}/${platesExpected} · shown ${shownCorrect} correct / ${shownWrong} wrong · ${done}/${n} processed · avg ${Math.round(totalMs / done)}ms`
}

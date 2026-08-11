// Diagnose one photo: run the app's exact pipeline config and dump every
// candidate with the full validation verdict — the "why did this photo not
// show a plate" tool. Never prints anything to committed files.
// Usage: tsx scripts/diagnose.ts <image>
import { loadNodeSession } from '../src/node/ort-node'
import { decodeImageFile } from '../src/node/decode'
import { extractPlates } from '../src/pipeline/pipeline'
import { isCertain } from '../src/pipeline/certainty'

const imagePath = process.argv[2]
if (!imagePath) {
  console.error('usage: tsx scripts/diagnose.ts <image>')
  process.exit(1)
}

const detector = await loadNodeSession('public/models/yolo-v9-t-384-license-plates-end2end.onnx')
const ocr = await loadNodeSession('public/models/cct_s_v2_global.onnx')

const image = await decodeImageFile(imagePath)
console.log(`image ${image.width}x${image.height}  config: app default (cct_s + deskew)\n`)
const res = await extractPlates(image, { detector, ocr }, { deskew: true })

if (res.candidates.length === 0) console.log('NO DETECTIONS — the detector proposed no box at all')
for (const [i, c] of res.candidates.entries()) {
  const b = c.box
  const w = Math.round(b.x2 - b.x1)
  const h = Math.round(b.y2 - b.y1)
  const v = c.validation
  console.log(`candidate ${i + 1}: box [${Math.round(b.x1)},${Math.round(b.y1)} ${w}x${h}] (h/w=${(h / w).toFixed(2)}) detector=${b.score.toFixed(2)}`)
  console.log(`  read        "${c.read.text}"  charProbs=[${c.read.charProbs.map((p) => p.toFixed(2)).join(',')}]`)
  console.log(`  region      ${c.read.region}@${(c.read.regionProb ?? 0).toFixed(2)}`)
  console.log(`  validation  rule=${v.rule} plate="${v.plate}" display="${v.display}" confidence=${v.confidence.toFixed(3)}`)
  console.log(`              formatValid=${v.formatValid} corrections=${JSON.stringify(v.corrections)} ambiguous=${v.ambiguous} districtIssued=${v.districtIssued}`)
  const certain = isCertain(c)
  let why = ''
  if (!certain) {
    if (v.rule !== 'DE') why = 'no German format match'
    else if (v.corrections.length > 0) why = 'needed lookalike corrections'
    else if (v.confidence < 0.995) why = 'confidence below bar'
    else why = 'region head not Germany'
  }
  console.log(`  VERDICT     ${certain ? 'SHOWN (certain)' : `HIDDEN — ${why}`}\n`)
}
console.log(`timings: detect ${Math.round(res.timings.detectMs)}ms, ocr ${Math.round(res.timings.ocrMs)}ms`)

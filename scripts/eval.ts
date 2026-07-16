import fs from 'node:fs'
import path from 'node:path'
import { loadNodeSession } from '../src/node/ort-node'
import { decodeImageFile, saveRegionPng } from '../src/node/decode'
import { extractPlates } from '../src/pipeline/pipeline'
import { normalizePlateText } from '../src/pipeline/validate'
import { crop, expandBox } from '../src/pipeline/image'

function arg(name: string, def: string): string {
  const i = process.argv.indexOf(`--${name}`)
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1]! : def
}

function flag(name: string): boolean {
  return process.argv.includes(`--${name}`)
}

const detectorSize = Number(arg('detector', '384'))
const ocrName = arg('ocr', 'xs') === 's' ? 'cct_s_v2_global' : 'cct_xs_v2_global'
const cropMargin = Number(arg('margin', '0'))
// conditional levers (Task 12) — opt-in, off by default; see docs/eval-results.md for results
const smallBoxMargin = flag('small-margin')
const normalizeCrop = flag('normalize-crop')
const rotationSweep = flag('rotation-sweep') ? [-10, -5, 0, 5, 10] : undefined

const expectedPath = 'eval/expected.json'
if (!fs.existsSync(expectedPath)) {
  console.error('eval/expected.json missing — see eval/expected.example.json')
  process.exit(1)
}
// value: one plate or an unordered array of ALL readable plates (equal priority)
const expected: Record<string, string | string[]> = JSON.parse(fs.readFileSync(expectedPath, 'utf8'))

const detector = await loadNodeSession(`public/models/yolo-v9-t-${detectorSize}-license-plates-end2end.onnx`)
const ocr = await loadNodeSession(`public/models/${ocrName}.onnx`)
fs.mkdirSync('eval/out', { recursive: true })

let platesExpected = 0
let platesFound = 0
let photosFull = 0
let totalMs = 0
const results: object[] = []

console.log(
  `config: detector=${detectorSize} ocr=${ocrName} margin=${cropMargin}` +
    `${smallBoxMargin ? ' +smallBoxMargin' : ''}${normalizeCrop ? ' +normalizeCrop' : ''}${rotationSweep ? ` +rotationSweep=${rotationSweep.join(',')}` : ''}\n`,
)
for (const [file, want] of Object.entries(expected)) {
  const plates = (Array.isArray(want) ? want : [want]).map(normalizePlateText)
  const image = await decodeImageFile(path.join('attachments', file))
  const res = await extractPlates(image, { detector, ocr }, { detectorSize, cropMargin, smallBoxMargin, normalizeCrop, rotationSweep })
  totalMs += res.timings.totalMs
  const got = res.candidates.map((c) => c.validation.plate)
  const missed = plates.filter((p) => !got.includes(p))
  const extras = got.filter((g) => !plates.includes(g))
  platesExpected += plates.length
  platesFound += plates.length - missed.length

  let status: string
  if (missed.length === 0) {
    status = 'PASS'
    photosFull++
  } else if (res.candidates.length === 0) {
    status = 'NONE'
  } else {
    status = `MISS ${missed.join(',')}`
  }
  if (missed.length && res.candidates[0]) {
    const b = expandBox(res.candidates[0].box, 0.1, image.width, image.height)
    await saveRegionPng(crop(image, b), `eval/out/${file.replace('.jpg', '')}-crop.png`)
  }
  results.push({ file, want: plates, got, status, ms: Math.round(res.timings.totalMs) })
  console.log(
    `${status.padEnd(28)} ${file}  found=${plates.length - missed.length}/${plates.length}${extras.length ? `  extra=${extras.join(',')}` : ''}  ${Math.round(res.timings.totalMs)}ms`,
  )
}

const n = Object.keys(expected).length
console.log(
  `\nplates found: ${platesFound}/${platesExpected}   photos fully covered: ${photosFull}/${n}   avg ${Math.round(totalMs / n)}ms/image`,
)
// "extra" reads are informational, not failures: often a real plate we deemed
// unreadable during labeling, or a misread of one — inspect, don't panic.
fs.writeFileSync('eval/out/results.json', JSON.stringify({ config: { detectorSize, ocrName, cropMargin }, results }, null, 2))

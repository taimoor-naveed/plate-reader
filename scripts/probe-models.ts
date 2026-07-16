import * as ort from 'onnxruntime-node'

async function probe(path: string, feedsFactory: (inputName: string) => Record<string, ort.Tensor>) {
  const s = await ort.InferenceSession.create(path)
  console.log(`\n=== ${path}`)
  console.log('inputs :', s.inputNames.join(', '))
  console.log('outputs:', s.outputNames.join(', '))
  const inputName = s.inputNames[0]
  if (!inputName) throw new Error('model has no inputs')
  const res = await s.run(feedsFactory(inputName))
  for (const [name, t] of Object.entries(res)) {
    console.log(`output ${name}: dims=[${t.dims.join(',')}] type=${t.type}`)
  }
  return res
}

// Detector: expect output rows of 7 (possibly 0 rows on a blank image)
for (const size of [384, 512]) {
  const res = await probe(
    `public/models/yolo-v9-t-${size}-license-plates-end2end.onnx`,
    (name) => ({ [name]: new ort.Tensor('float32', new Float32Array(1 * 3 * size * size), [1, 3, size, size]) }),
  )
  const out = Object.values(res)[0]!
  const cols = out.dims[out.dims.length - 1]
  if (cols !== 7 || (out.data as Float32Array).length % 7 !== 0)
    throw new Error(`detector ${size}: expected rows of 7, got dims [${out.dims.join(',')}]`)
}

// OCR: expect one output with 370 elems (10 slots x 37 alphabet) and one with 66 (regions)
for (const m of ['cct_xs_v2_global', 'cct_s_v2_global']) {
  const res = await probe(`public/models/${m}.onnx`, (name) => ({
    [name]: new ort.Tensor('uint8', new Uint8Array(1 * 64 * 128 * 3), [1, 64, 128, 3]),
  }))
  const sizes = Object.values(res).map((t) => t.dims.reduce((a, b) => a * b, 1))
  if (!sizes.includes(370)) throw new Error(`${m}: no 370-element plate head (got ${sizes.join(',')})`)
  if (!sizes.includes(66)) throw new Error(`${m}: no 66-element region head (got ${sizes.join(',')})`)
}
console.log('\nAll model contracts verified.')

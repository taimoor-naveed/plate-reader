// wasm-only build (we never use webgpu/jsep) + Vite ?url imports so the runtime
// files are served in dev and bundled in prod from OUR origin (never a CDN).
// NOTE: do NOT copy ort files into public/ and point wasmPaths there — Vite
// refuses dynamic import() of modules under public/ (500), which breaks ort
// with "no available backend" in every real browser.
import * as ort from 'onnxruntime-web/wasm'
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url'
import ortMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url'
import type { OrtSessionLike, TensorLike } from '../pipeline/types'

let configured = false

export async function loadWebSession(url: string): Promise<OrtSessionLike> {
  if (!configured) {
    ort.env.wasm.wasmPaths = { wasm: ortWasmUrl, mjs: ortMjsUrl }
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1
    configured = true
  }
  const s = await ort.InferenceSession.create(url, { executionProviders: ['wasm'] })
  return {
    inputNames: s.inputNames,
    outputNames: s.outputNames,
    async run(feeds: Record<string, TensorLike>) {
      const ortFeeds: Record<string, ort.Tensor> = {}
      for (const [k, t] of Object.entries(feeds)) ortFeeds[k] = new ort.Tensor(t.type, t.data, t.dims)
      const res = await s.run(ortFeeds)
      const out: Record<string, TensorLike> = {}
      for (const [k, t] of Object.entries(res)) {
        out[k] = { type: t.type as TensorLike['type'], data: t.data as Float32Array, dims: [...t.dims] }
      }
      return out
    },
  }
}

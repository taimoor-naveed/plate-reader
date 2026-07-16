import * as ort from 'onnxruntime-node'
import type { OrtSessionLike, TensorLike } from '../pipeline/types'

export async function loadNodeSession(path: string): Promise<OrtSessionLike> {
  const s = await ort.InferenceSession.create(path)
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

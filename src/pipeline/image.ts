import type { Box, ImageDataLike } from './types'

/** Center-aligned bilinear resize (matches cv2.INTER_LINEAR mapping). */
export function resizeBilinear(src: ImageDataLike, dstW: number, dstH: number): ImageDataLike {
  if (dstW === src.width && dstH === src.height) {
    return { data: new Uint8ClampedArray(src.data), width: src.width, height: src.height }
  }
  const dst = new Uint8ClampedArray(dstW * dstH * 4)
  const sx = src.width / dstW
  const sy = src.height / dstH
  for (let y = 0; y < dstH; y++) {
    const fy = Math.min(Math.max((y + 0.5) * sy - 0.5, 0), src.height - 1)
    const y0 = Math.floor(fy)
    const y1 = Math.min(y0 + 1, src.height - 1)
    const wy = fy - y0
    for (let x = 0; x < dstW; x++) {
      const fx = Math.min(Math.max((x + 0.5) * sx - 0.5, 0), src.width - 1)
      const x0 = Math.floor(fx)
      const x1 = Math.min(x0 + 1, src.width - 1)
      const wx = fx - x0
      const i00 = (y0 * src.width + x0) * 4
      const i10 = (y0 * src.width + x1) * 4
      const i01 = (y1 * src.width + x0) * 4
      const i11 = (y1 * src.width + x1) * 4
      const o = (y * dstW + x) * 4
      for (let c = 0; c < 4; c++) {
        const top = src.data[i00 + c]! * (1 - wx) + src.data[i10 + c]! * wx
        const bot = src.data[i01 + c]! * (1 - wx) + src.data[i11 + c]! * wx
        dst[o + c] = Math.round(top * (1 - wy) + bot * wy)
      }
    }
  }
  return { data: dst, width: dstW, height: dstH }
}

/** Crop box region (rounded, clamped, min 1px). */
export function crop(src: ImageDataLike, box: Box): ImageDataLike {
  const x1 = Math.min(Math.max(Math.round(box.x1), 0), src.width - 1)
  const y1 = Math.min(Math.max(Math.round(box.y1), 0), src.height - 1)
  const x2 = Math.min(Math.max(Math.round(box.x2), x1 + 1), src.width)
  const y2 = Math.min(Math.max(Math.round(box.y2), y1 + 1), src.height)
  const w = x2 - x1
  const h = y2 - y1
  const data = new Uint8ClampedArray(w * h * 4)
  for (let y = 0; y < h; y++) {
    const srcOff = ((y1 + y) * src.width + x1) * 4
    data.set(src.data.subarray(srcOff, srcOff + w * 4), y * w * 4)
  }
  return { data, width: w, height: h }
}

/** Expand each side by margin × that dimension's box size, clamped to the image. */
export function expandBox(box: Box, margin: number, imgW: number, imgH: number): Box {
  const mw = (box.x2 - box.x1) * margin
  const mh = (box.y2 - box.y1) * margin
  return {
    x1: Math.max(0, box.x1 - mw),
    y1: Math.max(0, box.y1 - mh),
    x2: Math.min(imgW, box.x2 + mw),
    y2: Math.min(imgH, box.y2 + mh),
    score: box.score,
  }
}

export function cropResize(src: ImageDataLike, box: Box, dstW: number, dstH: number): ImageDataLike {
  return resizeBilinear(crop(src, box), dstW, dstH)
}

export interface Letterboxed {
  image: ImageDataLike
  ratio: number
  dw: number
  dh: number
}

/** Per-channel linear min→max contrast stretch (alpha untouched). */
export function stretchContrast(im: ImageDataLike): ImageDataLike {
  const data = new Uint8ClampedArray(im.data)
  for (let c = 0; c < 3; c++) {
    let min = 255
    let max = 0
    for (let i = c; i < data.length; i += 4) {
      const v = data[i]!
      if (v < min) min = v
      if (v > max) max = v
    }
    const range = max - min
    if (range < 10) continue
    for (let i = c; i < data.length; i += 4) data[i] = ((data[i]! - min) * 255) / range
  }
  return { data, width: im.width, height: im.height }
}

/** Rotate around center by deg (bilinear, gray 114 fill). */
export function rotateImage(im: ImageDataLike, deg: number): ImageDataLike {
  const rad = (-deg * Math.PI) / 180
  const cos = Math.cos(rad)
  const sin = Math.sin(rad)
  const cx = im.width / 2
  const cy = im.height / 2
  const data = new Uint8ClampedArray(im.data.length)
  for (let y = 0; y < im.height; y++) {
    for (let x = 0; x < im.width; x++) {
      const sx = cos * (x - cx) - sin * (y - cy) + cx
      const sy = sin * (x - cx) + cos * (y - cy) + cy
      const o = (y * im.width + x) * 4
      if (sx < 0 || sy < 0 || sx > im.width - 1 || sy > im.height - 1) {
        data[o] = 114
        data[o + 1] = 114
        data[o + 2] = 114
        data[o + 3] = 255
        continue
      }
      const x0 = Math.floor(sx)
      const y0 = Math.floor(sy)
      const x1 = Math.min(x0 + 1, im.width - 1)
      const y1 = Math.min(y0 + 1, im.height - 1)
      const wx = sx - x0
      const wy = sy - y0
      for (let c = 0; c < 4; c++) {
        const top = im.data[(y0 * im.width + x0) * 4 + c]! * (1 - wx) + im.data[(y0 * im.width + x1) * 4 + c]! * wx
        const bot = im.data[(y1 * im.width + x0) * 4 + c]! * (1 - wx) + im.data[(y1 * im.width + x1) * 4 + c]! * wx
        data[o + c] = Math.round(top * (1 - wy) + bot * wy)
      }
    }
  }
  return { data, width: im.width, height: im.height }
}

/** YOLOv9-style letterbox: aspect-preserving resize into size×size with gray(114) padding. */
export function letterbox(src: ImageDataLike, size: number): Letterboxed {
  const r = Math.min(size / src.height, size / src.width)
  const newW = Math.round(src.width * r)
  const newH = Math.round(src.height * r)
  const resized = resizeBilinear(src, newW, newH)
  const dw = (size - newW) / 2
  const dh = (size - newH) / 2
  const left = Math.round(dw - 0.1)
  const top = Math.round(dh - 0.1)
  const data = new Uint8ClampedArray(size * size * 4)
  for (let i = 0; i < size * size; i++) {
    data[i * 4] = 114
    data[i * 4 + 1] = 114
    data[i * 4 + 2] = 114
    data[i * 4 + 3] = 255
  }
  for (let y = 0; y < newH; y++) {
    const dstOff = ((top + y) * size + left) * 4
    data.set(resized.data.subarray(y * newW * 4, (y + 1) * newW * 4), dstOff)
  }
  return { image: { data, width: size, height: size }, ratio: r, dw, dh }
}

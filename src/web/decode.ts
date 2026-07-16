import type { Box, ImageDataLike } from '../pipeline/types'
import { crop, expandBox } from '../pipeline/image'

/** Decode a photo file to RGBA ImageData, honoring EXIF orientation. */
export async function fileToImageData(file: Blob): Promise<ImageDataLike> {
  let bmp: ImageBitmap
  try {
    bmp = await createImageBitmap(file, { imageOrientation: 'from-image' })
  } catch {
    bmp = await createImageBitmap(file) // older Safari: option object unsupported
  }
  const canvas = document.createElement('canvas')
  canvas.width = bmp.width
  canvas.height = bmp.height
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!
  ctx.drawImage(bmp, 0, 0)
  bmp.close()
  return ctx.getImageData(0, 0, canvas.width, canvas.height)
}

/** JPEG data-url thumbnail of a box region (for the candidate menu). */
export function cropToDataUrl(image: ImageDataLike, box: Box): string {
  const c = crop(image, expandBox(box, 0.15, image.width, image.height))
  const canvas = document.createElement('canvas')
  canvas.width = c.width
  canvas.height = c.height
  canvas.getContext('2d')!.putImageData(new ImageData(c.data as Uint8ClampedArray<ArrayBuffer>, c.width, c.height), 0, 0)
  return canvas.toDataURL('image/jpeg', 0.8)
}

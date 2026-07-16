import sharp from 'sharp'
import type { ImageDataLike } from '../pipeline/types'

/** Decode an image file to RGBA, applying EXIF orientation. */
export async function decodeImageFile(path: string): Promise<ImageDataLike> {
  const { data, info } = await sharp(path).rotate().ensureAlpha().raw().toBuffer({ resolveWithObject: true })
  return { data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength), width: info.width, height: info.height }
}

/** Save an RGBA region as PNG (debugging failed reads). */
export async function saveRegionPng(image: ImageDataLike, path: string): Promise<void> {
  await sharp(Buffer.from(image.data.buffer, image.data.byteOffset, image.data.byteLength), {
    raw: { width: image.width, height: image.height, channels: 4 },
  })
    .png()
    .toFile(path)
}

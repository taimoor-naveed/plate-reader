import type { Box, ImageDataLike } from '../pipeline/types'

const DISPLAY_MAX_WIDTH = 1000
/* Brand palette (FLEXOPTIX, closed): FO White for candidates, FO Orange for the
   selected one — selection is the single accent. Each stroke sits on an FO
   Black halo so boxes stay visible over white plates and bright scenes. */
const UNSELECTED_COLOR = '#FFFFFF'
const SELECTED_COLOR = '#FF6B00'
const HALO_COLOR = '#00080A'
const TAG_TEXT_COLOR = '#00080A'
const CORNER_RADIUS = 8

export interface PhotoView {
  /** Highlight the box at this index (or clear the highlight with null). */
  select(index: number | null): void
  /** Replace the set of boxes drawn over the (already-decoded) photo. */
  setBoxes(boxes: Box[]): void
}

function roundedRectPath(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.arcTo(x + w, y, x + w, y + h, r)
  ctx.arcTo(x + w, y + h, x, y + h, r)
  ctx.arcTo(x, y + h, x, y, r)
  ctx.arcTo(x, y, x + w, y, r)
  ctx.closePath()
}

/**
 * Draw `image` into `canvas` (downscaled so its display width is at most 1000px)
 * with a rounded rectangle over every box, numbered when there is more than one.
 * Tapping a rectangle invokes onTap with its index into the current box list.
 */
export function renderPhotoView(
  canvas: HTMLCanvasElement,
  image: ImageDataLike,
  initialBoxes: Box[],
  onTap: (index: number) => void,
): PhotoView {
  const scale = Math.min(1, DISPLAY_MAX_WIDTH / image.width)
  canvas.width = Math.round(image.width * scale)
  canvas.height = Math.round(image.height * scale)
  const ctx = canvas.getContext('2d')!

  // Draw the full-resolution image into an offscreen canvas once, then blit it
  // scaled down — avoids re-decoding ImageData whenever boxes/selection change.
  const full = document.createElement('canvas')
  full.width = image.width
  full.height = image.height
  full
    .getContext('2d')!
    .putImageData(new ImageData(image.data as Uint8ClampedArray<ArrayBuffer>, image.width, image.height), 0, 0)

  const displayRect = (b: Box) => ({
    x: b.x1 * scale,
    y: b.y1 * scale,
    w: (b.x2 - b.x1) * scale,
    h: (b.y2 - b.y1) * scale,
  })

  let boxes = initialBoxes
  let selected: number | null = null

  function draw() {
    ctx.drawImage(full, 0, 0, canvas.width, canvas.height)
    const showNumbers = boxes.length > 1
    boxes.forEach((box, i) => {
      const { x, y, w, h } = displayRect(box)
      const isSelected = i === selected
      const color = isSelected ? SELECTED_COLOR : UNSELECTED_COLOR
      roundedRectPath(ctx, x, y, w, h, CORNER_RADIUS)
      ctx.strokeStyle = HALO_COLOR
      ctx.lineWidth = isSelected ? 6 : 5
      ctx.stroke()
      ctx.strokeStyle = color
      ctx.lineWidth = isSelected ? 3.5 : 2.5
      ctx.stroke()
      if (showNumbers) {
        const label = String(i + 1)
        ctx.font = '600 15px Inter, system-ui, sans-serif'
        const tw = ctx.measureText(label).width
        const tagW = tw + 10
        const tagH = 20
        ctx.fillStyle = color
        roundedRectPath(ctx, x, y - tagH, tagW, tagH, 5)
        ctx.fill()
        ctx.fillStyle = TAG_TEXT_COLOR
        ctx.textBaseline = 'middle'
        ctx.fillText(label, x + 5, y - tagH / 2 + 1)
      }
    })
  }
  draw()

  // single-slot handler: renderPhotoView is re-invoked on the SAME persistent canvas
  // for every photo — addEventListener would accumulate listeners whose closures hold
  // stale boxes/scale and fire phantom onTap calls with the previous photo's geometry
  canvas.onclick = (e) => {
    const r = canvas.getBoundingClientRect()
    const px = ((e.clientX - r.left) / r.width) * canvas.width
    const py = ((e.clientY - r.top) / r.height) * canvas.height
    // topmost (last-drawn) box wins on overlap
    for (let i = boxes.length - 1; i >= 0; i--) {
      const { x, y, w, h } = displayRect(boxes[i]!)
      if (px >= x && px <= x + w && py >= y && py <= y + h) {
        onTap(i)
        return
      }
    }
  }

  return {
    select(index: number | null) {
      selected = index
      draw()
    },
    setBoxes(next: Box[]) {
      boxes = next
      selected = null
      draw()
    },
  }
}

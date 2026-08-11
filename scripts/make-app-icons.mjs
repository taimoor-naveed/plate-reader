#!/usr/bin/env node
// App launcher icons: official FLEXOPTIX eyecon (verbatim path from the brand
// package — never redrawn) over a simplified German license plate, on the
// FO Black canvas. The plate is the app's subject matter, reproduced like the
// in-app plate cards (EU-band blue / white face — content, not UI palette).
// One artwork serves both manifest purposes: everything sits inside the 80%
// safe-zone circle that Android's adaptive-icon masks are allowed to crop to,
// so `any` and `maskable` share the same files.
//
// Usage: node scripts/make-app-icons.mjs   (writes public/icon-*.png)
import sharp from 'sharp'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..')
const eyecon = readFileSync(path.join(root, 'flexoptix-ai-brand-system/logos/eyecon-dark.svg'), 'utf8')
const eyeconPath = eyecon.match(/<path[^>]+\/>/)?.[0]
if (!eyeconPath) throw new Error('eyecon path not found in brand package SVG')

// 100x100 canvas. Eyecon (120-unit artwork) scaled to 36 units; plate 58x14
// below it. Farthest plate corner sits ~39.6 from center — inside the r=40
// maskable safe zone (rounded corners add slack).
const art = `
<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 100 100" fill="none">
  <rect width="100" height="100" fill="#00080A"/>
  <g transform="translate(32,21) scale(0.3)">${eyeconPath}</g>
  <g>
    <rect x="21" y="63" width="58" height="14" rx="3" fill="#FFFFFF"/>
    <path d="M21 66a3 3 0 0 1 3-3h4.5v14H24a3 3 0 0 1-3-3z" fill="#003399"/>
    <rect x="33" y="66" width="5" height="8" rx="1.2" fill="#00080A"/>
    <rect x="41" y="66" width="5" height="8" rx="1.2" fill="#00080A"/>
    <rect x="54" y="66" width="5" height="8" rx="1.2" fill="#00080A"/>
    <rect x="62" y="66" width="5" height="8" rx="1.2" fill="#00080A"/>
    <rect x="70" y="66" width="5" height="8" rx="1.2" fill="#00080A"/>
  </g>
</svg>`

for (const size of [192, 512]) {
  const density = 72 * (size / 96)
  const png = await sharp(Buffer.from(art), { density }).resize(size, size).png()
  await png.clone().toFile(path.join(root, `public/icon-${size}.png`))
  await png.clone().toFile(path.join(root, `public/icon-maskable-${size}.png`))
  console.log(`icon-${size}.png + icon-maskable-${size}.png written`)
}

import sharp from 'sharp'

const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512">
  <rect width="512" height="512" rx="96" fill="#1a73e8"/>
  <rect x="66" y="196" width="380" height="120" rx="16" fill="#fff" stroke="#111" stroke-width="10"/>
  <text x="256" y="282" font-family="monospace" font-size="72" font-weight="bold" fill="#111" text-anchor="middle">BN·CR 7</text>
</svg>`
for (const size of [192, 512]) {
  await sharp(Buffer.from(svg)).resize(size, size).png().toFile(`public/icon-${size}.png`)
}
console.log('icons written')

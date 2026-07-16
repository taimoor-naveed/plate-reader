/**
 * Registers the offline service worker — production builds only. In dev the
 * SW is never registered: a SW would intercept and cache dev requests (Vite's
 * hot-reload traffic, unbundled module graph), which would fight HMR and mask
 * source-of-truth failures during development.
 *
 * Base-path aware: `import.meta.env.BASE_URL` is '/' locally and
 * '/plate-reader/' on GitHub Pages, so the same build registers correctly at
 * either scope.
 */
export function registerServiceWorker() {
  if (!import.meta.env.PROD) return
  if (!('serviceWorker' in navigator)) return
  window.addEventListener('load', () => {
    void navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`)
  })
}

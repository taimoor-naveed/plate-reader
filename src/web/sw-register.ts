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

  // The SW uses skipWaiting + clients.claim, so a freshly installed version
  // takes control of this already-running page — but the page's old JS/CSS
  // keep running. Reload once on that takeover so an update becomes visible
  // in the SAME launch instead of the next one. The pre-existing-controller
  // check keeps the very first visit (first-ever SW install) from reloading.
  let hadController = !!navigator.serviceWorker.controller
  navigator.serviceWorker.addEventListener('controllerchange', () => {
    if (!hadController) {
      hadController = true
      return
    }
    window.location.reload()
  })

  window.addEventListener('load', () => {
    const registering = navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`)
    // Installed PWAs can live in the app switcher for days without a real
    // "launch" — also check for updates whenever the app returns to the
    // foreground, not just on navigation.
    void registering.then((reg) => {
      document.addEventListener('visibilitychange', () => {
        if (!document.hidden) void reg.update()
      })
    })
  })
}

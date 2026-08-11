/**
 * Viewport height shim (2026-08-11 user report): Android Chrome in installed
 * standalone mode (edge-to-edge) computes dvh/vh larger than the real window
 * — a 100dvh shell overflowed by roughly the system-bar heights and pushed
 * the action bar off-screen under the gesture bar. window.innerHeight is the
 * one value that matches the actual window everywhere, so publish it as
 * --app-h and let the CSS size the shell from it (100dvh stays as the no-JS
 * fallback). Keyboard behavior is unaffected on iOS (innerHeight is keyboard-
 * stable there; only visualViewport shrinks) and native on Android (the
 * window itself resizes, and the shell should follow it).
 */
export function installViewportHeightVar() {
  const set = () => document.documentElement.style.setProperty('--app-h', `${window.innerHeight}px`)
  set()
  window.addEventListener('resize', set)
}

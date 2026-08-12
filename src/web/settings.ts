import { listInfo, listState, refreshMessage, type RefreshOutcome } from '../registry/registry'
import { getCachedAt, getIndex, getUrl, maybeAutoRefresh, refreshRegistry, setUrl } from './registry-client'

const $ = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T

/**
 * Settings dropdown (plates-list URL + force update + status line), the gear
 * warning badge, and the transient update notice. All strings come from
 * registry.ts — nothing user-facing is composed here.
 */
export function initSettings() {
  const btn = $<HTMLButtonElement>('#settings-btn')
  const panel = $('#settings')
  const urlInput = $<HTMLInputElement>('#plates-url')
  const updateBtn = $<HTMLButtonElement>('#plates-update')
  const info = $('#list-info')
  const badge = $('#settings-badge')
  const notice = $('#notice')
  let noticeTimer: number | undefined
  /** Set while the app has NO usable list: that notice must stay until the problem is fixed. */
  let stickyMsg: string | null = null

  const refreshBadge = () => {
    badge.hidden = listState(getIndex() !== null, getCachedAt(), Date.now()) === 'ok'
  }

  /** Panel status line; pass an explicit message (e.g. an error) to override the state line. */
  const renderInfo = (msg?: string, isError = false) => {
    info.textContent = msg ?? listInfo(getIndex()?.size, getCachedAt(), Date.now())
    info.classList.toggle('error', isError)
  }

  const showNotice = (msg: string, isError: boolean, sticky = false) => {
    if (noticeTimer !== undefined) clearTimeout(noticeTimer)
    noticeTimer = undefined
    notice.textContent = msg
    notice.classList.toggle('error', isError)
    notice.hidden = false
    // sticky (no usable list) never auto-hides; errors linger longer than successes
    if (!sticky) noticeTimer = window.setTimeout(() => (notice.hidden = true), isError ? 8000 : 4000)
  }
  notice.addEventListener('click', (e) => {
    e.stopPropagation() // keep the tap-outside closer from immediately re-closing the panel
    if (stickyMsg) setOpen(true) // no list — the notice is actionable, take the user to settings
    else notice.hidden = true
  })

  /**
   * Panel open → the outcome lands in the panel's status line (a notice would
   * overlap the panel); panel closed → notice below the topbar. With no usable
   * list the notice is sticky: it stays until a list actually loads.
   */
  const showOutcome = (outcome: RefreshOutcome) => {
    const isError = outcome.kind !== 'updated'
    const msg = refreshMessage(outcome, getCachedAt())
    stickyMsg = isError && getIndex() === null ? msg : null
    if (panel.hidden) showNotice(msg, isError, stickyMsg !== null)
    else renderInfo(isError ? msg : undefined, isError)
    refreshBadge()
  }

  const setOpen = (open: boolean) => {
    panel.hidden = !open
    btn.setAttribute('aria-expanded', String(open))
    if (open) {
      notice.hidden = true // the user is acting on it — and the panel overlaps it
      urlInput.value = getUrl()
      renderInfo()
    } else if (stickyMsg && getIndex() === null) {
      showNotice(stickyMsg, true, true) // still no list — the reminder comes back
    }
  }
  btn.addEventListener('click', () => setOpen(Boolean(panel.hidden)))
  document.addEventListener('click', (e) => {
    const t = e.target as Node
    if (!panel.hidden && !panel.contains(t) && !btn.contains(t)) setOpen(false)
  })

  /** Persist the input's URL; false only for a non-empty value that isn't http(s). */
  const applyUrl = (): boolean => {
    const value = urlInput.value.trim()
    if (value === '') {
      setUrl('')
      return true
    }
    try {
      const u = new URL(value)
      if (u.protocol !== 'http:' && u.protocol !== 'https:') throw new Error()
    } catch {
      renderInfo('Not a valid http(s) URL.', true)
      return false
    }
    setUrl(value)
    return true
  }
  urlInput.addEventListener('change', () => {
    if (applyUrl()) renderInfo()
  })

  updateBtn.addEventListener('click', () => {
    if (!applyUrl()) return
    updateBtn.disabled = true // single-flight upstream; disabled is the visual feedback
    void refreshRegistry()
      .then(showOutcome)
      .finally(() => (updateBtn.disabled = false))
  })

  // Launch: badge first, then the auto-update (fire-and-forget — nothing here
  // blocks first paint; a fresh cache resolves to null and stays silent).
  refreshBadge()
  const auto = maybeAutoRefresh()
  if (auto) void auto.then(showOutcome)
}

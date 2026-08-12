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

  const refreshBadge = () => {
    badge.hidden = listState(getIndex() !== null, getCachedAt(), Date.now()) === 'ok'
  }

  /** Panel status line; pass an explicit message (e.g. an error) to override the state line. */
  const renderInfo = (msg?: string, isError = false) => {
    info.textContent = msg ?? listInfo(getIndex()?.size, getCachedAt(), Date.now())
    info.classList.toggle('error', isError)
  }

  const showNotice = (msg: string, isError: boolean) => {
    if (noticeTimer !== undefined) clearTimeout(noticeTimer)
    notice.textContent = msg
    notice.classList.toggle('error', isError)
    notice.hidden = false
    // errors linger longer; replacement clears the pending timer so notices never race
    noticeTimer = window.setTimeout(() => (notice.hidden = true), isError ? 8000 : 4000)
  }
  notice.addEventListener('click', () => (notice.hidden = true))

  /**
   * Panel open → the outcome lands in the panel's status line (a notice would
   * overlap the panel); panel closed → transient notice below the topbar.
   */
  const showOutcome = (outcome: RefreshOutcome) => {
    const isError = outcome.kind !== 'updated'
    const msg = refreshMessage(outcome, getCachedAt())
    if (panel.hidden) showNotice(msg, isError)
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

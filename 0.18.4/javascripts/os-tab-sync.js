// Auto-pick OS-specific content tabs and remember the user's choice across pages.
//
// Tabs authored as `=== "Linux" / "macOS" / "Windows"` render (with
// pymdownx.tabbed alternate_style) as radio inputs grouped by name.
// On first visit we click the tab matching navigator heuristics; on later
// visits we restore the user's last manual choice. Material's
// `content.tabs.link` feature then syncs every other matching group on the
// page automatically.
(function () {
  const STORAGE_KEY = 'cuvisDocs.osTab';
  const OS_LABELS = ['Linux', 'macOS', 'Windows'];

  function detectOs() {
    const ua = navigator.userAgent || '';
    const platform = (navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || '';
    // Android UA contains "Linux" — skip to avoid mislabeling mobile visitors.
    if (/Android/i.test(ua)) return null;
    if (/Mac|iPhone|iPad|iPod/i.test(ua) || /Mac/i.test(platform)) return 'macOS';
    if (/Windows/i.test(ua) || /Win/i.test(platform)) return 'Windows';
    if (/Linux|X11/i.test(ua) || /Linux/i.test(platform)) return 'Linux';
    return null;
  }

  function readStoredPreference() {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored && OS_LABELS.indexOf(stored) !== -1) return stored;
    } catch (_) {}
    return null;
  }

  function writeStoredPreference(value) {
    try {
      localStorage.setItem(STORAGE_KEY, value);
    } catch (_) {}
  }

  function getPreferredOs() {
    return readStoredPreference() || detectOs();
  }

  function applyPreference(os) {
    if (!os) return;
    const labels = document.querySelectorAll('.tabbed-labels > label');
    labels.forEach(function (label) {
      if (label.textContent.trim() !== os) return;
      const id = label.getAttribute('for');
      const input = id && document.getElementById(id);
      if (input && !input.checked) {
        // click() flips the radio and fires change — Material's tab-link
        // listener picks it up and syncs sibling groups on the same page.
        input.click();
      }
    });
  }

  // Single document-level listener — survives Material's instant navigation
  // since `document` is reused across page swaps.
  document.addEventListener('click', function (event) {
    const label = event.target.closest && event.target.closest('.tabbed-labels > label');
    if (!label) return;
    const text = label.textContent.trim();
    if (OS_LABELS.indexOf(text) === -1) return;
    writeStoredPreference(text);
  });

  function run() {
    applyPreference(getPreferredOs());
  }

  // Material exposes a document$ observable that fires on initial load AND
  // on every instant-navigation page swap. Prefer that when available.
  if (typeof document$ !== 'undefined' && document$ && typeof document$.subscribe === 'function') {
    document$.subscribe(run);
  } else if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', run);
  } else {
    run();
  }
})();

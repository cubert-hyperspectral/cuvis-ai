/* Client-side filter for /catalogs/nodes/.
   One-row toolbar: search + foldable facet buttons (Category/Tags/Source) with
   active-count badges, a removable active-filter strip, a state-aware Clear,
   and an empty state. Cards carry data-category, data-tags, data-source,
   data-search; free text matches data-search. Tags combine OR within the
   facet, AND across facets. State persists in location.hash so
   /catalogs/nodes/#tag=hsi&category=transform&source=plugin restores a view
   (panels stay folded; badges and the strip carry the restored state). */

(function () {
  "use strict";

  const FACETS = [
    { key: "categories", attr: "category", label: "Category", noun: "category" },
    { key: "tags", attr: "tag", label: "Tags", noun: "tag" },
    { key: "sources", attr: "source", label: "Source", noun: "source" },
  ];

  function init() {
    const grid = document.getElementById("node-catalog-grid");
    if (!grid) return;
    // init() runs from both the readyState path below and the document$
    // replay; without this guard every listener binds twice and a chip click
    // toggles on+off in one go.
    if (grid.dataset.filterBound) return;
    grid.dataset.filterBound = "1";

    const cards = Array.from(grid.querySelectorAll(".node-row"));
    const container = document.querySelector(".node-filter");
    const searchInput = document.getElementById("node-filter-search");
    const countEl = document.getElementById("node-filter-count");
    const statusEl = document.getElementById("node-filter-status");
    const resetBtn = document.getElementById("node-filter-reset");
    const stripEl = document.getElementById("node-filter-active");
    const emptyEl = document.getElementById("node-filter-empty");
    const emptyMsgEl = document.getElementById("node-filter-empty-msg");
    const emptyResetBtn = document.getElementById("node-filter-empty-reset");
    const toggles = Array.from(document.querySelectorAll(".filter-group-toggle"));

    const facets = FACETS.map((f) => ({
      ...f,
      set: new Set(),
      panel: document.getElementById("node-filter-" + f.key),
      toggle: toggles.find((t) => t.dataset.panel === "node-filter-" + f.key) || null,
      chips: Array.from(
        document.querySelectorAll("#node-filter-" + f.key + " .filter-chip"),
      ),
    }));
    const state = { query: "" };
    facets.forEach((f) => (state[f.key] = f.set));

    let statusTimer = null;

    function safeDecode(raw) {
      // A hand-edited hash like "#q=%" must not throw and kill the filter.
      try {
        return decodeURIComponent(raw);
      } catch (_err) {
        return null;
      }
    }

    function readHash() {
      const hash = window.location.hash.replace(/^#/, "");
      if (!hash) return;
      hash.split("&").forEach((pair) => {
        const [key, raw] = pair.split("=");
        if (!raw) return;
        const value = safeDecode(raw);
        if (value === null) return;
        if (key === "q") state.query = value;
        else {
          const facet = facets.find((f) => f.attr === key);
          if (facet) value.split(",").forEach((v) => v && facet.set.add(v));
        }
      });
    }

    function writeHash() {
      const parts = [];
      if (state.query) parts.push("q=" + encodeURIComponent(state.query));
      facets.forEach((f) => {
        if (f.set.size) parts.push(f.attr + "=" + Array.from(f.set).join(","));
      });
      const next = parts.length ? "#" + parts.join("&") : window.location.pathname + window.location.search;
      history.replaceState(null, "", next);
    }

    function chipLabel(facet, value) {
      const chip = facet.chips.find((c) => c.dataset[facet.attr] === value);
      if (!chip) return value; // stale hash value: show it raw so it stays removable
      const label = chip.querySelector(".chip-label");
      return (label ? label.textContent : chip.textContent).trim();
    }

    function applyChipActive() {
      facets.forEach((f) => {
        f.chips.forEach((chip) => {
          const active = f.set.has(chip.dataset[f.attr]);
          chip.classList.toggle("active", active);
          chip.setAttribute("aria-pressed", active ? "true" : "false");
        });
      });
      if (searchInput && searchInput.value !== state.query) searchInput.value = state.query;
    }

    function applyBadges() {
      facets.forEach((f) => {
        if (!f.toggle) return;
        const badge = f.toggle.querySelector(".filter-group-badge");
        if (badge) {
          badge.textContent = f.set.size ? String(f.set.size) : "";
          badge.classList.toggle("is-empty", f.set.size === 0);
        }
        f.toggle.setAttribute(
          "aria-label",
          f.label + " filters" + (f.set.size ? ", " + f.set.size + " selected" : ""),
        );
      });
    }

    function applyStrip() {
      if (!stripEl) return;
      // Built with DOM APIs, never innerHTML: stale hash values are
      // user-controlled input.
      stripEl.textContent = "";
      let total = 0;
      facets.forEach((f) => {
        Array.from(f.set).forEach((value) => {
          total += 1;
          const chip = document.createElement("button");
          chip.type = "button";
          chip.className = "active-chip";
          chip.textContent = chipLabel(f, value);
          chip.setAttribute("aria-label", "Remove " + f.noun + " filter: " + value);
          chip.addEventListener("click", () => {
            const chipsBefore = Array.from(stripEl.children);
            const idx = chipsBefore.indexOf(chip);
            f.set.delete(value);
            apply();
            const chipsNow = stripEl.querySelectorAll("button");
            if (chipsNow.length) chipsNow[Math.min(idx, chipsNow.length - 1)].focus();
            else if (f.toggle) f.toggle.focus();
          });
          stripEl.appendChild(chip);
        });
      });
      stripEl.hidden = total === 0;
    }

    function anyActive() {
      return Boolean(state.query) || facets.some((f) => f.set.size > 0);
    }

    function apply() {
      const q = state.query.trim().toLowerCase();
      let visible = 0;
      cards.forEach((card) => {
        const cat = card.dataset.category || "";
        const tags = (card.dataset.tags || "").split(/\s+/).filter(Boolean);
        const source = card.dataset.source || "";
        const search = card.dataset.search || "";
        const catOk = state.categories.size === 0 || state.categories.has(cat);
        const tagOk =
          state.tags.size === 0 ||
          Array.from(state.tags).some((t) => tags.includes(t));
        const srcOk = state.sources.size === 0 || state.sources.has(source);
        const queryOk = !q || search.indexOf(q) !== -1;
        const hidden = !(catOk && tagOk && srcOk && queryOk);
        card.classList.toggle("is-hidden", hidden);
        if (!hidden) visible += 1;
      });
      const countText = visible === cards.length
        ? cards.length + " items"
        : visible + " of " + cards.length + " items";
      if (countEl) countEl.textContent = countText;
      if (statusEl) {
        // Debounce announcements so screen readers speak once per pause, not
        // once per keystroke.
        clearTimeout(statusTimer);
        statusTimer = setTimeout(() => { statusEl.textContent = countText; }, 400);
      }
      if (emptyEl) {
        emptyEl.hidden = visible !== 0;
        if (emptyMsgEl) {
          emptyMsgEl.textContent = q
            ? 'No items match "' + state.query.trim() + '".'
            : "No items match your search and filters.";
        }
      }
      if (resetBtn) resetBtn.hidden = !anyActive();
      applyChipActive();
      applyBadges();
      applyStrip();
      writeHash();
    }

    function closeAllPanels() {
      facets.forEach((f) => {
        if (f.panel) f.panel.hidden = true;
        if (f.toggle) f.toggle.setAttribute("aria-expanded", "false");
      });
    }

    facets.forEach((f) => {
      if (!f.toggle || !f.panel) return;
      f.toggle.addEventListener("click", () => {
        const wasHidden = f.panel.hidden;
        closeAllPanels();
        if (wasHidden) {
          f.panel.hidden = false;
          f.toggle.setAttribute("aria-expanded", "true");
        }
      });
      f.chips.forEach((chip) => {
        chip.addEventListener("click", () => {
          const value = chip.dataset[f.attr];
          if (f.set.has(value)) f.set.delete(value);
          else f.set.add(value);
          apply(); // panel stays open: multi-select should not re-fold the facet
        });
      });
    });

    if (container) {
      container.addEventListener("keydown", (e) => {
        if (e.key !== "Escape") return;
        const open = facets.find((f) => f.panel && !f.panel.hidden);
        if (!open) return;
        closeAllPanels();
        if (open.toggle) open.toggle.focus();
      });
    }

    if (searchInput) {
      searchInput.addEventListener("input", (e) => { state.query = e.target.value; apply(); });
    }

    function resetAll() {
      state.query = "";
      facets.forEach((f) => f.set.clear());
      apply();
      if (searchInput) searchInput.focus();
    }
    if (resetBtn) resetBtn.addEventListener("click", resetAll);
    if (emptyResetBtn) emptyResetBtn.addEventListener("click", resetAll);

    readHash();
    apply();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // mkdocs-material navigation.instant swaps the body without a full reload;
  // re-bind on each page change.
  if (window.document$) {
    window.document$.subscribe(init);
  }
})();

/* Client-side filter for /catalogs/nodes/.
   Cards carry data-category, data-tags, data-source, data-search; chips toggle
   facets; free text matches data-search. State persists in location.hash so
   /catalogs/nodes/#tag=hyperspectral&category=transform&source=plugin
   restores a view. */

(function () {
  "use strict";

  function init() {
    const grid = document.getElementById("node-catalog-grid");
    if (!grid) return;
    const cards = Array.from(grid.querySelectorAll(".node-row"));
    const searchInput = document.getElementById("node-filter-search");
    const catChips = Array.from(document.querySelectorAll("#node-filter-categories .filter-chip"));
    const tagChips = Array.from(document.querySelectorAll("#node-filter-tags .filter-chip"));
    const srcChips = Array.from(document.querySelectorAll("#node-filter-sources .filter-chip"));
    const countEl = document.getElementById("node-filter-count");
    const resetBtn = document.getElementById("node-filter-reset");

    const state = { query: "", categories: new Set(), tags: new Set(), sources: new Set() };

    function readHash() {
      const hash = window.location.hash.replace(/^#/, "");
      if (!hash) return;
      hash.split("&").forEach((pair) => {
        const [key, raw] = pair.split("=");
        if (!raw) return;
        const value = decodeURIComponent(raw);
        if (key === "q") state.query = value;
        else if (key === "category") value.split(",").forEach((v) => v && state.categories.add(v));
        else if (key === "tag") value.split(",").forEach((v) => v && state.tags.add(v));
        else if (key === "source") value.split(",").forEach((v) => v && state.sources.add(v));
      });
    }

    function writeHash() {
      const parts = [];
      if (state.query) parts.push("q=" + encodeURIComponent(state.query));
      if (state.categories.size) parts.push("category=" + Array.from(state.categories).join(","));
      if (state.tags.size) parts.push("tag=" + Array.from(state.tags).join(","));
      if (state.sources.size) parts.push("source=" + Array.from(state.sources).join(","));
      const next = parts.length ? "#" + parts.join("&") : window.location.pathname + window.location.search;
      history.replaceState(null, "", next);
    }

    function applyChipActive() {
      catChips.forEach((chip) => chip.classList.toggle("active", state.categories.has(chip.dataset.category)));
      tagChips.forEach((chip) => chip.classList.toggle("active", state.tags.has(chip.dataset.tag)));
      srcChips.forEach((chip) => chip.classList.toggle("active", state.sources.has(chip.dataset.source)));
      if (searchInput && searchInput.value !== state.query) searchInput.value = state.query;
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
          Array.from(state.tags).every((t) => tags.includes(t));
        const srcOk = state.sources.size === 0 || state.sources.has(source);
        const queryOk = !q || search.indexOf(q) !== -1;
        const hidden = !(catOk && tagOk && srcOk && queryOk);
        card.classList.toggle("is-hidden", hidden);
        if (!hidden) visible += 1;
      });
      if (countEl) {
        countEl.textContent = visible === cards.length
          ? cards.length + " items"
          : visible + " of " + cards.length + " items";
      }
      applyChipActive();
      writeHash();
    }

    function toggle(setObj, value) {
      if (setObj.has(value)) setObj.delete(value);
      else setObj.add(value);
    }

    catChips.forEach((chip) => {
      chip.addEventListener("click", () => { toggle(state.categories, chip.dataset.category); apply(); });
    });
    tagChips.forEach((chip) => {
      chip.addEventListener("click", () => { toggle(state.tags, chip.dataset.tag); apply(); });
    });
    srcChips.forEach((chip) => {
      chip.addEventListener("click", () => { toggle(state.sources, chip.dataset.source); apply(); });
    });
    if (searchInput) {
      searchInput.addEventListener("input", (e) => { state.query = e.target.value; apply(); });
    }
    if (resetBtn) {
      resetBtn.addEventListener("click", () => {
        state.query = "";
        state.categories.clear();
        state.tags.clear();
        state.sources.clear();
        apply();
      });
    }

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

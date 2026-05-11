/* Client-side filter for /catalogs/nodes/.
   Cards carry data-category, data-tags, data-search; chips toggle facets;
   free text matches data-search. State persists in location.hash so
   /catalogs/nodes/#tag=hyperspectral&category=transform restores a view. */

(function () {
  "use strict";

  function init() {
    const grid = document.getElementById("node-catalog-grid");
    if (!grid) return;
    const cards = Array.from(grid.querySelectorAll(".node-row"));
    const searchInput = document.getElementById("node-filter-search");
    const catChips = Array.from(document.querySelectorAll("#node-filter-categories .filter-chip"));
    const tagChips = Array.from(document.querySelectorAll("#node-filter-tags .filter-chip"));
    const countEl = document.getElementById("node-filter-count");
    const resetBtn = document.getElementById("node-filter-reset");

    const state = { query: "", categories: new Set(), tags: new Set() };

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
      });
    }

    function writeHash() {
      const parts = [];
      if (state.query) parts.push("q=" + encodeURIComponent(state.query));
      if (state.categories.size) parts.push("category=" + Array.from(state.categories).join(","));
      if (state.tags.size) parts.push("tag=" + Array.from(state.tags).join(","));
      const next = parts.length ? "#" + parts.join("&") : window.location.pathname + window.location.search;
      history.replaceState(null, "", next);
    }

    function applyChipActive() {
      catChips.forEach((chip) => chip.classList.toggle("active", state.categories.has(chip.dataset.category)));
      tagChips.forEach((chip) => chip.classList.toggle("active", state.tags.has(chip.dataset.tag)));
      if (searchInput && searchInput.value !== state.query) searchInput.value = state.query;
    }

    function apply() {
      const q = state.query.trim().toLowerCase();
      let visible = 0;
      cards.forEach((card) => {
        const cat = card.dataset.category || "";
        const tags = (card.dataset.tags || "").split(/\s+/).filter(Boolean);
        const search = card.dataset.search || "";
        const catOk = state.categories.size === 0 || state.categories.has(cat);
        const tagOk =
          state.tags.size === 0 ||
          Array.from(state.tags).every((t) => tags.includes(t));
        const queryOk = !q || search.indexOf(q) !== -1;
        const hidden = !(catOk && tagOk && queryOk);
        card.classList.toggle("is-hidden", hidden);
        if (!hidden) visible += 1;
      });
      if (countEl) {
        countEl.textContent = visible === cards.length
          ? cards.length + " nodes"
          : visible + " of " + cards.length + " nodes";
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
    if (searchInput) {
      searchInput.addEventListener("input", (e) => { state.query = e.target.value; apply(); });
    }
    if (resetBtn) {
      resetBtn.addEventListener("click", () => {
        state.query = "";
        state.categories.clear();
        state.tags.clear();
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

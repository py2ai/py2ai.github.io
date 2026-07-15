document.addEventListener("DOMContentLoaded", function () {
  const input = document.getElementById("post-search");
  if (!input) return;
  const btn = document.getElementById("search-btn");
  const noResults = document.getElementById("cat-no-results");
  const tilesWrap = document.getElementById("cat-tiles");
  const sections = document.querySelectorAll(".cat-section");
  const items = document.querySelectorAll(".post-item");

  function filterPosts() {
    const query = (input.value || "").trim().toLowerCase();

    // No query: show everything.
    if (!query) {
      items.forEach((li) => (li.style.display = ""));
      sections.forEach((s) => (s.style.display = ""));
      if (tilesWrap) tilesWrap.style.display = "";
      if (noResults) noResults.classList.remove("show");
      return;
    }

    // With a query: hide the category index tiles (they're for navigation,
    // not search results) and filter posts, hiding any now-empty section.
    if (tilesWrap) tilesWrap.style.display = "none";

    let totalVisible = 0;
    sections.forEach((section) => {
      let sectionHasMatch = false;
      section.querySelectorAll(".post-item").forEach((li) => {
        const titleEl = li.querySelector("a");
        const descEl = li.querySelector(".post-desc");
        const title = titleEl ? titleEl.textContent.toLowerCase() : "";
        const desc = descEl ? descEl.textContent.toLowerCase() : "";
        const match = title.includes(query) || desc.includes(query);
        li.style.display = match ? "" : "none";
        if (match) {
          sectionHasMatch = true;
          totalVisible++;
        }
      });
      section.style.display = sectionHasMatch ? "" : "none";
    });

    if (noResults) {
      noResults.classList.toggle("show", totalVisible === 0);
    }
  }

  ["input", "keyup", "change", "search"].forEach((ev) =>
    input.addEventListener(ev, filterPosts)
  );
  if (btn) btn.addEventListener("click", filterPosts);
});
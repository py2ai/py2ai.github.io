document.addEventListener("DOMContentLoaded", function () {

  function filterPosts() {
    const input = document.getElementById("post-search");
    if (!input) return;

    const query = input.value.trim().toLowerCase();
    const items = document.querySelectorAll(".post-item");

    items.forEach(function (li) {
      const titleEl = li.querySelector("a");
      const descEl = li.querySelector(".post-desc");

      const title = titleEl ? titleEl.textContent.toLowerCase() : "";
      const desc = descEl ? descEl.textContent.toLowerCase() : "";

      li.style.display =
        title.includes(query) || desc.includes(query)
          ? ""
          : "none";
    });
  }

  const searchInput = document.getElementById("post-search");
  const searchBtn = document.getElementById("search-btn");

  if (!searchInput || !searchBtn) return;

  ["input", "keyup", "change", "search"].forEach(function (event) {
    searchInput.addEventListener(event, filterPosts);
  });

  searchBtn.addEventListener("click", filterPosts);

});

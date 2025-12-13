---
layout: page
title: Search All Blog Posts
permalink: /categories/
---

<h2 style="margin-bottom: 20px; text-align:center;">All Blog Posts</h2>

<!-- 🔍 Search Bar with Button -->
<div style="text-align:center; margin-bottom: 20px;">

  <input id="post-search" 
         type="text" 
         placeholder="Search posts..." 
         style="padding: 10px 15px; width: 55%; max-width: 350px; font-size: 1rem; border-radius: 6px; border: 1px solid #ccc;">

  <button id="search-btn"
          style="padding: 10px 15px; font-size: 1rem; border-radius: 6px; background:#007acc; color:white; border:none; cursor:pointer;">
    Search
  </button>

</div>

<ul id="post-list" style="list-style: none; padding-left: 0;">
  {% assign all_posts = site.posts | sort: "date" | reverse %}
  {% for post in all_posts %}
    <li class="post-item" style="margin-bottom: 12px;">
      <a href="{{ site.baseurl }}{{ post.url }}" style="font-size: 1.1rem; color: #007acc; text-decoration: none;">
        {{ post.title }}
      </a>
      <br>
      <small style="color: #777;">
        📅 {{ post.date | date: "%B %d, %Y" }}
      </small>
      {% if post.description %}
        <p class="post-desc" style="margin: 5px 0 0; color: #555;">{{ post.description }}</p>
      {% endif %}
    </li>
  {% endfor %}
</ul>

<script>
document.addEventListener("DOMContentLoaded", function () {

  function filterPosts() {
    const input = document.getElementById("post-search");
    if (!input) return;

    const q = input.value.toLowerCase();
    const items = document.querySelectorAll(".post-item");

    items.forEach(li => {
      const titleEl = li.querySelector("a");
      const descEl = li.querySelector(".post-desc");

      const title = titleEl ? titleEl.textContent.toLowerCase() : "";
      const desc = descEl ? descEl.textContent.toLowerCase() : "";

      li.style.display = (title.includes(q) || desc.includes(q)) ? "" : "none";
    });
  }

  const searchInput = document.getElementById("post-search");
  const searchBtn = document.getElementById("search-btn");

  if (!searchInput || !searchBtn) return;

  // Mobile + desktop safe events
  ["input", "keyup", "change", "search"].forEach(evt =>
    searchInput.addEventListener(evt, filterPosts)
  );

  searchBtn.addEventListener("click", filterPosts);

});
</script>

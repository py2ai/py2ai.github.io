---
layout: page
title: All Posts
permalink: /all-posts/
description: "Browse all PyShine blog posts — tutorials, AI tools, open-source projects, and developer guides."
---

<style>
  .all-posts-page { color: var(--text-color); }
  .all-posts-page h2 { color: var(--heading-color); }

  .ap-search {
    display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;
    margin: 0 auto 28px; max-width: 520px;
  }
  .ap-search input {
    flex: 1 1 240px; padding: 10px 15px; font-size: 1rem;
    background: var(--input-bg); color: var(--input-text);
    border: 1px solid var(--input-border); border-radius: 6px;
  }
  .ap-search input::placeholder { color: var(--text-secondary); }
  .ap-search input:focus {
    border-color: var(--link-color); outline: none;
    box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.3);
  }
  .ap-search button {
    padding: 10px 18px; font-size: 1rem; border-radius: 6px; cursor: pointer;
    background: var(--button-bg); color: var(--button-text);
    border: 1px solid var(--border-color);
  }
  .ap-search button:hover { border-color: var(--link-color); opacity: 0.92; }

  .ap-list { list-style: none; padding-left: 0; max-width: 800px; margin: 0 auto; }
  .ap-list .ap-item {
    margin-bottom: 14px; padding: 12px 16px;
    background: var(--card-bg); border: 1px solid var(--border-color);
    border-radius: 8px; transition: border-color 0.2s ease;
  }
  .ap-list .ap-item:hover { border-color: var(--link-color); }
  .ap-list .ap-item a {
    font-size: 1.05rem; font-weight: 600; color: var(--link-color);
    text-decoration: none;
  }
  .ap-list .ap-item a:hover { text-decoration: underline; }
  .ap-list .ap-item .ap-date { color: var(--text-secondary); font-size: 0.85rem; }
  .ap-list .ap-item .ap-desc {
    margin: 4px 0 0; color: var(--text-color); font-size: 0.9rem;
    line-height: 1.5;
  }
  .ap-list .ap-item .ap-tags {
    margin-top: 6px;
  }
  .ap-list .ap-item .ap-tag {
    display: inline-block; padding: 2px 8px; margin-right: 4px;
    font-size: 0.75rem; border-radius: 999px;
    background: var(--bg-secondary); color: var(--text-secondary);
    border: 1px solid var(--border-color);
  }

  .ap-no-results { display: none; text-align: center; padding: 40px 10px; color: var(--text-secondary); }
  .ap-no-results.show { display: block; }

  .ap-count {
    text-align: center; color: var(--text-secondary);
    margin-bottom: 20px; font-size: 0.9rem;
  }
</style>

<div class="all-posts-page">
  <h2 style="margin-bottom: 8px; text-align:center;">All Posts</h2>
  <p class="ap-count">{{ site.posts | size }} posts total</p>

  <div class="ap-search">
    <input id="ap-search-input" type="search" placeholder="Search all posts..." aria-label="Search all posts" />
    <button id="ap-search-btn" type="button">Search</button>
  </div>

  <p class="ap-no-results" id="ap-no-results">
    No posts match your search. Try a different keyword.
  </p>

  <ul class="ap-list" id="ap-list">
  {% assign sorted_posts = site.posts | sort: "date" | reverse %}
  {% for post in sorted_posts %}
    <li class="ap-item" data-title="{{ post.title | downcase }}" data-desc="{{ post.description | default: '' | downcase }}">
      <a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a>
      <br />
      <small class="ap-date">{{ post.date | date: "%B %d, %Y" }}</small>
      {% if post.description %}
        <p class="ap-desc">{{ post.description | truncatewords: 25 }}</p>
      {% endif %}
      {% if post.tags.size > 0 %}
        <div class="ap-tags">
        {% for tag in post.tags limit: 5 %}
          <span class="ap-tag">{{ tag }}</span>
        {% endfor %}
        </div>
      {% endif %}
    </li>
  {% endfor %}
  </ul>
</div>

<script>
document.addEventListener("DOMContentLoaded", function () {
  const input = document.getElementById("ap-search-input");
  if (!input) return;
  const btn = document.getElementById("ap-search-btn");
  const noResults = document.getElementById("ap-no-results");
  const items = document.querySelectorAll(".ap-item");

  function filter() {
    const query = (input.value || "").trim().toLowerCase();
    let visible = 0;
    items.forEach(function (li) {
      if (!query) { li.style.display = ""; visible++; return; }
      const title = li.getAttribute("data-title") || "";
      const desc = li.getAttribute("data-desc") || "";
      const match = title.includes(query) || desc.includes(query);
      li.style.display = match ? "" : "none";
      if (match) visible++;
    });
    if (noResults) noResults.classList.toggle("show", visible === 0);
  }

  ["input", "keyup", "change", "search"].forEach(function (ev) {
    input.addEventListener(ev, filter);
  });
  if (btn) btn.addEventListener("click", filter);
});
</script>
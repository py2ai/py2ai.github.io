---
layout: page
title: All Categories
permalink: /categories/
---

<style>
  /* Theme-aware category page styles — uses the site's CSS variables
     so the search box, tiles, and sections adapt to light/dark themes. */
  .cat-page { color: var(--text-color); }
  .cat-page h2 { color: var(--heading-color); }
  .cat-page .cat-meta { color: var(--text-secondary); }

  .cat-search {
    display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;
    margin: 0 auto 28px; max-width: 520px;
  }
  .cat-search input {
    flex: 1 1 240px; padding: 10px 15px; font-size: 1rem;
    background: var(--input-bg); color: var(--input-text);
    border: 1px solid var(--input-border); border-radius: 6px;
  }
  .cat-search input::placeholder { color: var(--text-secondary); }
  .cat-search input:focus {
    border-color: var(--link-color); outline: none;
    box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.3);
  }
  .cat-search button {
    padding: 10px 18px; font-size: 1rem; border-radius: 6px; cursor: pointer;
    background: var(--button-bg); color: var(--button-text);
    border: 1px solid var(--border-color);
  }
  .cat-search button:hover { border-color: var(--link-color); opacity: 0.92; }

  .cat-tiles { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; margin-bottom: 32px; }
  .cat-tile {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 14px; border-radius: 999px; text-decoration: none;
    font-size: 0.95rem;
    background: var(--card-bg); color: var(--link-color);
    border: 1px solid var(--border-color);
  }
  .cat-tile:hover { border-color: var(--link-color); }
  .cat-tile .cat-count {
    background: var(--link-color); color: var(--bg-color);
    border-radius: 999px; padding: 1px 8px; font-size: 0.78rem;
  }

  .cat-section { margin-bottom: 36px; }
  .cat-section h3 {
    border-bottom: 2px solid var(--border-color);
    padding-bottom: 6px; margin-bottom: 14px;
    color: var(--heading-color);
  }
  .cat-section h3 .cat-count-inline { color: var(--text-secondary); font-weight: 400; font-size: 0.85rem; }
  .cat-section h3 .cat-top { float: right; font-size: 0.78rem; color: var(--link-color); text-decoration: none; }

  .cat-posts { list-style: none; padding-left: 0; }
  .cat-posts .post-item { margin-bottom: 12px; }
  .cat-posts .post-item a { font-size: 1.05rem; color: var(--link-color); text-decoration: none; }
  .cat-posts .post-item a:hover { text-decoration: underline; }
  .cat-posts .post-item .post-date { color: var(--text-secondary); }
  .cat-posts .post-item .post-desc { margin: 4px 0 0; color: var(--text-color); }

  .cat-no-results { display: none; text-align: center; padding: 40px 10px; color: var(--text-secondary); }
  .cat-no-results.show { display: block; }
</style>

<div class="cat-page">
  <h2 style="margin-bottom: 8px; text-align:center;">Browse by Category</h2>
  <p class="cat-meta" style="text-align:center; margin-bottom: 24px;">
    {{ site.posts | size }} posts across {{ site.categories | size }} categories.
  </p>

  <div class="cat-search">
    <input id="post-search" type="search" placeholder="Search posts..." aria-label="Search posts" />
    <button id="search-btn" type="button">Search</button>
  </div>

  <!-- Category index: one tile per category -->
  <div class="cat-tiles" id="cat-tiles">
  {% assign cats = site.categories | sort %}
  {% for cat in cats %}
    {% assign cat_name = cat[0] %}
    {% assign cat_posts = cat[1] %}
    {% assign count = cat_posts | size %}
    <a class="cat-tile" href="#cat-{{ cat_name | slugify }}" data-cat="{{ cat_name | slugify }}">
      {{ cat_name }}
      <span class="cat-count">{{ count }}</span>
    </a>
  {% endfor %}
  </div>

  <p class="cat-no-results" id="cat-no-results">
    No posts match your search. Try a different keyword, or
    <a href="{{ '/categories/' | relative_url }}" style="color: var(--link-color);">browse all categories</a>.
  </p>

  <!-- Per-category sections -->
  {% for cat in site.categories | sort %}
    {% assign cat_name = cat[0] %}
    {% assign cat_posts = cat[1] | sort: "date" | reverse %}
    <section class="cat-section" data-cat="{{ cat_name | slugify }}">
      <h3 id="cat-{{ cat_name | slugify }}">
        {{ cat_name }}
        <span class="cat-count-inline">({{ cat_posts | size }} posts)</span>
        <a class="cat-top" href="#top">↑ top</a>
      </h3>
      <ul class="cat-posts">
      {% for post in cat_posts %}
        <li class="post-item">
          <a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a>
          <br />
          <small class="post-date">📅 {{ post.date | date: "%B %d, %Y" }}</small>
          {% if post.description %}
            <p class="post-desc">{{ post.description }}</p>
          {% endif %}
        </li>
      {% endfor %}
      </ul>
    </section>
  {% endfor %}

  <div id="top"></div>
</div>

<script src="{{ '/assets/js/post-search.js' | relative_url }}"></script>
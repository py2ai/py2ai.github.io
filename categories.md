---
layout: page
title: All Categories
permalink: /categories/
---

<h2 style="margin-bottom: 8px; text-align:center;">Browse by Category</h2>
<p style="text-align:center; color:#777; margin-bottom: 24px;">
  {{ site.posts | size }} posts across {{ site.categories | size }} categories.
</p>

<div style="text-align:center; margin-bottom: 28px;">
  <input
    id="post-search"
    type="search"
    placeholder="Search posts..."
    style="padding: 10px 15px; width: 55%; max-width: 350px; font-size: 1rem; border-radius: 6px; border: 1px solid #ccc;"
  />
  <button
    id="search-btn"
    type="button"
    style="padding: 10px 15px; font-size: 1rem; border-radius: 6px; background:#007acc; color:white; border:none; cursor:pointer;"
  >
    Search
  </button>
</div>

<!-- Category index: one tile per category, sorted by post count -->
<div style="display:flex; flex-wrap:wrap; gap:10px; justify-content:center; margin-bottom:32px;">
{% assign cats = site.categories | sort %}
{% for cat in cats %}
  {% assign cat_name = cat[0] %}
  {% assign cat_posts = cat[1] %}
  {% assign count = cat_posts | size %}
  <a href="#cat-{{ cat_name | slugify }}"
     style="display:inline-flex; align-items:center; gap:6px; padding:6px 14px;
            border-radius:999px; background:#eef4ff; color:#1e40af; text-decoration:none;
            font-size:0.95rem; border:1px solid #dbeafe;">
    {{ cat_name }}
    <span style="background:#1e40af; color:#fff; border-radius:999px; padding:1px 8px; font-size:0.78rem;">{{ count }}</span>
  </a>
{% endfor %}
</div>

<!-- Per-category sections -->
{% for cat in site.categories | sort %}
  {% assign cat_name = cat[0] %}
  {% assign cat_posts = cat[1] | sort: "date" | reverse %}
  <section style="margin-bottom: 36px;">
    <h3 id="cat-{{ cat_name | slugify }}"
        style="border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; margin-bottom: 14px;">
      {{ cat_name }}
      <span style="color:#94a3b8; font-weight:400; font-size:0.85rem;">
        ({{ cat_posts | size }} posts)
      </span>
      <a href="#top" style="float:right; font-size:0.78rem; color:#007acc; text-decoration:none;">↑ top</a>
    </h3>
    <ul style="list-style:none; padding-left:0;">
    {% for post in cat_posts %}
      <li class="post-item" style="margin-bottom: 12px;">
        <a href="{{ site.baseurl }}{{ post.url }}"
           style="font-size: 1.05rem; color: #007acc; text-decoration: none;">
          {{ post.title }}
        </a>
        <br />
        <small style="color: #777;">
          📅 {{ post.date | date: "%B %d, %Y" }}
        </small>
        {% if post.description %}
          <p class="post-desc" style="margin: 4px 0 0; color: #555;">
            {{ post.description }}
          </p>
        {% endif %}
      </li>
    {% endfor %}
    </ul>
  </section>
{% endfor %}

<div id="top"></div>
<script src="{{ '/assets/js/post-search.js' | relative_url }}"></script>
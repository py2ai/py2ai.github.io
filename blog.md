---
layout: page
title: Blog
permalink: /blog/
description: "Latest PyShine posts — source tours of open-source projects, AI tools, tutorials, and developer guides."
---

<style>
  .blog-page { color: var(--text-color); }
  .blog-page h2 { color: var(--heading-color); }

  .blog-list { list-style: none; padding-left: 0; max-width: 800px; margin: 0 auto; }
  .blog-list .blog-item {
    margin-bottom: 14px; padding: 12px 16px;
    background: var(--card-bg); border: 1px solid var(--border-color);
    border-radius: 8px; transition: border-color 0.2s ease;
  }
  .blog-list .blog-item:hover { border-color: var(--link-color); }
  .blog-list .blog-item a {
    font-size: 1.05rem; font-weight: 600; color: var(--link-color);
    text-decoration: none;
  }
  .blog-list .blog-item a:hover { text-decoration: underline; }
  .blog-list .blog-item .blog-date { color: var(--text-secondary); font-size: 0.85rem; }
  .blog-list .blog-item .blog-desc {
    margin: 4px 0 0; color: var(--text-color); font-size: 0.9rem;
    line-height: 1.5;
  }

  .blog-count { text-align: center; color: var(--text-secondary); margin-bottom: 20px; font-size: 0.9rem; }
  .blog-more { text-align: center; margin: 24px 0 8px; }
  .blog-more a {
    display: inline-block; padding: 10px 22px; border-radius: 6px;
    background: var(--button-bg); color: var(--button-text);
    border: 1px solid var(--border-color); text-decoration: none; font-weight: 600;
  }
  .blog-more a:hover { border-color: var(--link-color); opacity: 0.92; }
</style>

<div class="blog-page">
  <h2 style="margin-bottom: 8px; text-align:center;">Blog</h2>
  <p class="blog-count">{{ site.posts | size }} posts total &mdash; showing the latest 30</p>

  <ul class="blog-list">
  {% assign sorted_posts = site.posts | sort: "date" | reverse %}
  {% assign latest_limit = 30 %}
  {% for post in sorted_posts limit: latest_limit %}
    <li class="blog-item">
      <a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a>
      <br />
      <small class="blog-date">{{ post.date | date: "%B %d, %Y" }}</small>
      {% if post.description %}
        <p class="blog-desc">{{ post.description | truncatewords: 25 }}</p>
      {% endif %}
    </li>
  {% endfor %}
  </ul>

  <p class="blog-more">
    <a href="/all-posts/">Browse all {{ site.posts | size }} posts &rarr;</a>
  </p>
</div>

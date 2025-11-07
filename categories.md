---
layout: page
title: All Posts
permalink: /categories/
---

<h2 style="margin-bottom: 20px;">All Blog Posts</h2>

<ul style="list-style: none; padding-left: 0;">
  {% assign all_posts = site.posts | sort: "date" | reverse %}
  {% for post in all_posts %}
    <li style="margin-bottom: 12px;">
      <a href="{{ site.baseurl }}{{ post.url }}" style="font-size: 1.1rem; color: #007acc; text-decoration: none;">
        {{ post.title }}
      </a>
      <br>
      <small style="color: #777;">
        📅 {{ post.date | date: "%B %d, %Y" }}
      </small>
      {% if post.description %}
        <p style="margin: 5px 0 0; color: #555;">{{ post.description }}</p>
      {% endif %}
    </li>
  {% endfor %}
</ul>

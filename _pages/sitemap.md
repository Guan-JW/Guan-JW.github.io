---
layout: archive
title: "Sitemap"
permalink: /sitemap/
author_profile: true
---

{% include base_path %}

A list of all the posts and pages found on the site. For you robots out there is an [XML version]({{ base_path }}/sitemap.xml) available for digesting as well.

<h2>Pages</h2>
{% for post in site.pages %}
  {% include archive-single.html %}
{% endfor %}

{% if site.blog_enabled %}
<h2>Blog</h2>
<p><a href="{{ '/blog/' | relative_url }}">All posts</a></p>
{% assign blog_posts = site.blog | where_exp: 'post', 'post.date <= site.time' | sort: 'date' | reverse %}
{% for post in blog_posts %}
  {% include archive-single.html %}
{% endfor %}
{% endif %}

{% capture written_label %}'None'{% endcapture %}

{% for collection in site.collections %}
{% unless collection.output == false or collection.label == "posts" or collection.label == "blog_pages" or collection.label == "blog" %}
  {% capture label %}{{ collection.label }}{% endcapture %}
  {% if label != written_label %}
  <h2>{{ label }}</h2>
  {% capture written_label %}{{ label }}{% endcapture %}
  {% endif %}
{% endunless %}
{% for post in collection.docs %}
  {% unless collection.output == false or collection.label == "posts" or collection.label == "blog_pages" or collection.label == "blog" %}
  {% include archive-single.html %}
  {% endunless %}
{% endfor %}
{% endfor %}

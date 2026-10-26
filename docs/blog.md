# Writing and publishing the blog

The blog is **off online by default**. You can keep updating the homepage,
publications, and other sections normally.

## Preview locally

From the repository directory, run:

```sh
bash bin/preview-blog
```

Open <http://localhost:4000/> for the homepage or
<http://localhost:4000/blog/> for the archive. This includes unfinished entries
and the sample draft, without changing the online switch. Stop with Ctrl+C.
If port 4000 is occupied, use `bash bin/preview-blog --port 4001` and open
<http://localhost:4001/>.

The preview combines `_config.yml`, `_config.dev.yml`, and
`_config.blog-preview.yml`. Its output goes into the ignored
`_site-blog-preview/` directory. Do not deploy that preview output.

To preview the normal site with the online switch respected:

```sh
bash bin/serve --config _config.yml,_config.dev.yml
```

Restart the server after changing configuration files.

## Write a post

Copy `_templates/blog-post.md` to `_blog/your-post-slug.md`. Replace the title,
date (`YYYY-MM-DD`), summary, tags, and body. Keep `published: false` while
writing. The URL will be `/blog/your-post-slug/`.

- Set `toc: true` for a table of contents generated from level 2 and 3 headings.
- Set `modified: YYYY-MM-DD` when updating an existing article.
- Use Markdown for code blocks, tables, images, and English or Chinese text.
- Use `$...$` or `$$...$$` for equations. The existing MathJax integration needs
  an internet connection in the browser.
- For images, use `images/blog/` and reference them with
  `![Description]({{ '/images/blog/filename.png' | relative_url }})`.
  Put downloadable files in `files/`. These ordinary asset directories publish
  independently of the blog switch, so add their contents when ready to share.
  Keep `_blog/` for Markdown entries.

Set `published: true` on each finished post you want to release. Missing
`published` defaults to false. Future-dated posts stay out of the normal build
until their date arrives; a new build is needed for them to appear.

`_blog/a-place-for-learning.md` is a sample draft for checking the layout. It
stays unpublished unless you explicitly mark it ready. The original theme
examples are preserved in `_examples/legacy-posts/` and never enter the blog.

## Turn the blog on or off online

In `_config.yml`, change this one line:

```yaml
blog_enabled             : &blog_enabled false
```

Change only `false` to `true`, leaving `&blog_enabled` in place. Then commit
and publish the site through your usual GitHub Pages workflow.

| Setting | Normal build | Local blog preview |
| --- | --- | --- |
| `false` | No Blog navigation, homepage preview, archive, article URLs, tag/category pages, or blog sitemap/feed entries | Blog and unfinished entries visible |
| `true` | Blog visible; only entries marked `published: true` with a date no later than today | Blog and unfinished entries visible |

Switch back to `false` and rebuild to remove the blog from the generated site.
The existing feed endpoint remains valid but has no blog entries while off.
The other homepage sections continue to build normally.

The switch controls the generated website. Files committed to a public GitHub
repository remain readable in the repository, including unfinished writing.

## Implementation and checks

Likes and comments are described in [the interaction setup guide](blog-comments.md).
Local preview includes temporary interactions; shared counts, public comments,
and email notifications require the separate service described there.

Blog Markdown lives in `_blog/`, a Jekyll collection, rather than `_posts/`.
Jekyll always forces its built-in posts collection to output; a separate
collection allows one switch to disable article output reliably, even for
entries marked ready. Both blog collections share the switch via a YAML anchor.
No custom Jekyll plugin or deployment workflow is required.

Run the publishing checks with:

```sh
python3 tests/check_blog.py
```

These build isolated copies in off, on, preview, and on-then-off modes, and
check direct URLs, sitemap/feed entries, draft handling, and homepage content.

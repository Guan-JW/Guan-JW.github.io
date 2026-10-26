# Likes, comments, and notification email

The blog integrates Waline for a shared article Like counter, guest comments,
replies, and moderation. GitHub Pages continues to serve the site; a separate
Waline server stores interactions and sends email. Nothing has been deployed or
connected yet: `blog_interactions.server_url` is intentionally empty.

## Local preview

Run `bash bin/preview-blog`, open an article, and scroll to **Likes & discussion**.
The preview contains a working Like toggle and comment form for checking the
layout. These are explicitly labeled preview interactions: they stay in memory,
reset on reload, and never load Waline, contact the live database, or send email.
This remains true even after you configure the production server URL. The real
Waline widget has its own reply/login/moderation controls; the preview is a simple
style demonstration, not a copy of that service.

## Connect the live service

1. Create a separate Vercel project using `services/blog-comments` as its root
   directory (or deploy that directory as a separate repository). Choose the
   **Other** framework preset and Node 22. The package pins Waline server 1.42.0;
   the browser client is pinned to 3.1.3.
2. Follow Waline's [official Vercel/database setup](https://waline.js.org/en/guide/deploy/vercel.html)
   to provision persistent storage and initialize its schema. The current guide
   uses Neon/PostgreSQL. Database credentials belong in Vercel environment
   variables, never the Jekyll config or browser JavaScript.
3. Set a long random `JWT_TOKEN` and configure SMTP in the project's environment
   variables using `services/blog-comments/.env.example` as the checklist.
   `AUTHOR_EMAIL=guanjw@ruc.edu.cn` is already the server default. `SMTP_USER` and
   `SMTP_PASS` authorize the **sender**; knowing the recipient address alone is
   insufficient. Use the sender's SMTP/app password. For port 465 set
   `SMTP_SECURE=true`; for a provider's STARTTLS port 587 use `false`.
4. Redeploy to apply the environment variables. Visit the server's
   `/ui/register` and register the owner account before exposing the widget;
   Waline makes its first registered account the administrator. Use `/ui` to
   manage comments thereafter.
5. Paste the deployed HTTPS server origin into `_config.yml`:

   ```yaml
   blog_interactions:
     enabled: true
     server_url: "https://YOUR-COMMENT-SERVICE.vercel.app"
   ```

   The server accepts the homepage domain; update `secureDomains` in `index.cjs`
   if the website later moves to a custom domain. Waline also allows localhost
   for development. No sender password or database secret belongs in this URL.

This does not publish the blog. Keep `blog_enabled: false` until ready. Once the
blog is enabled, live interactions appear only on published articles with a
configured service. Readers click **Load likes and comments** to load Waline's
client and contact the comment server. The article and table of contents remain
usable if the service or CDN is unavailable.

## Notifications and verification

The server's `postSave` hook emails **guanjw@ruc.edu.cn** for each newly saved
comment or reply, including owner comments and comments awaiting moderation.
Rejected submissions are not saved and do not trigger this hook. The hook avoids
Waline's usual owner-comment exception; built-in owner notifications are disabled
to prevent duplicate emails. Notification failures are logged without discarding
the saved comment; this setup has no durable email retry queue. See the
[official hooks](https://waline.js.org/en/reference/server/config.html#postsave-comment-pcomment)
and [SMTP settings](https://waline.js.org/en/guide/features/notification.html).
Actual delivery remains unverified until a real test succeeds.

After deployment, test with a staging server or a published article:

- Add a Like; check the shared count from a second browser.
- Post a visitor comment and reply; confirm both persist after reload and that
  notifications arrive at **guanjw@ruc.edu.cn**. Check spam and server SMTP logs
  if a message is missing. Owner comments should also trigger one notification.
- Remove the test comments through `/ui` and verify moderation works.
- Check the ordinary local preview still makes no requests to the live server.

Like counts are lightweight reactions, not verified unique-person vote totals.
Comments are public; email addresses are not displayed in the comment list.
The service stores comment metadata; optional visitor emails can receive reply
notifications under Waline's normal behavior.

## Controls and stable threads

- `blog_enabled: false`: removes blog pages and all their interactions.
- `blog_interactions.enabled: false`: hides likes/comments across the blog.
- `interactions: false` in an article's front matter: disables both for that post.
- `blog_interactions_preview: true`: local-only, temporary demonstration mode.

The canonical article path, e.g. `/blog/quantization-in-serving-stacks/`, is the
thread identifier. Query strings and fragments do not create separate threads.
Keep published slugs stable; if changing one, migrate its Waline records too.

Server source and environment files are excluded from Jekyll output. The existing
publishing checks also verify the preview/live boundary and exclusion of the
server directory. Interaction and notification logic checks run with
`node --test tests/blog-interactions.test.cjs`.

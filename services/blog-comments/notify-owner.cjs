'use strict';

function createOwnerNotifier(env, createTransport) {
  var transport;
  return async function notifyOwner(comment) {
    if (!env.SMTP_HOST || !env.SMTP_USER || !env.SMTP_PASS) {
      throw Object.assign(new Error('SMTP is not configured'), { code: 'SMTP_NOT_CONFIGURED' });
    }
    if (!transport) {
      transport = createTransport({
        host: env.SMTP_HOST,
        port: Number(env.SMTP_PORT || 465),
        secure: env.SMTP_SECURE !== 'false',
        auth: { user: env.SMTP_USER, pass: env.SMTP_PASS },
        connectionTimeout: 10000,
        socketTimeout: 15000
      });
    }
    var site = new URL(env.SITE_URL || 'https://guan-jw.github.io');
    var post = new URL(comment.url || '/', site);
    // The link must remain on our own site, even for direct API submissions.
    if (post.origin !== site.origin || post.protocol !== 'https:') post = site;
    post.hash = String(comment.objectId || 'blog-discussion-title');
    return transport.sendMail({
      from: { name: env.SENDER_NAME || "Jiawei's Blog", address: env.SENDER_EMAIL || env.SMTP_USER },
      to: env.AUTHOR_EMAIL || 'guanjw@ruc.edu.cn',
      subject: 'New blog comment — ' + (env.SITE_NAME || "Jiawei Guan's Blog"),
      // Plain text treats visitor HTML as text, never as executable email markup.
      text: [
        'Author: ' + (comment.nick || 'Guest'),
        'Status: ' + (comment.status || 'saved'),
        'Comment ID: ' + (comment.objectId || ''),
        '', String(comment.comment || ''), '',
        'Read the discussion: ' + post.href
      ].join('\n')
    });
  };
}

module.exports = { createOwnerNotifier };

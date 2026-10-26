const Waline = require('@waline/vercel');
const nodemailer = require('nodemailer');
const { createOwnerNotifier } = require('./notify-owner.cjs');

// Public defaults only. SMTP/database credentials come from the host's secrets.
process.env.AUTHOR_EMAIL = process.env.AUTHOR_EMAIL || 'guanjw@ruc.edu.cn';
process.env.SITE_NAME = process.env.SITE_NAME || "Jiawei Guan's Blog";
process.env.SITE_URL = process.env.SITE_URL || 'https://guan-jw.github.io';
// Our postSave hook includes owner comments too; avoid duplicate owner mail.
process.env.DISABLE_AUTHOR_NOTIFY = 'true';
const notifyOwner = createOwnerNotifier(process.env, nodemailer.createTransport);

module.exports = Waline({
  secureDomains: ['guan-jw.github.io'],
  async postSave(comment) {
    // Await sending before the serverless invocation finishes. A failed email
    // must not make an already-saved comment look like a failed submission.
    try {
      await notifyOwner(comment);
    } catch (error) {
      console.error('Owner notification failed for comment %s (%s). Check SMTP configuration.',
        comment.objectId, error.code || 'MAIL_ERROR');
    }
  }
});

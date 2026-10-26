const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const { createOwnerNotifier } = require('../services/blog-comments/notify-owner.cjs');

test('preview likes toggle and comments render as text without network or storage', () => {
  const events = {};
  const count = { textContent: '0' };
  const like = {
    disabled: true, pressed: 'false',
    getAttribute() { return this.pressed; },
    setAttribute(key, value) { this.pressed = value; },
    querySelector() { return count; },
    addEventListener(name, fn) { events.like = fn; }
  };
  const name = { value: 'Reader', disabled: true };
  const comment = { value: '<img src=x onerror=alert(1)>', disabled: true };
  const fields = [name, comment];
  fields.name = name; fields.comment = comment;
  const form = { elements: fields, addEventListener(name, fn) { events.submit = fn; } };
  const entries = [];
  const section = {
    dataset: { mode: 'preview', server: 'https://live.invalid' },
    querySelector(selector) {
      return { '.blog-like': like, form, '.blog-preview-comments': { appendChild: e => entries.push(e) } }[selector];
    }
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../assets/js/blog-interactions.js'), 'utf8'), {
    document: {
      querySelector: () => section,
      createElement: tag => ({ tag, children: [], appendChild(e) { this.children.push(e); } })
    }
    // No fetch, localStorage, window, or dynamic-import callback available.
  });
  events.like(); assert.equal(count.textContent, '1');
  events.like(); assert.equal(count.textContent, '0');
  assert.equal(like.pressed, 'false');
  events.submit({ preventDefault() {} });
  assert.equal(entries.length, 1);
  assert.equal(entries[0].children[1].textContent, '<img src=x onerror=alert(1)>');
  assert.equal(comment.value, '');
  events.submit({ preventDefault() {} });
  assert.equal(entries.length, 1);
});

const env = { SMTP_HOST: 'smtp.example.test', SMTP_USER: 'sender@example.test',
  SMTP_PASS: 'test-only', SMTP_PORT: '587', SMTP_SECURE: 'false' };

test('every saved comment, including owner replies, sends to the fixed owner address', async () => {
  const messages = [];
  const notify = createOwnerNotifier(env, options => {
    assert.equal(options.secure, false);
    assert.equal(options.port, 587);
    return { sendMail: async message => { messages.push(message); } };
  });
  for (const mail of ['visitor@example.test', 'guanjw@ruc.edu.cn']) {
    await notify({ mail, nick: 'Reader', url: '/blog/post/', objectId: '123',
      comment: '<script>test</script>', status: 'waiting' });
  }
  assert.equal(messages.length, 2);
  for (const message of messages) {
    assert.equal(message.to, 'guanjw@ruc.edu.cn');
    assert.equal(message.html, undefined);
    assert.match(message.text, /https:\/\/guan-jw.github.io\/blog\/post\/#123/);
  }
});

test('untrusted comment paths cannot change the notification link origin', async () => {
  const notify = createOwnerNotifier(env, () => ({ sendMail: async message => {
    assert.match(message.text, /Read the discussion: https:\/\/guan-jw.github.io\/#123/);
    assert.doesNotMatch(message.text, /evil.example/);
  } }));
  await notify({ url: 'https://evil.example/', objectId: '123' });
});

test('missing SMTP is reported rather than pretending that email was sent', async () => {
  const notify = createOwnerNotifier({}, () => assert.fail('Must not create a transport'));
  await assert.rejects(notify({}), { code: 'SMTP_NOT_CONFIGURED' });
});

test('SMTP errors propagate to the server hook for logging', async () => {
  const notify = createOwnerNotifier(env, () => ({ sendMail: async () => {
    throw Object.assign(new Error('Unavailable'), { code: 'ECONNECTION' });
  } }));
  await assert.rejects(notify({}), { code: 'ECONNECTION' });
});

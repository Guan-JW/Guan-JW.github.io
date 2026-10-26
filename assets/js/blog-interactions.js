(function () {
  'use strict';
  var section = document.querySelector('.blog-interactions');
  if (!section) return;

  // This branch never loads Waline, reads live data, or sends email.
  if (section.dataset.mode === 'preview') {
    var like = section.querySelector('.blog-like');
    var form = section.querySelector('form');
    like.disabled = false;
    Array.prototype.forEach.call(form.elements, function (field) { field.disabled = false; });
    like.addEventListener('click', function () {
      var pressed = like.getAttribute('aria-pressed') !== 'true';
      like.setAttribute('aria-pressed', String(pressed));
      like.querySelector('[data-like-count]').textContent = pressed ? '1' : '0';
    });
    form.addEventListener('submit', function (event) {
      event.preventDefault();
      var name = form.elements.name.value.trim();
      var comment = form.elements.comment.value.trim();
      if (!name || !comment) return;
      var entry = document.createElement('article');
      var author = document.createElement('strong');
      var body = document.createElement('p');
      author.textContent = name;
      body.textContent = comment;
      entry.appendChild(author);
      entry.appendChild(body);
      section.querySelector('.blog-preview-comments').appendChild(entry);
      form.elements.comment.value = '';
    });
    return;
  }

  var button = section.querySelector('[data-load-comments]');
  var status = section.querySelector('[data-comments-status]');
  var stylesheet;
  button.addEventListener('click', async function () {
    button.disabled = true;
    status.textContent = 'Loading discussion…';
    try {
      var server = new URL(section.dataset.server);
      if (server.protocol !== 'https:' || server.username || server.password) {
        throw new Error('An HTTPS comment service is required.');
      }
      if (!stylesheet) {
        stylesheet = document.createElement('link');
        stylesheet.rel = 'stylesheet';
        stylesheet.href = 'https://cdn.jsdelivr.net/npm/@waline/client@3.1.3/dist/waline.css';
        document.head.appendChild(stylesheet);
      }
      var waline = await import('https://cdn.jsdelivr.net/npm/@waline/client@3.1.3/dist/waline.js');
      waline.init({
        el: '#blog-waline',
        serverURL: server.href.replace(/\/$/, ''),
        path: section.dataset.path,
        lang: 'en-US',
        login: 'enable',
        meta: ['nick', 'mail'],
        requiredMeta: ['nick'],
        locale: { reactionTitle: 'Enjoyed this post?', reaction0: 'Like' },
        reaction: [new URL(section.dataset.heart, window.location.href).href],
        emoji: false,
        imageUploader: false,
        search: false,
        pageview: false,
        wordLimit: 2000,
        pageSize: 10
      });
      button.hidden = true;
      status.textContent = '';
    } catch (error) {
      status.textContent = 'Comments could not load. Please try again.';
      button.textContent = 'Retry loading comments';
      button.disabled = false;
    }
  });
}());

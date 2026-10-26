(function () {
  'use strict';
  var content = document.querySelector('.blog-post .page__content');
  if (!content) return;
  var toc = content.querySelector('.blog-toc');
  if (!toc) return;
  var list = toc.querySelector('.blog-toc__items');
  var headings = content.querySelectorAll('h2[id], h3[id]');
  Array.prototype.forEach.call(headings, function (heading) {
    var item = document.createElement('li');
    var link = document.createElement('a');
    link.href = '#' + encodeURIComponent(heading.id);
    link.textContent = heading.textContent;
    if (heading.tagName === 'H3') item.className = 'blog-toc__subheading';
    item.appendChild(link);
    list.appendChild(item);
  });
  toc.hidden = headings.length === 0;
}());

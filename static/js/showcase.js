// Renders the WL@UW showcase (posters / demos) from static/data/showcase.yaml
// into #showcase. Humans edit showcase.yaml (data) only — not this file.
(function () {
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }

  function tags(list) {
    if (!list || !list.length) return '';
    return '<div class="tags mb-2">' + list.map(function (t) {
      return '<span class="tag is-info is-light">' + esc(t) + '</span>';
    }).join('') + '</div>';
  }

  function card(item) {
    var href = item.pdf ? './static/pdfs/' + esc(item.pdf) : (item.url || '#');
    var image = item.image
      ? '<figure class="image mb-4"><img src="./static/images/posters/' + esc(item.image) + '" ' +
        'alt="' + esc(item.title) + '" style="border:1px solid #e5e5e5;border-radius:6px;"></figure>'
      : '';
    var authors = item.authors
      ? '<p class="has-text-grey is-size-6 mb-2">' + esc(item.authors) + '</p>' : '';
    var open = (item.pdf || item.url)
      ? '<span class="icon-text has-text-link is-size-7"><span class="icon">' +
        '<i class="fas fa-file-pdf"></i></span><span>Open poster (PDF)</span></span>'
      : '';
    return '<div class="column is-8">' +
      '<a href="' + esc(href) + '" target="_blank" rel="noopener" class="box" style="display:block;">' +
        image +
        '<p class="title is-5 mb-1">' + esc(item.title) + '</p>' +
        authors + tags(item.tags) + open +
      '</a></div>';
  }

  function render(mount, items) {
    if (!items || !items.length) {
      // Nothing to show — hide the whole section.
      var sec = mount.closest('section');
      if (sec) sec.style.display = 'none';
      return;
    }
    mount.innerHTML = '<div class="columns is-multiline is-centered">' +
      items.map(card).join('') + '</div>';
  }

  document.addEventListener('DOMContentLoaded', function () {
    var mount = document.getElementById('showcase');
    if (!mount) return;
    fetch('./static/data/showcase.yaml')
      .then(function (r) { return r.text(); })
      .then(function (text) {
        render(mount, (window.jsyaml ? jsyaml.load(text) : []) || []);
      })
      .catch(function () {
        var sec = mount.closest('section');
        if (sec) sec.style.display = 'none';
      });
  });
})();

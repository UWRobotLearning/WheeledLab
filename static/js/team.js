// Renders the WL@UW people cards from static/data/team.yaml into #team-showcase.
// Humans edit team.yaml (data) only — not this file.
(function () {
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }

  function avatar(member) {
    var photos = Array.isArray(member.photo) ? member.photo
               : (member.photo ? [member.photo] : []);
    if (photos.length) {
      return photos.map(function (file) {
        return '<p class="image is-96x96" style="overflow:hidden;border-radius:50%;">' +
          '<img src="./static/images/headshots/' + esc(file) + '" alt="' + esc(member.name) + '" ' +
          'style="width:96px;height:96px;object-fit:cover;"></p>';
      }).join('');
    }
    var initial = (member.name || '?').trim().charAt(0).toUpperCase();
    return '<p class="image is-96x96 avatar-initials">' + esc(initial) + '</p>';
  }

  function tags(list) {
    if (!list || !list.length) return '';
    return '<div class="tags mt-3">' + list.map(function (t) {
      return '<span class="tag is-info is-light">' + esc(t) + '</span>';
    }).join('') + '</div>';
  }

  function links(list) {
    if (!list || !list.length) return '';
    return '<p class="project-links is-size-7 mt-2">' + list.map(function (l) {
      var icon = l.icon ? '<span class="icon"><i class="' + esc(l.icon) + '"></i></span>' : '';
      return '<a href="' + esc(l.url) + '"><span class="icon-text">' + icon +
        '<span>' + esc(l.label || 'Link') + '</span></span></a>';
    }).join('&nbsp;&nbsp;') + '</p>';
  }

  function card(member) {
    var role = member.role
      ? '<p class="has-text-grey is-size-6 mb-2">' + esc(member.role) + '</p>' : '';
    var bio = member.bio
      ? '<p class="is-size-6">' + esc(member.bio) + '</p>'
      : '';
    return '<div class="box"><article class="media">' +
      '<figure class="media-left">' + avatar(member) + '</figure>' +
      '<div class="media-content">' +
        '<p class="title is-5 mb-1">' + esc(member.name) + '</p>' +
        role + bio + tags(member.tags) + links(member.links) +
      '</div></article></div>';
  }

  // Each roster: a mount element id + the YAML file that fills it.
  var ROSTERS = [
    { id: 'team-showcase', file: 'team.yaml' },
    { id: 'alumni-showcase', file: 'alumni.yaml' }
  ];

  function render(mount, people) {
    mount.innerHTML = (people && people.length) ? people.map(card).join('') : '';
  }

  document.addEventListener('DOMContentLoaded', function () {
    ROSTERS.forEach(function (roster) {
      var mount = document.getElementById(roster.id);
      if (!mount) return;
      fetch('./static/data/' + roster.file)
        .then(function (r) { return r.text(); })
        .then(function (text) {
          render(mount, (window.jsyaml ? jsyaml.load(text) : []) || []);
        })
        .catch(function () {
          mount.innerHTML = '<p class="has-text-grey has-text-centered">' +
            'Could not load <code>static/data/' + roster.file + '</code>.</p>';
        });
    });
  });
})();

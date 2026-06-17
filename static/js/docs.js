// Docs sidebar: mobile collapse toggle + scroll-spy active highlighting.
(function () {
  document.addEventListener('DOMContentLoaded', function () {
    // --- Sidebar collapse toggle (all viewports, choice persisted) ---
    var toggle = document.getElementById('docs-sidebar-toggle');
    var layout = document.querySelector('.docs-layout');
    if (toggle && layout) {
      var isMobile = window.matchMedia('(max-width: 1023px)').matches;
      var stored = null;
      try { stored = localStorage.getItem('wl_sidebar_collapsed'); } catch (e) {}
      // Default: collapsed on mobile, expanded on desktop — unless the user chose otherwise.
      var collapsed = stored === null ? isMobile : stored === '1';
      layout.classList.toggle('sidebar-collapsed', collapsed);
      toggle.addEventListener('click', function () {
        var nowCollapsed = layout.classList.toggle('sidebar-collapsed');
        try { localStorage.setItem('wl_sidebar_collapsed', nowCollapsed ? '1' : '0'); } catch (e) {}
      });
    }

    // --- Reading controls: "No Distractions" (hide media) + font-size zoom ---
    var content = document.querySelector('.docs-content');

    var distract = document.getElementById('distraction-toggle');
    if (distract && layout) {
      var setMedia = function (on) {
        layout.classList.toggle('hide-media', on);
        distract.classList.toggle('is-active', on);
        var lbl = distract.querySelector('.control-label');
        var ic = distract.querySelector('i');
        if (lbl) lbl.textContent = on ? 'Show Media' : 'No Distractions';
        if (ic) ic.className = on ? 'fas fa-eye' : 'fas fa-eye-slash';
      };
      var hideMedia = false;
      try { hideMedia = localStorage.getItem('wl_hide_media') === '1'; } catch (e) {}
      setMedia(hideMedia);
      distract.addEventListener('click', function () {
        var on = !layout.classList.contains('hide-media');
        setMedia(on);
        try { localStorage.setItem('wl_hide_media', on ? '1' : '0'); } catch (e) {}
      });
    }

    var slider = document.getElementById('font-size-slider');
    if (slider && content) {
      var z = 1;
      try { var sv = localStorage.getItem('wl_reading_zoom'); if (sv) z = parseFloat(sv) || 1; } catch (e) {}
      slider.value = String(Math.round(z * 100));
      content.style.zoom = z;
      slider.addEventListener('input', function () {
        var nz = (parseInt(slider.value, 10) || 100) / 100;
        content.style.zoom = nz;
        try { localStorage.setItem('wl_reading_zoom', String(nz)); } catch (e) {}
      });
    }

    // --- Scroll-spy: highlight the sub-anchor for the section in view ---
    var subLinks = Array.prototype.slice.call(
      document.querySelectorAll('#docs-sidebar .menu-list ul a[href^="#"]')
    );
    if (!subLinks.length) return;

    var targets = subLinks
      .map(function (link) {
        var el = document.getElementById(link.getAttribute('href').slice(1));
        return el ? { link: link, el: el } : null;
      })
      .filter(Boolean);
    if (!targets.length) return;

    function setCurrent(link) {
      subLinks.forEach(function (l) { l.classList.remove('is-current'); });
      if (link) link.classList.add('is-current');
    }

    if ('IntersectionObserver' in window) {
      var visible = {};
      var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          visible[entry.target.id] = entry.isIntersecting;
        });
        // Pick the first target (in document order) currently visible.
        for (var i = 0; i < targets.length; i++) {
          if (visible[targets[i].el.id]) {
            setCurrent(targets[i].link);
            return;
          }
        }
      }, { rootMargin: '-20% 0px -70% 0px', threshold: 0 });
      targets.forEach(function (t) { observer.observe(t.el); });
    }

    // Click: set immediately for responsiveness.
    subLinks.forEach(function (link) {
      link.addEventListener('click', function () { setCurrent(link); });
    });
  });
})();

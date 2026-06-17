// Global visitor map — self-hosted with Leaflet (vendored locally) + OpenStreetMap
// tiles. No third-party widget service that can go down.
//
// Two modes:
//   * LIVE (default, zero setup): shows a world map and pins the *current* visitor's
//     location (via a free IP-geolocation lookup). Works immediately.
//   * PERSISTENT (optional): if you fill in the Supabase config below, the map also
//     loads every past visitor's dot and records each new one — a true "where people
//     have viewed from" map — plus a running total.
//
// --- Persistent-mode setup (one time, free) ---------------------------------
//   1. Create a free project at https://supabase.com .
//   2. In the SQL editor, run:
//        create table visits (
//          id bigint generated always as identity primary key,
//          lat double precision, lng double precision,
//          city text, country text,
//          created_at timestamptz default now()
//        );
//        alter table visits enable row level security;
//        create policy "anon read"   on visits for select to anon using (true);
//        create policy "anon insert" on visits for insert to anon with check (true);
//   3. Settings -> API: copy the Project URL and the anon public key into the two
//      constants below.
// ----------------------------------------------------------------------------

var SUPABASE_URL = "";        // e.g. "https://abcd1234.supabase.co"  (leave "" for live-only mode)
var SUPABASE_ANON_KEY = "";   // anon / public key
var SUPABASE_TABLE = "visits";

(function () {
  document.addEventListener('DOMContentLoaded', function () {
    var mount = document.getElementById('visitor-map');
    if (!mount) return;
    loadLeaflet(function (ok) {
      if (!ok || !window.L) {
        mount.innerHTML = note('Map could not load.');
        return;
      }
      setupMap(mount);
    });
  });

  // --- Load vendored Leaflet (css + js) on demand --------------------------
  function loadLeaflet(cb) {
    if (window.L) return cb(true);
    var css = document.createElement('link');
    css.rel = 'stylesheet';
    css.href = './static/vendor/leaflet/leaflet.css';
    document.head.appendChild(css);
    var js = document.createElement('script');
    js.src = './static/vendor/leaflet/leaflet.js';
    js.onload = function () { cb(true); };
    js.onerror = function () { cb(false); };
    document.head.appendChild(js);
  }

  // --- Build the map -------------------------------------------------------
  function setupMap(mount) {
    var mapDiv = document.createElement('div');
    mapDiv.style.width = '100%';
    mapDiv.style.height = '220px';
    mapDiv.style.borderRadius = '8px';
    mapDiv.style.overflow = 'hidden';
    var caption = document.createElement('p');
    caption.className = 'is-size-7 has-text-grey mt-2';
    mount.appendChild(mapDiv);
    mount.appendChild(caption);

    var map = L.map(mapDiv, {
      attributionControl: true,
      scrollWheelZoom: false,
      worldCopyJump: true
    }).setView([20, 0], 1);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 7,
      attribution: '&copy; OpenStreetMap'
    }).addTo(map);

    var persistent = supabaseReady();

    // Plot all historical visitors (persistent mode only).
    if (persistent) {
      fetchPoints(function (points) {
        points.forEach(function (p) {
          if (p.lat != null && p.lng != null) dot(map, p.lat, p.lng);
        });
        if (points.length) caption.textContent = points.length.toLocaleString() + ' views and counting';
      });
    }

    // Locate and pin the current visitor.
    geolocate(function (loc) {
      if (!loc) {
        if (!caption.textContent) caption.textContent = 'Visitor location unavailable';
        return;
      }
      L.circleMarker([loc.lat, loc.lng], {
        radius: 6, color: '#ffffff', weight: 2,
        fillColor: '#e63100', fillOpacity: 1
      }).addTo(map).bindPopup('You are viewing from ' +
        (loc.city ? loc.city + ', ' : '') + (loc.country || ''));
      map.setView([loc.lat, loc.lng], 3);

      if (persistent) {
        if (shouldLog()) insertPoint(loc);
      } else if (!caption.textContent) {
        caption.textContent = 'Showing your location' + (loc.city ? ' — ' + loc.city : '');
      }
    });
  }

  function dot(map, lat, lng) {
    L.circleMarker([lat, lng], {
      radius: 3, stroke: false, fillColor: '#6366f1', fillOpacity: 0.7
    }).addTo(map);
  }

  // --- Visitor IP geolocation (free, no key) -------------------------------
  function geolocate(cb) {
    fetch('https://ipwho.is/')
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (d && d.success !== false && d.latitude != null) {
          cb({ lat: d.latitude, lng: d.longitude, city: d.city, country: d.country });
        } else { throw new Error('no geo'); }
      })
      .catch(function () {
        fetch('https://ipapi.co/json/')
          .then(function (r) { return r.json(); })
          .then(function (d) {
            if (d && d.latitude != null) {
              cb({ lat: d.latitude, lng: d.longitude, city: d.city, country: d.country_name });
            } else { cb(null); }
          })
          .catch(function () { cb(null); });
      });
  }

  // --- Supabase (optional persistence) -------------------------------------
  function supabaseReady() {
    return SUPABASE_URL.indexOf('http') === 0 && SUPABASE_ANON_KEY.length > 0;
  }
  function sbHeaders(extra) {
    var h = {
      'apikey': SUPABASE_ANON_KEY,
      'Authorization': 'Bearer ' + SUPABASE_ANON_KEY,
      'Content-Type': 'application/json'
    };
    if (extra) for (var k in extra) h[k] = extra[k];
    return h;
  }
  function fetchPoints(cb) {
    fetch(SUPABASE_URL + '/rest/v1/' + SUPABASE_TABLE + '?select=lat,lng', { headers: sbHeaders() })
      .then(function (r) { return r.json(); })
      .then(function (d) { cb(Array.isArray(d) ? d : []); })
      .catch(function () { cb([]); });
  }
  function insertPoint(loc) {
    fetch(SUPABASE_URL + '/rest/v1/' + SUPABASE_TABLE, {
      method: 'POST',
      headers: sbHeaders({ 'Prefer': 'return=minimal' }),
      body: JSON.stringify({ lat: loc.lat, lng: loc.lng, city: loc.city, country: loc.country })
    }).catch(function () {});
  }
  // Avoid double-counting: log at most once per 12h per browser.
  function shouldLog() {
    try {
      var last = +localStorage.getItem('wl_visit_logged') || 0;
      if (Date.now() - last < 12 * 3600 * 1000) return false;
      localStorage.setItem('wl_visit_logged', String(Date.now()));
    } catch (e) { /* private mode: just log */ }
    return true;
  }

  function note(msg) {
    return '<p class="is-size-7 has-text-grey">' + msg + '</p>';
  }
})();

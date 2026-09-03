(() => {
  const STORAGE_KEY = "tracking-consent";
  const registry = [];
  let cached = null;
  let banner = null;

  function read() {
    if (cached) return cached;
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      cached = raw ? JSON.parse(raw) : null;
    } catch {
      cached = null;
    }
    return cached;
  }

  function write(choice) {
    cached = choice;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(choice));
    } catch {}
  }

  function allowed(category) {
    const choice = read();
    if (!choice) return false;
    if (typeof choice[category] === "boolean") return choice[category];
    return choice.analytics === true;
  }

  function flush() {
    for (const entry of registry) {
      if (!entry.called && allowed(entry.category)) {
        entry.called = true;
        entry.load();
      }
    }
  }

  function hideBanner() {
    banner?.remove();
    banner = null;
  }

  function decide(granted) {
    const choice = { analytics: granted };
    for (const entry of registry) choice[entry.category] = granted;
    write(choice);
    hideBanner();
    flush();
  }

  function showBanner() {
    if (banner || read()) return;

    banner = document.createElement("div");
    banner.id = "consent";
    banner.tabIndex = -1;
    banner.setAttribute("role", "dialog");
    banner.setAttribute("aria-label", "cookie consent");
    banner.innerHTML = `
      <p>this site uses analytics cookies. accept to allow tracking, or decline to continue without it.</p>
      <div>
        <button type="button" data-consent="decline">decline</button>
        <button type="button" data-consent="accept">accept</button>
      </div>
    `;
    banner.addEventListener("click", (event) => {
      const action = event.target.closest("[data-consent]")?.dataset.consent;
      if (action === "accept") decide(true);
      if (action === "decline") decide(false);
    });
    document.body.appendChild(banner);
    banner.focus();
  }

  function gate(category, load) {
    registry.push({ category, load, called: false });
    flush();
  }

  function mount() {
    if (read()) flush();
    else showBanner();
  }

  window.trackingConsent = {
    gate,
    getConsent: read,
    accept: () => decide(true),
    decline: () => decide(false),
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mount, { once: true });
  } else {
    mount();
  }
})();

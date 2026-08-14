function loadScript(src) {
  return new Promise((resolve, reject) => {
    const el = document.createElement("script");
    el.src = src;
    el.onload = resolve;
    el.onerror = reject;
    document.head.appendChild(el);
  });
}

function probeWorker() {
  return new Promise((resolve) => {
    let worker;
    try {
      worker = new Worker("/js/render-worker.js");
    } catch {
      resolve(null);
      return;
    }
    const timer = setTimeout(() => settle(false), 3000);
    function settle(ok) {
      clearTimeout(timer);
      worker.onmessage = null;
      worker.onerror = null;
      if (ok) {
        resolve(worker);
      } else {
        worker.terminate();
        resolve(null);
      }
    }
    worker.onerror = () => settle(false);
    worker.onmessage = (event) => settle(event.data && event.data.ok === true);
    worker.postMessage({ type: "probe" });
  });
}

const orbitEls = document.getElementById("bh-orbits").children;
const labelEl = document.getElementById("bh-label");
const orbitsRoot = document.getElementById("bh-orbits");

let lastHud = null;
let cursorX = 0;
let cursorY = 0;
let hoverActive = false;
let sendHover = () => {};
let pinned = null;
const lastShown = [];

function smoothstep(edge0, edge1, x) {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function repel(x, y) {
  const minRes = Math.min(window.innerWidth, window.innerHeight);
  const dx = x - cursorX;
  const dy = y - cursorY;
  const dist = Math.hypot(dx, dy) || 0.001;
  const warp = 1 - smoothstep(0, 0.25 * minRes, dist);
  const push = warp * 2 * minRes * 0.12;
  return { x: x + (dx / dist) * push, y: y + (dy / dist) * push };
}

function applyHud(data) {
  lastHud = data;
  paintHud();
}

function paintHud() {
  const data = lastHud;
  if (!data) return;
  const a = data.a || 0;
  if (a <= 0) {
    labelEl.style.opacity = "0";
    orbitsRoot.style.opacity = "0";
    return;
  }
  const label = repel(data.x, data.y);
  labelEl.style.opacity = String(a);
  labelEl.style.transform =
    "translate3d(" +
    label.x +
    "px," +
    label.y +
    "px,0) translate(-50%,-100%) scale(" +
    data.s +
    ")";
  orbitsRoot.style.opacity = "1";
  const orbits = data.orbits || [];
  let next = -1;
  if (pinned) {
    const hold = Math.hypot(pinned.x - cursorX, pinned.y - cursorY);
    if (hold < 160) next = pinned.i;
    else pinned = null;
  }
  if (next < 0) {
    let best = 90;
    for (let i = 0; i < orbits.length; i++) {
      const o = orbits[i];
      if (!o || !o.a) continue;
      const shown = lastShown[i] || o;
      const d = Math.min(
        Math.hypot(shown.x - cursorX, shown.y - cursorY),
        Math.hypot(o.x - cursorX, o.y - cursorY)
      );
      if (d < best) {
        best = d;
        next = i;
      }
    }
  }
  if (next >= 0) {
    if (!pinned || pinned.i !== next) {
      const shown = lastShown[next] || orbits[next];
      pinned = { i: next, x: shown.x, y: shown.y };
    }
  }
  if ((next >= 0) !== hoverActive) {
    hoverActive = next >= 0;
    sendHover(hoverActive);
  }
  for (let i = 0; i < orbitEls.length; i++) {
    const item = orbitEls[i];
    const o = orbits[i];
    if (!o || !o.a) {
      item.style.opacity = "0";
      continue;
    }
    const pos = pinned && i === pinned.i ? pinned : repel(o.x, o.y);
    lastShown[i] = pos;
    item.style.opacity = String(o.a);
    item.style.zIndex = String(o.z);
    item.style.transform =
      "translate3d(" +
      pos.x +
      "px," +
      pos.y +
      "px,0) translate(-50%,-50%) scale(" +
      o.s +
      ")";
  }
}

function init() {
  const canvas = document.getElementById("canvas");

  let sink = null;
  const pending = new Map();
  let pendingRadius = 0;

  function send(msg) {
    if (sink) {
      sink(msg);
    } else if (msg.type === "radius") {
      pendingRadius += msg.d;
    } else {
      pending.set(msg.type, msg);
    }
  }

  function attach(next) {
    sink = next;
    for (const msg of pending.values()) sink(msg);
    pending.clear();
    if (pendingRadius !== 0) {
      sink({ type: "radius", d: pendingRadius });
      pendingRadius = 0;
    }
  }

  async function startRenderer() {
    const offscreenReady =
      typeof Worker === "function" &&
      typeof OffscreenCanvas !== "undefined" &&
      typeof canvas.transferControlToOffscreen === "function";
    const worker = offscreenReady ? await probeWorker() : null;

    if (worker) {
      const offscreen = canvas.transferControlToOffscreen();
      worker.postMessage({ type: "init", canvas: offscreen }, [offscreen]);
      worker.addEventListener("message", (event) => {
        if (event.data && event.data.type === "hud") applyHud(event.data);
      });
      let pointerX = 0;
      let pointerY = 0;
      let pointerDirty = false;
      let radiusDelta = 0;
      let inputRaf = 0;
      function flushInput() {
        inputRaf = 0;
        if (pointerDirty) {
          worker.postMessage({ type: "pointer", x: pointerX, y: pointerY });
          pointerDirty = false;
        }
        if (radiusDelta !== 0) {
          worker.postMessage({ type: "radius", d: radiusDelta });
          radiusDelta = 0;
        }
      }
      attach((msg) => {
        if (msg.type === "pointer") {
          pointerX = msg.x;
          pointerY = msg.y;
          pointerDirty = true;
          if (!inputRaf) inputRaf = requestAnimationFrame(flushInput);
          return;
        }
        if (msg.type === "radius") {
          radiusDelta += msg.d;
          if (!inputRaf) inputRaf = requestAnimationFrame(flushInput);
          return;
        }
        worker.postMessage(msg);
      });
      return;
    }

    let renderer = null;
    try {
      await loadScript("/js/renderer.js");
      renderer = self.createRenderer(canvas, applyHud);
    } catch {
      renderer = null;
    }
    if (!renderer) {
      attach(() => {});
      return;
    }
    attach((msg) => {
      switch (msg.type) {
        case "size":
          renderer.setSize(msg.w, msg.h);
          break;
        case "pointer":
          renderer.setPointer(msg.x, msg.y);
          break;
        case "radius":
          renderer.adjustRadius(msg.d);
          break;
        case "running":
          renderer.setRunning(msg.v);
          break;
        case "hidden":
          renderer.setHidden(msg.v);
          break;
        case "hover":
          renderer.setHover(msg.v);
          break;
      }
    });
  }

  if (localStorage.getItem("invert") === "1") {
    document.documentElement.classList.add("inverted");
  }

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const webglEnabled = !prefersReducedMotion;
  canvas.style.display = webglEnabled ? "" : "none";
  if (!webglEnabled) applyHud({ a: 0 });

  let lastTouchY = 0;

  new ResizeObserver(() => {
    send({ type: "size", w: canvas.clientWidth, h: canvas.clientHeight });
  }).observe(canvas);

  document.addEventListener("visibilitychange", () => {
    send({ type: "hidden", v: document.hidden });
  });

  document.addEventListener(
    "wheel",
    (e) => {
      send({ type: "radius", d: -e.deltaY * 0.00008 });
    },
    { passive: true }
  );

  document.addEventListener(
    "touchstart",
    (e) => {
      if (e.touches.length === 1) lastTouchY = e.touches[0].clientY;
    },
    { passive: true }
  );

  document.addEventListener(
    "touchmove",
    (e) => {
      if (e.touches.length === 1) {
        const delta = lastTouchY - e.touches[0].clientY;
        lastTouchY = e.touches[0].clientY;
        send({ type: "radius", d: delta * 0.0005 });
      }
    },
    { passive: true }
  );

  document.addEventListener(
    "pointermove",
    (e) => {
      cursorX = e.clientX;
      cursorY = e.clientY;
      paintHud();
      send({
        type: "pointer",
        x: e.clientX / window.innerWidth,
        y: 1 - e.clientY / window.innerHeight,
      });
    },
    { passive: true }
  );

  sendHover = (v) => send({ type: "hover", v });

  send({ type: "size", w: canvas.clientWidth, h: canvas.clientHeight });
  send({ type: "running", v: webglEnabled });
  startRenderer();
}

init();

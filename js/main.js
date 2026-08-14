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
      attach((msg) => worker.postMessage(msg));
      return;
    }

    let renderer = null;
    try {
      await loadScript("/js/renderer.js");
      renderer = self.createRenderer(canvas);
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
      }
    });
  }

  if (localStorage.getItem("invert") === "1") {
    document.documentElement.classList.add("inverted");
  }

  document.getElementById("star").addEventListener("click", () => {
    document.documentElement.classList.toggle("inverted");
    localStorage.setItem(
      "invert",
      document.documentElement.classList.contains("inverted") ? "1" : "0"
    );
  });

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const webglStored = localStorage.getItem("webgl");
  let webglEnabled = webglStored === null ? true : webglStored !== "0";
  const webglToggleEl = document.getElementById("webgl-toggle");

  function toggleWebGL() {
    if (prefersReducedMotion) return;
    webglEnabled = !webglEnabled;
    localStorage.setItem("webgl", webglEnabled ? "1" : "0");
    canvas.style.display = webglEnabled ? "" : "none";
    webglToggleEl.classList.toggle("webgl-off", !webglEnabled);
    send({ type: "running", v: webglEnabled });
  }

  canvas.style.display = webglEnabled ? "" : "none";
  webglToggleEl.classList.toggle("webgl-off", !webglEnabled);
  webglToggleEl.addEventListener("click", toggleWebGL);

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
      send({
        type: "pointer",
        x: e.clientX / window.innerWidth,
        y: 1 - e.clientY / window.innerHeight,
      });
    },
    { passive: true }
  );

  send({ type: "size", w: canvas.clientWidth, h: canvas.clientHeight });
  send({ type: "running", v: webglEnabled });
  startRenderer();
}

init();

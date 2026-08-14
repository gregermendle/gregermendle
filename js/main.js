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

const minimapEl = document.getElementById("minimap");
const minimapCtx = document.getElementById("minimap-canvas").getContext("2d");
let lastHud = null;

function paintMinimap(map) {
  if (!map || !minimapCtx) return;
  const ctx = minimapCtx;
  const size = 132;
  const bounds = map.bounds || 40;
  const to = (x, z) => [
    ((x + bounds) / (bounds * 2)) * size,
    ((z + bounds) / (bounds * 2)) * size,
  ];
  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, size, size);
  const hole = to(0, 0);
  ctx.fillStyle = "#fff";
  ctx.beginPath();
  ctx.arc(hole[0], hole[1], 3, 0, Math.PI * 2);
  ctx.fill();
  const stars = map.stars || [];
  for (let i = 0; i < stars.length; i++) {
    const p = to(stars[i][0], stars[i][1]);
    ctx.fillRect(p[0] - 0.5, p[1] - 0.5, 1.5, 1.5);
  }
  const c = to(map.camX || 0, map.camZ || 0);
  const yaw = map.yaw || 0;
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(c[0] + Math.sin(yaw) * 8, c[1] + Math.cos(yaw) * 8);
  ctx.lineTo(c[0] + Math.sin(yaw + 2.5) * 5, c[1] + Math.cos(yaw + 2.5) * 5);
  ctx.lineTo(c[0] + Math.sin(yaw - 2.5) * 5, c[1] + Math.cos(yaw - 2.5) * 5);
  ctx.closePath();
  ctx.stroke();
}

function applyHud(data) {
  lastHud = data;
  if (data.map) paintMinimap(data.map);
}

function init() {
  const canvas = document.getElementById("canvas");

  let sink = null;
  const pending = new Map();
  let pendingRadius = 0;
  let pendingThrust = 0;
  let pendingLookX = 0;
  let pendingLookY = 0;

  function send(msg) {
    if (sink) {
      sink(msg);
    } else if (msg.type === "radius") {
      pendingRadius += msg.d;
    } else if (msg.type === "thrust") {
      pendingThrust += msg.d;
    } else if (msg.type === "look") {
      pendingLookX += msg.x;
      pendingLookY += msg.y;
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
    if (pendingThrust !== 0) {
      sink({ type: "thrust", d: pendingThrust });
      pendingThrust = 0;
    }
    if (pendingLookX !== 0 || pendingLookY !== 0) {
      sink({ type: "look", x: pendingLookX, y: pendingLookY });
      pendingLookX = 0;
      pendingLookY = 0;
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
      let thrustDelta = 0;
      let lookX = 0;
      let lookY = 0;
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
        if (thrustDelta !== 0) {
          worker.postMessage({ type: "thrust", d: thrustDelta });
          thrustDelta = 0;
        }
        if (lookX !== 0 || lookY !== 0) {
          worker.postMessage({ type: "look", x: lookX, y: lookY });
          lookX = 0;
          lookY = 0;
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
        if (msg.type === "thrust") {
          thrustDelta += msg.d;
          if (!inputRaf) inputRaf = requestAnimationFrame(flushInput);
          return;
        }
        if (msg.type === "look") {
          lookX += msg.x;
          lookY += msg.y;
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
        case "look":
          renderer.look(msg.x, msg.y);
          break;
        case "thrust":
          renderer.thrust(msg.d);
          break;
        case "keys":
          renderer.setKeys(msg);
          break;
        case "nav":
          renderer.setNav(msg.x, msg.z);
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

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const webglEnabled = !prefersReducedMotion;
  canvas.style.display = webglEnabled ? "" : "none";

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
      send({ type: "thrust", d: -e.deltaY * 0.004 });
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
        send({ type: "thrust", d: delta * 0.012 });
      }
    },
    { passive: true }
  );

  const keys = { f: false, b: false, l: false, r: false, u: false, d: false };
  function sendKeys() {
    send({ type: "keys", ...keys });
  }
  document.addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();
    if (k === "w" || k === "arrowup") keys.f = true;
    else if (k === "s" || k === "arrowdown") keys.b = true;
    else if (k === "a" || k === "arrowleft") keys.l = true;
    else if (k === "d" || k === "arrowright") keys.r = true;
    else if (k === "e") keys.u = true;
    else if (k === "q") keys.d = true;
    else return;
    sendKeys();
  });
  document.addEventListener("keyup", (e) => {
    const k = e.key.toLowerCase();
    if (k === "w" || k === "arrowup") keys.f = false;
    else if (k === "s" || k === "arrowdown") keys.b = false;
    else if (k === "a" || k === "arrowleft") keys.l = false;
    else if (k === "d" || k === "arrowright") keys.r = false;
    else if (k === "e") keys.u = false;
    else if (k === "q") keys.d = false;
    else return;
    sendKeys();
  });

  let lastLookX = null;
  let lastLookY = null;

  function mapToWorld(clientX, clientY) {
    const rect = minimapEl.getBoundingClientRect();
    const bounds = (lastHud && lastHud.map && lastHud.map.bounds) || 40;
    const x = ((clientX - rect.left) / rect.width) * bounds * 2 - bounds;
    const z = ((clientY - rect.top) / rect.height) * bounds * 2 - bounds;
    return { x, z };
  }

  minimapEl.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    minimapEl.setPointerCapture(e.pointerId);
    const w = mapToWorld(e.clientX, e.clientY);
    send({ type: "nav", x: w.x, z: w.z });
  });

  minimapEl.addEventListener("pointermove", (e) => {
    if ((e.buttons & 1) === 0) return;
    const w = mapToWorld(e.clientX, e.clientY);
    send({ type: "nav", x: w.x, z: w.z });
  });

  document.addEventListener(
    "pointermove",
    (e) => {
      if (!e.target.closest(".minimap") && lastLookX !== null) {
        send({
          type: "look",
          x: (e.clientX - lastLookX) * 0.0035,
          y: (e.clientY - lastLookY) * 0.0035,
        });
      }
      lastLookX = e.clientX;
      lastLookY = e.clientY;
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

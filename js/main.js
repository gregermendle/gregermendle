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

function showFps(n) {
  const el = document.getElementById("fps");
  if (el) el.textContent = String(n);
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
      worker.onmessage = (event) => {
        if (event.data && event.data.type === "fps") showFps(event.data.v);
      };
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
      renderer = self.createRenderer(canvas, (data) => {
        if (data && data.type === "fps") showFps(data.v);
      });
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

  const keys = { f: false, b: false, l: false, r: false, u: false, d: false, boost: false };
  function sendKeys() {
    send({ type: "keys", ...keys });
  }
  function applyKey(k, down) {
    if (k === "w" || k === "arrowup") keys.f = down;
    else if (k === "s" || k === "arrowdown") keys.b = down;
    else if (k === "a" || k === "arrowleft") keys.l = down;
    else if (k === "d" || k === "arrowright") keys.r = down;
    else if (k === "e") keys.u = down;
    else if (k === "q") keys.d = down;
    else if (k === "shift") keys.boost = down;
    else return false;
    return true;
  }
  document.addEventListener("keydown", (e) => {
    if (e.repeat) return;
    if (applyKey(e.key.toLowerCase(), true)) sendKeys();
  });
  document.addEventListener("keyup", (e) => {
    if (applyKey(e.key.toLowerCase(), false)) sendKeys();
  });
  window.addEventListener("blur", () => {
    keys.f = keys.b = keys.l = keys.r = keys.u = keys.d = keys.boost = false;
    sendKeys();
  });

  let lastLookX = null;
  let lastLookY = null;
  const lookScale = 0.0035;

  function pointerLocked() {
    return Boolean(document.pointerLockElement);
  }

  function lockPointer() {
    const el = document.documentElement;
    if (!el.requestPointerLock || pointerLocked()) return;
    const req = el.requestPointerLock({ unadjustedMovement: true });
    if (req && typeof req.catch === "function") {
      req.catch(() => el.requestPointerLock());
    }
  }

  document.addEventListener("click", () => {
    lockPointer();
  });

  document.addEventListener("pointerlockchange", () => {
    lastLookX = null;
    lastLookY = null;
  });

  document.addEventListener("mousemove", (e) => {
    if (pointerLocked()) {
      send({
        type: "look",
        x: e.movementX * lookScale,
        y: e.movementY * lookScale,
      });
      send({ type: "pointer", x: 0.5, y: 0.5 });
      return;
    }
    if (lastLookX !== null) {
      send({
        type: "look",
        x: e.movementX * lookScale,
        y: e.movementY * lookScale,
      });
    }
    lastLookX = e.clientX;
    lastLookY = e.clientY;
    send({
      type: "pointer",
      x: e.clientX / window.innerWidth,
      y: 1 - e.clientY / window.innerHeight,
    });
  });

  send({ type: "size", w: canvas.clientWidth, h: canvas.clientHeight });
  send({ type: "running", v: webglEnabled });
  startRenderer();
}

init();

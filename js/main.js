function init() {
  const canvas = document.getElementById("canvas");
  canvas.style.display = "";
  canvas.style.pointerEvents = "auto";
  canvas.style.touchAction = "none";

  let renderer = null;
  const pending = [];

  function send(msg) {
    if (renderer) {
      if (msg.type === "size") renderer.setSize(msg.w, msg.h);
      else if (msg.type === "running") renderer.setRunning(msg.v);
      else if (msg.type === "hidden") renderer.setHidden(msg.v);
      else if (msg.type === "steps") renderer.setSteps(msg.v);
      else if (msg.type === "systems") renderer.setSystems(msg.list);
      else if (msg.type === "impulse") renderer.addImpulse(msg);
      else if (msg.type === "front") renderer.addFront(msg);
      else if (msg.type === "warm") renderer.addWarm(msg);
      return;
    }
    pending.push(msg);
  }

  const script = document.createElement("script");
  script.src = "js/renderer.js?v=252";
  script.onload = () => {
    const r = self.createRenderer(canvas);
    if (!r) {
      console.error("renderer init failed");
      return;
    }
    renderer = r;
    const fpsValue = document.getElementById("fps-value");
    r.setOnFps((fps) => {
      if (fpsValue) fpsValue.textContent = String(Math.round(fps));
    });
    for (const msg of pending) send(msg);
    pending.length = 0;
  };
  script.onerror = () => console.error("failed to load renderer.js");
  document.head.appendChild(script);

  new ResizeObserver(() => {
    send({ type: "size", w: canvas.clientWidth, h: canvas.clientHeight });
  }).observe(canvas);

  document.addEventListener("visibilitychange", () => {
    send({ type: "hidden", v: document.hidden });
  });

  function toScreen(e) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) / Math.max(rect.width, 1),
      y: 1 - (e.clientY - rect.top) / Math.max(rect.height, 1),
    };
  }

  let down = false;
  let last = null;
  let moved = false;
  let holdTimer = 0;
  let pen = 1;
  let steps = 40;
  const stepMin = 16;
  const stepMax = 96;
  const stepNudge = 8;
  const marchValue = document.getElementById("march-value");
  const marks = document.getElementById("marks");
  const cyclones = [];
  let mode = null;
  let markId = 0;

  function setMode(next) {
    mode = next;
    document.getElementById("mode-cyclone")?.classList.toggle("on", mode === "C");
    canvas.style.cursor = mode ? "crosshair" : "";
  }

  function setSteps(next) {
    steps = Math.max(stepMin, Math.min(stepMax, next));
    if (marchValue) marchValue.textContent = String(steps);
    send({ type: "steps", v: steps });
  }

  function seedMoisture(p, strength) {
    send({
      type: "impulse",
      sx: p.x,
      sy: p.y,
      radius: 0.055 * pen,
      heat: 0.04 * strength,
      moist: 0.28 * strength,
    });
  }

  function syncCyclones() {
    send({
      type: "systems",
      list: cyclones.map((c) => ({
        sx: c.x,
        sy: c.y,
        radius: 0.28 * pen,
        spin: 2.0,
        born: c.born,
      })),
    });
  }

  function nearestCyclone(p, max) {
    let best = null;
    let bestD = max;
    for (const c of cyclones) {
      const d = Math.hypot(c.x - p.x, c.y - p.y);
      if (d < bestD) {
        bestD = d;
        best = c;
      }
    }
    return best;
  }

  function removeCyclone(c) {
    const i = cyclones.indexOf(c);
    if (i >= 0) cyclones.splice(i, 1);
    c.el.remove();
    syncCyclones();
  }

  function placeCyclone(p) {
    const hit = nearestCyclone(p, 0.05);
    if (hit) {
      hit.born = performance.now();
      send({
        type: "impulse",
        sx: hit.x,
        sy: hit.y,
        radius: 0.16 * pen,
        heat: 0.05,
        moist: 0.18,
        spin: 1.0,
        converge: 0.2,
      });
      syncCyclones();
      return;
    }
    if (cyclones.length >= 4) removeCyclone(cyclones[0]);
    const el = document.createElement("div");
    el.className = "mark";
    el.textContent = "c";
    el.style.cssText = `left:${p.x * 100}%;top:${(1 - p.y) * 100}%`;
    marks.appendChild(el);
    cyclones.push({ id: ++markId, x: p.x, y: p.y, el, born: performance.now() });
    send({
      type: "impulse",
      sx: p.x,
      sy: p.y,
      radius: 0.16 * pen,
      heat: 0.05,
      moist: 0.18,
      spin: 1.0,
      converge: 0.2,
    });
    send({
      type: "warm",
      sx: p.x,
      sy: p.y,
      radius: 0.26 * pen,
      amount: 0.6,
    });
    syncCyclones();
  }

  function onDown(e) {
    if (e.target.closest("#legend")) return;
    if (e.button === 2) return;
    if (mode === "C") {
      placeCyclone(toScreen(e));
      return;
    }
    down = true;
    moved = false;
    last = toScreen(e);
    holdTimer = window.setInterval(() => {
      if (down && !moved && last) seedMoisture(last, 0.45);
    }, 90);
  }

  function onMove(e) {
    if (!down || !last) return;
    const p = toScreen(e);
    const dx = p.x - last.x;
    const dy = p.y - last.y;
    if (dx * dx + dy * dy < 1.6e-6) return;
    moved = true;
    send({
      type: "front",
      ax: last.x,
      ay: last.y,
      bx: p.x,
      by: p.y,
      width: 0.032 * pen,
      cold: 0.05,
      moist: 0.09,
      along: 0.24,
      converge: 0.18,
      spin: 0.1,
    });
    last = p;
  }

  function onUp() {
    if (!down) return;
    if (!moved && last) seedMoisture(last, 1);
    down = false;
    last = null;
    window.clearInterval(holdTimer);
  }

  function onContext(e) {
    if (e.target.closest("#legend")) return;
    e.preventDefault();
    const hit = nearestCyclone(toScreen(e), 0.05);
    if (hit) removeCyclone(hit);
  }

  window.addEventListener("keydown", (e) => {
    if (e.repeat) return;
    const key = e.key.toLowerCase();
    if (key === "c") setMode(mode === "C" ? null : "C");
    if (e.key === "Escape") setMode(null);
    if (e.key === "1") pen = Math.max(0.4, pen - 0.2);
    if (e.key === "2") pen = Math.min(2.8, pen + 0.2);
    if (e.key === "[") setSteps(steps - stepNudge);
    if (e.key === "]") setSteps(steps + stepNudge);
  });

  document.getElementById("march-down")?.addEventListener("click", () => {
    setSteps(steps - stepNudge);
  });
  document.getElementById("march-up")?.addEventListener("click", () => {
    setSteps(steps + stepNudge);
  });

  window.addEventListener("pointerdown", onDown);
  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp);
  window.addEventListener("pointercancel", onUp);
  window.addEventListener("contextmenu", onContext);

  send({ type: "size", w: canvas.clientWidth, h: canvas.clientHeight });
  send({ type: "running", v: true });
}

init();

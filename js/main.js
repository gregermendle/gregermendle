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
      else if (msg.type === "impulse") renderer.addImpulse(msg);
      else if (msg.type === "view") renderer.setView(msg.v);
      return;
    }
    pending.push(msg);
  }

  const script = document.createElement("script");
  script.src = "js/renderer.js?v=96";
  script.onload = () => {
    const r = self.createRenderer(canvas);
    if (!r) {
      console.error("renderer init failed");
      return;
    }
    renderer = r;
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
  let spinning = false;
  let holdTimer = 0;

  function seed(p, strength) {
    send({
      type: "impulse",
      sx: p.x,
      sy: p.y,
      radius: 0.045,
      heat: 0.18 * strength,
      moist: 0.34 * strength,
      spin: 0.16 * strength,
      converge: 0.3 * strength,
    });
  }

  function onDown(e) {
    down = true;
    moved = false;
    spinning = e.shiftKey;
    last = toScreen(e);
    if (spinning) {
      send({ type: "impulse", sx: last.x, sy: last.y, radius: 0.07, spin: 0.5 });
    } else {
      seed(last, 1);
      holdTimer = window.setInterval(() => {
        if (down && !moved && last) seed(last, 0.5);
      }, 90);
    }
  }

  function onMove(e) {
    if (!down || !last) return;
    const p = toScreen(e);
    const dx = p.x - last.x;
    const dy = p.y - last.y;
    if (dx * dx + dy * dy < 2e-6) return;
    moved = true;
    if (spinning) {
      send({ type: "impulse", sx: p.x, sy: p.y, radius: 0.07, spin: 0.22 });
    } else {
      send({ type: "impulse", sx: p.x, sy: p.y, dx, dy, radius: 0.03 });
    }
    last = p;
  }

  function onUp() {
    if (!down) return;
    down = false;
    spinning = false;
    last = null;
    window.clearInterval(holdTimer);
  }

  self.setCloudView = (mode) => send({ type: "view", v: mode });
  self.cloudStats = () => (renderer ? renderer.stats() : null);

  window.addEventListener("pointerdown", onDown);
  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp);
  window.addEventListener("pointercancel", onUp);

  send({ type: "size", w: canvas.clientWidth, h: canvas.clientHeight });
  send({ type: "running", v: true });
}

init();

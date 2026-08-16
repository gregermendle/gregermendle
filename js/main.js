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
      else if (msg.type === "front") renderer.addFront(msg);
      return;
    }
    pending.push(msg);
  }

  const script = document.createElement("script");
  script.src = "js/renderer.js?v=214";
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
  let cyclone = false;
  let holdTimer = 0;
  let pen = 1;

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

  function seedCyclone(p, strength) {
    send({
      type: "impulse",
      sx: p.x,
      sy: p.y,
      radius: 0.08 * pen,
      heat: 0.06 * strength,
      moist: 0.22 * strength,
      spin: 0.42 * strength,
      converge: 0.28 * strength,
    });
  }

  function onDown(e) {
    down = true;
    moved = false;
    cyclone = e.shiftKey;
    last = toScreen(e);
    holdTimer = window.setInterval(() => {
      if (down && !moved && last) {
        if (cyclone) seedCyclone(last, 0.45);
        else seedMoisture(last, 0.45);
      }
    }, 90);
  }

  function onMove(e) {
    if (!down || !last) return;
    const p = toScreen(e);
    const dx = p.x - last.x;
    const dy = p.y - last.y;
    if (dx * dx + dy * dy < 1.6e-6) return;
    moved = true;
    if (cyclone) {
      send({
        type: "impulse",
        sx: p.x,
        sy: p.y,
        radius: 0.07 * pen,
        spin: 0.2,
        converge: 0.1,
        moist: 0.04,
      });
    } else {
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
    }
    last = p;
  }

  function onUp() {
    if (!down) return;
    if (!moved && last) {
      if (cyclone) seedCyclone(last, 1);
      else seedMoisture(last, 1);
    }
    down = false;
    cyclone = false;
    last = null;
    window.clearInterval(holdTimer);
  }

  window.addEventListener("keydown", (e) => {
    if (e.key === "1") pen = Math.max(0.4, pen - 0.2);
    if (e.key === "2") pen = Math.min(2.8, pen + 0.2);
  });

  window.addEventListener("pointerdown", onDown);
  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp);
  window.addEventListener("pointercancel", onUp);

  send({ type: "size", w: canvas.clientWidth, h: canvas.clientHeight });
  send({ type: "running", v: true });
}

init();

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
  script.src = "js/renderer.js?v=241";
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
  const systems = [];
  let draft = null;
  let hold = null;
  let markId = 0;

  function setHold(next) {
    hold = next;
    document.getElementById("mode-high")?.classList.toggle("on", hold === "H");
    document.getElementById("mode-low")?.classList.toggle("on", hold === "L");
    canvas.style.cursor = hold ? "crosshair" : "";
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

  function centroid(points) {
    let x = 0;
    let y = 0;
    for (const p of points) {
      x += p.x;
      y += p.y;
    }
    const n = Math.max(points.length, 1);
    return { x: x / n, y: y / n };
  }

  function systemRadius(points, kind) {
    const c = centroid(points);
    let max = 0;
    for (const p of points) {
      const d = Math.hypot(p.x - c.x, p.y - c.y);
      if (d > max) max = d;
    }
    if (kind === "L" && max < 0.05) return Math.max(0.14, 0.17 * pen);
    return Math.max(0.11 * pen, Math.min(0.34, max * 1.2 + 0.08 * pen));
  }

  function segsOf(points) {
    const segs = [];
    for (let i = 1; i < points.length; i++) {
      segs.push({
        ax: points[i - 1].x,
        ay: points[i - 1].y,
        bx: points[i].x,
        by: points[i].y,
      });
    }
    return segs;
  }

  function placeLabel(sys) {
    const c = centroid(sys.points);
    if (!sys.el) {
      sys.el = document.createElement("div");
      sys.el.className = `mark ${sys.kind === "H" ? "high" : "low"}`;
      sys.el.textContent = sys.kind;
      marks.appendChild(sys.el);
    }
    sys.el.style.cssText = `left:${c.x * 100}%;top:${(1 - c.y) * 100}%`;
  }

  function syncSystems() {
    send({
      type: "systems",
      list: systems.map((s) => {
        const c = centroid(s.points);
        return {
          kind: s.kind,
          sx: c.x,
          sy: c.y,
          radius: systemRadius(s.points, s.kind),
          spin: 1.55,
          born: s.born,
          segs: segsOf(s.points),
        };
      }),
    });
  }

  function seedWarm(p, amount, radius) {
    send({
      type: "warm",
      sx: p.x,
      sy: p.y,
      radius: radius || 0.26 * pen,
      amount,
    });
  }

  function seedSystem(p, kind, strength) {
    const low = kind === "L";
    send({
      type: "impulse",
      sx: p.x,
      sy: p.y,
      radius: (low ? 0.14 : 0.16) * pen,
      heat: (low ? 0.05 : -0.045) * strength,
      moist: (low ? 0.14 : -0.12) * strength,
      spin: (low ? 0.95 : -0.7) * strength,
      converge: (low ? 0.18 : -0.14) * strength,
    });
    if (low) seedWarm(p, 0.58 * strength, 0.28 * pen);
  }

  function strokeSystem(a, b, kind) {
    const low = kind === "L";
    send({
      type: "front",
      ax: a.x,
      ay: a.y,
      bx: b.x,
      by: b.y,
      width: 0.03 * pen,
      cold: low ? -0.02 : 0.035,
      moist: low ? 0.07 : -0.06,
      along: 0.1,
      converge: low ? 0.16 : -0.16,
      spin: low ? 0.1 : -0.1,
    });
    if (low) {
      seedWarm(
        { x: (a.x + b.x) * 0.5, y: (a.y + b.y) * 0.5 },
        0.07,
        0.12 * pen
      );
    }
  }

  function startDraft(p, kind) {
    if (systems.length >= 8) systems.shift()?.el?.remove();
    draft = { id: ++markId, kind, points: [p], el: null, born: performance.now() };
    systems.push(draft);
    placeLabel(draft);
    seedSystem(p, kind, 0.85);
    syncSystems();
  }

  function extendDraft(p) {
    if (!draft) return;
    const prev = draft.points[draft.points.length - 1];
    draft.points.push(p);
    placeLabel(draft);
    strokeSystem(prev, p, draft.kind);
    syncSystems();
  }

  function finishDraft() {
    if (!draft) return;
    placeLabel(draft);
    syncSystems();
    draft = null;
  }

  function distToSeg(p, a, b) {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len2 = dx * dx + dy * dy;
    if (len2 < 1e-10) return Math.hypot(p.x - a.x, p.y - a.y);
    let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(p.x - (a.x + dx * t), p.y - (a.y + dy * t));
  }

  function nearestSystem(p, max) {
    let best = null;
    let bestD = max;
    for (const s of systems) {
      const c = centroid(s.points);
      let d = Math.hypot(c.x - p.x, c.y - p.y);
      for (let i = 1; i < s.points.length; i++) {
        d = Math.min(d, distToSeg(p, s.points[i - 1], s.points[i]));
      }
      if (d < bestD) {
        bestD = d;
        best = s;
      }
    }
    return best;
  }

  function removeSystem(sys) {
    const i = systems.indexOf(sys);
    if (i >= 0) systems.splice(i, 1);
    if (sys.el) sys.el.remove();
    if (draft === sys) draft = null;
    syncSystems();
  }

  function onDown(e) {
    if (e.target.closest("#legend")) return;
    if (e.button === 2) return;
    down = true;
    moved = false;
    last = toScreen(e);
    if (hold) startDraft(last, hold);
    holdTimer = window.setInterval(() => {
      if (down && !moved && last && !hold) seedMoisture(last, 0.45);
    }, 90);
  }

  function onMove(e) {
    if (!down || !last) return;
    const p = toScreen(e);
    const dx = p.x - last.x;
    const dy = p.y - last.y;
    if (dx * dx + dy * dy < 1.6e-6) return;
    moved = true;
    if (draft) {
      extendDraft(p);
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
    if (!moved && last && !draft) seedMoisture(last, 1);
    finishDraft();
    down = false;
    last = null;
    window.clearInterval(holdTimer);
  }

  function onContext(e) {
    if (e.target.closest("#legend")) return;
    e.preventDefault();
    const hit = nearestSystem(toScreen(e), 0.045);
    if (hit) removeSystem(hit);
  }

  window.addEventListener("keydown", (e) => {
    if (e.repeat) return;
    const key = e.key.toLowerCase();
    if (key === "h") setHold(hold === "H" ? null : "H");
    if (key === "l") setHold(hold === "L" ? null : "L");
    if (e.key === "Escape") setHold(null);
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

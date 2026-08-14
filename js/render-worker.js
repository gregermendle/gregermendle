let renderer = null;

function canRender() {
  if (typeof requestAnimationFrame !== "function" || typeof OffscreenCanvas === "undefined") {
    return false;
  }
  let ok = false;
  try {
    const probe = new OffscreenCanvas(1, 1);
    const gl = probe.getContext("webgl2") || probe.getContext("webgl");
    ok = !!gl;
    if (gl) {
      const lose = gl.getExtension("WEBGL_lose_context");
      if (lose) lose.loseContext();
    }
  } catch {
    return false;
  }
  if (!ok) return false;
  try {
    self.importScripts("/js/renderer.js");
  } catch {
    return false;
  }
  return typeof self.createRenderer === "function";
}

self.onmessage = (event) => {
  const msg = event.data;
  switch (msg.type) {
    case "probe":
      self.postMessage({ ok: canRender() });
      break;
    case "init":
      renderer = self.createRenderer(msg.canvas, (data) => self.postMessage(data));
      break;
    case "size":
      if (renderer) renderer.setSize(msg.w, msg.h);
      break;
    case "pointer":
      if (renderer) renderer.setPointer(msg.x, msg.y);
      break;
    case "radius":
      if (renderer) renderer.adjustRadius(msg.d);
      break;
    case "look":
      if (renderer) renderer.look(msg.x, msg.y);
      break;
    case "thrust":
      if (renderer) renderer.thrust(msg.d);
      break;
    case "keys":
      if (renderer) renderer.setKeys(msg);
      break;
    case "running":
      if (renderer) renderer.setRunning(msg.v);
      break;
    case "hidden":
      if (renderer) renderer.setHidden(msg.v);
      break;
  }
};

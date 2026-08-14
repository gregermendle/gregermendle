function isMobile() {
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    navigator.userAgent
  );
}

function loadShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function initShaderProgram(gl, vsSource, fsSource) {
  const vertexShader = loadShader(gl, gl.VERTEX_SHADER, vsSource);
  const fragmentShader = loadShader(gl, gl.FRAGMENT_SHADER, fsSource);
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    return null;
  }
  return program;
}

function initBuffers(gl) {
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, 1, 1, 1, -1, -1, 1, -1]),
    gl.STATIC_DRAW
  );
  return { position: buffer };
}

function marchStepSource() {
  return `
        r = length(p);
        adaptiveStepSize = STEP_SIZE * max(1.0, r * 0.1);
        if (r >= influenceRadius) return col;
        rd = normalize(rd - p * (radius * adaptiveStepSize / (r * r * r)));
        p += rd * adaptiveStepSize;
        totalDist += adaptiveStepSize;
        if (abs(p.y) < 0.12) {
          float diskRadius = length(p.xz);
          if (diskRadius > radius) {
            float d = (diskRadius - radius) / radius;
            float innerTemp = smoothstep(2.0, 4.0, d);
            float outerTemp = smoothstep(4.0, 8.0, d);
            float gray = mix(1.0, mix(0.4, 0.1, outerTemp), innerTemp);
            vec4 diskCol = vec4(vec3(gray * exp(-d * 0.25)), 1.0) * exp(-totalDist * 0.12);
            diskCol *= 1.0 + 0.08 * sin((atan(p.z, p.x) + tRot) * 8.0);
            diskCol *= 0.75 + 0.08 * sin((atan(rd.x, rd.y) + tRot) * 8.0);
            col += diskCol;
          }
        }
        if (r < radius || totalDist > 100.0 || dot(col, col) > 100.0) return col;
`;
}

function getShaders(webgl2) {
  const mobile = isMobile();
  const maxSteps = mobile ? 84 : 128;
  const stepSize = mobile ? 0.25 : 0.15;
  const step = marchStepSource();
  const loop = webgl2
    ? `for (int i = 0; i < MAX_STEPS; i += 4) {${step}${step}${step}${step}      }`
    : `for (int i = 0; i < MAX_STEPS; i++) {${step}      }`;
  const ver = webgl2 ? "#version 300 es\n" : "";
  const fragOut = webgl2 ? "out vec4 fragColor;\n" : "";
  const writeColor = webgl2 ? "fragColor" : "gl_FragColor";
  const attr = webgl2 ? "in" : "attribute";
  const varyOut = webgl2 ? "out" : "varying";
  const varyIn = webgl2 ? "in" : "varying";
  const tex = webgl2 ? "texture" : "texture2D";

  const fs = `${ver}precision highp float;
    uniform vec2 resolution;
    uniform float time;
    uniform vec2 mouse;
    uniform float progress;
    uniform float schwarzschildRadius;
    ${fragOut}
    #define MAX_STEPS ${maxSteps}
    #define WARP_SIZE 0.25
    #define STEP_SIZE ${stepSize.toFixed(2)}

    float hash(vec2 p) {
      return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }

    vec4 rayMarch(vec3 ro, vec3 rd, vec2 uv, float radius) {
      float jitter = hash(uv) * STEP_SIZE;
      vec3 p = ro + rd * jitter;
      float r = length(p);
      vec4 col = vec4(0.0);
      float totalDist = jitter;
      float influenceRadius = radius * 50.0;
      float tRot = time * 0.3;
      float adaptiveStepSize;

      if (r > influenceRadius && dot(rd, p) > 0.0) return col;

      ${loop}
      return col;
    }

    void main() {
      float minRes = min(resolution.x, resolution.y);
      vec2 uv = (gl_FragCoord.xy - 0.5 * resolution.xy) / minRes;
      vec2 aspect = resolution.xy / minRes;
      vec2 muv = (mouse.xy - 0.5) * aspect;
      float warp = 1.0 - smoothstep(0.0, WARP_SIZE, length(uv - muv));
      vec2 outward = normalize(muv - uv + 0.001) * warp * 2.0;
      vec3 ro = vec3(-1.0, 2.0, -10.0) + vec3(muv * 2.0, 0.0) + vec3(outward, 0.0);
      vec3 rd = normalize(vec3(uv, 1.0)) - vec3(0.0, 0.2, 0.0);
      ${writeColor} = rayMarch(ro, rd, uv * aspect * 5.0, schwarzschildRadius * progress) * progress;
    }`;

  const vs = `${ver}${attr} vec4 aVertexPosition;
    void main() { gl_Position = aVertexPosition; }`;

  const blitVs = `${ver}${attr} vec4 aVertexPosition;
    ${varyOut} vec2 vTexCoord;
    void main() {
      gl_Position = aVertexPosition;
      vTexCoord = aVertexPosition.xy * 0.5 + 0.5;
    }`;

  const blitFs = `${ver}precision highp float;
    uniform sampler2D uTex;
    ${varyIn} vec2 vTexCoord;
    ${fragOut}
    void main() { ${writeColor} = ${tex}(uTex, vTexCoord); }`;

  return { vs, fs, blitVs, blitFs };
}

const LUM_SCALE = 1 / 3;
const E7 = 7 / 16;
const E3 = 3 / 16;
const E5 = 5 / 16;
const E1 = 1 / 16;
const RENDER_SCALE = 1;

let ditherWork, ditherOutput, ditherOut32, ditherSrc32;
let wasmExports = null;

function applyDitheringJS(pixels, width, height) {
  const n = width * height;

  if (!ditherWork || ditherWork.length < n) {
    ditherWork = new Float32Array(n);
    ditherOutput = new Uint8Array(n * 4);
    ditherOut32 = new Uint32Array(ditherOutput.buffer);
  }
  if (!ditherSrc32 || ditherSrc32.buffer !== pixels.buffer) {
    ditherSrc32 = new Uint32Array(pixels.buffer, pixels.byteOffset, n);
  }

  const work = ditherWork;
  const out32 = ditherOut32;
  const src32 = ditherSrc32;
  const src = pixels;
  const lastX = width - 1;
  const lastY = height - 1;

  for (let i = 0, i4 = 0; i < n; i++, i4 += 4) {
    work[i] = (src[i4] + src[i4 + 1] + src[i4 + 2]) * LUM_SCALE;
  }

  let i = 0;
  for (let y = 0; y < height; y++) {
    const notLastY = y < lastY;
    for (let x = 0; x < width; x++, i++) {
      const oldVal = work[i];
      if (oldVal < 256) {
        out32[i] = 0xff000000;
        if (oldVal !== 0) {
          if (x < lastX) work[i + 1] += oldVal * E7;
          if (notLastY) {
            if (x > 0) work[i + width - 1] += oldVal * E3;
            work[i + width] += oldVal * E5;
            if (x < lastX) work[i + width + 1] += oldVal * E1;
          }
        }
      } else {
        const err = oldVal - 255;
        out32[i] = src32[i] | 0xff000000;
        if (x < lastX) work[i + 1] += err * E7;
        if (notLastY) {
          if (x > 0) work[i + width - 1] += err * E3;
          work[i + width] += err * E5;
          if (x < lastX) work[i + width + 1] += err * E1;
        }
      }
    }
  }

  return ditherOutput;
}

function wasmDitherViews(n) {
  wasmExports.ensure(n * 12);
  const buf = wasmExports.memory.buffer;
  return {
    src: new Uint8Array(buf, 0, n * 4),
    dst: new Uint8Array(buf, n * 8, n * 4),
  };
}

function applyDithering(pixels, width, height) {
  const n = width * height;
  if (wasmExports) {
    const views = wasmDitherViews(n);
    if (pixels.buffer !== views.src.buffer || pixels.byteOffset !== 0) {
      views.src.set(pixels);
    }
    wasmExports.dither(0, n * 4, n * 8, width, height);
    return views.dst;
  }
  return applyDitheringJS(pixels, width, height);
}

async function loadWasmDither() {
  try {
    const res = await fetch("/js/dither.wasm");
    const compiled = WebAssembly.instantiateStreaming
      ? await WebAssembly.instantiateStreaming(res)
      : await WebAssembly.instantiate(await res.arrayBuffer());
    wasmExports = compiled.instance.exports;
  } catch {
    wasmExports = null;
  }
}

let framebuffer, sceneTexture, displayTexture, pixelBuffer;
let pbos = null;
let pboIndex = 0;
let pboHasPrev = false;

function createTexture(gl, width, height) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return tex;
}

function destroyPbos(gl) {
  if (!pbos) return;
  gl.deleteBuffer(pbos[0]);
  gl.deleteBuffer(pbos[1]);
  pbos = null;
  pboHasPrev = false;
}

function initFramebuffer(gl, width, height, webgl2) {
  if (framebuffer) {
    gl.deleteFramebuffer(framebuffer);
    gl.deleteTexture(sceneTexture);
    gl.deleteTexture(displayTexture);
  }
  destroyPbos(gl);

  sceneTexture = createTexture(gl, width, height);
  displayTexture = createTexture(gl, width, height);
  framebuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, sceneTexture, 0);
  pixelBuffer = new Uint8Array(width * height * 4);
  ditherSrc32 = null;
  fboWidth = width;
  fboHeight = height;

  if (webgl2) {
    const bytes = width * height * 4;
    pbos = [gl.createBuffer(), gl.createBuffer()];
    for (let i = 0; i < 2; i++) {
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbos[i]);
      gl.bufferData(gl.PIXEL_PACK_BUFFER, bytes, gl.STREAM_READ);
    }
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    pboIndex = 0;
  }
}

function easeOutBack(x) {
  const c1 = 1.70158;
  const c3 = c1 + 1;
  return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2);
}

function bindQuad(gl, loc) {
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(loc);
}

function blitDithered(gl, blitProgram, dithered, width, height, displayWidth, displayHeight) {
  gl.bindTexture(gl.TEXTURE_2D, displayTexture);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, dithered);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, displayWidth, displayHeight);
  gl.useProgram(blitProgram);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

function init() {
  const canvas = document.getElementById("canvas");
  const glAttrs = {
    alpha: false,
    depth: false,
    stencil: false,
    antialias: false,
    powerPreference: "high-performance",
    premultipliedAlpha: false,
    preserveDrawingBuffer: false,
    desynchronized: true,
    failIfMajorPerformanceCaveat: false,
  };
  const gl = canvas.getContext("webgl2", glAttrs) || canvas.getContext("webgl", glAttrs);
  if (!gl) return;

  const webgl2 = typeof WebGL2RenderingContext !== "undefined" && gl instanceof WebGL2RenderingContext;
  const shaders = getShaders(webgl2);
  const program = initShaderProgram(gl, shaders.vs, shaders.fs);
  const blitProgram = initShaderProgram(gl, shaders.blitVs, shaders.blitFs);
  const buffers = initBuffers(gl);
  loadWasmDither();

  const programInfo = {
    program,
    attribLocations: { vertexPosition: gl.getAttribLocation(program, "aVertexPosition") },
    uniformLocations: {
      resolution: gl.getUniformLocation(program, "resolution"),
      time: gl.getUniformLocation(program, "time"),
      mouse: gl.getUniformLocation(program, "mouse"),
      progress: gl.getUniformLocation(program, "progress"),
      schwarzschildRadius: gl.getUniformLocation(program, "schwarzschildRadius"),
    },
  };

  const blitAttrib = gl.getAttribLocation(blitProgram, "aVertexPosition");
  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.position);
  bindQuad(gl, programInfo.attribLocations.vertexPosition);
  if (blitAttrib !== programInfo.attribLocations.vertexPosition) {
    bindQuad(gl, blitAttrib);
  }

  gl.disable(gl.BLEND);
  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.CULL_FACE);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
  gl.pixelStorei(gl.PACK_ALIGNMENT, 4);
  gl.useProgram(blitProgram);
  gl.uniform1i(gl.getUniformLocation(blitProgram, "uTex"), 0);
  gl.activeTexture(gl.TEXTURE0);

  if (localStorage.getItem("invert") === "1") {
    document.documentElement.classList.add("inverted");
  }

  document.getElementById("star").addEventListener("click", () => {
    document.documentElement.classList.toggle("inverted");
    localStorage.setItem("invert", document.documentElement.classList.contains("inverted") ? "1" : "0");
  });

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const webglStored = localStorage.getItem("webgl");
  let webglEnabled = webglStored === null ? true : webglStored !== "0";
  const webglToggleEl = document.getElementById("webgl-toggle");
  let raf = 0;

  function startLoop() {
    if (!raf) raf = requestAnimationFrame(render);
  }

  function toggleWebGL() {
    if (prefersReducedMotion) return;
    webglEnabled = !webglEnabled;
    localStorage.setItem("webgl", webglEnabled ? "1" : "0");
    canvas.style.display = webglEnabled ? "" : "none";
    webglToggleEl.classList.toggle("webgl-off", !webglEnabled);
    if (webglEnabled) startLoop();
  }

  canvas.style.display = webglEnabled ? "" : "none";
  webglToggleEl.classList.toggle("webgl-off", !webglEnabled);
  webglToggleEl.addEventListener("click", toggleWebGL);

  let mouseX = 0, mouseY = 0;
  let nextMouseX = 0, nextMouseY = 0;
  let prevNow = 0, progress = 0, easedProgress = 0;
  const minRadius = 0.25;
  const maxRadius = 1.2;
  let schwarzschildRadius = maxRadius;
  let targetRadius = maxRadius;
  let lastTouchY = 0;
  let displayWidth = canvas.clientWidth;
  let displayHeight = canvas.clientHeight;

  function adjustRadius(delta) {
    targetRadius = Math.max(minRadius, Math.min(maxRadius, targetRadius + delta));
  }

  new ResizeObserver(() => {
    displayWidth = canvas.clientWidth;
    displayHeight = canvas.clientHeight;
  }).observe(canvas);

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && webglEnabled) startLoop();
  });

  document.addEventListener("wheel", (e) => {
    adjustRadius(-e.deltaY * 0.00008);
  }, { passive: true });

  document.addEventListener("touchstart", (e) => {
    if (e.touches.length === 1) lastTouchY = e.touches[0].clientY;
  }, { passive: true });

  document.addEventListener("touchmove", (e) => {
    if (e.touches.length === 1) {
      const delta = lastTouchY - e.touches[0].clientY;
      lastTouchY = e.touches[0].clientY;
      adjustRadius(delta * 0.0005);
    }
  }, { passive: true });

  document.addEventListener("pointermove", (e) => {
    nextMouseX = e.clientX / window.innerWidth;
    nextMouseY = 1 - e.clientY / window.innerHeight;
  }, { passive: true });

  function renderScene(renderWidth, renderHeight, now) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.viewport(0, 0, renderWidth, renderHeight);
    gl.useProgram(programInfo.program);
    gl.uniform2f(programInfo.uniformLocations.resolution, renderWidth, renderHeight);
    gl.uniform1f(programInfo.uniformLocations.time, now * 0.001);
    gl.uniform1f(programInfo.uniformLocations.progress, easedProgress);
    gl.uniform2f(programInfo.uniformLocations.mouse, mouseX, mouseY);
    gl.uniform1f(programInfo.uniformLocations.schwarzschildRadius, schwarzschildRadius);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.flush();
  }

  function ditherAndBlit(src, renderWidth, renderHeight) {
    const dithered = applyDithering(src, renderWidth, renderHeight);
    blitDithered(gl, blitProgram, dithered, renderWidth, renderHeight, displayWidth, displayHeight);
  }

  function render(now) {
    raf = 0;
    if (!webglEnabled || document.hidden) return;
    raf = requestAnimationFrame(render);

    const dt = now - prevNow;
    prevNow = now;
    const timeScale = Math.abs(1 - (dt - 1));

    if (progress < 1) {
      progress = Math.min(1, progress + 0.0025);
      easedProgress = easeOutBack(progress);
    }

    mouseX += (nextMouseX - mouseX) * 0.01 * timeScale;
    mouseY += (nextMouseY - mouseY) * 0.01 * timeScale;
    schwarzschildRadius += (targetRadius - schwarzschildRadius) * 0.02 * timeScale;

    const renderWidth = Math.max(1, (displayWidth * RENDER_SCALE) | 0);
    const renderHeight = Math.max(1, (displayHeight * RENDER_SCALE) | 0);

    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
      initFramebuffer(gl, renderWidth, renderHeight, webgl2);
    }

    if (!framebuffer && displayWidth > 0 && displayHeight > 0) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
      initFramebuffer(gl, renderWidth, renderHeight, webgl2);
    }

    if (!framebuffer || canvas.width === 0 || canvas.height === 0) return;

    renderScene(renderWidth, renderHeight, now);

    if (webgl2 && pbos) {
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbos[pboIndex]);
      gl.readPixels(0, 0, renderWidth, renderHeight, gl.RGBA, gl.UNSIGNED_BYTE, 0);

      if (pboHasPrev) {
        const n = renderWidth * renderHeight;
        const dest = wasmExports ? wasmDitherViews(n).src : pixelBuffer;
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbos[pboIndex ^ 1]);
        gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, dest);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        ditherAndBlit(dest, renderWidth, renderHeight);
      } else {
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        pboHasPrev = true;
      }
      pboIndex ^= 1;
      return;
    }

    if (wasmExports) {
      const dest = wasmDitherViews(renderWidth * renderHeight).src;
      gl.readPixels(0, 0, renderWidth, renderHeight, gl.RGBA, gl.UNSIGNED_BYTE, dest);
      ditherAndBlit(dest, renderWidth, renderHeight);
      return;
    }

    gl.readPixels(0, 0, renderWidth, renderHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixelBuffer);
    ditherAndBlit(pixelBuffer, renderWidth, renderHeight);
  }

  startLoop();
}

init();

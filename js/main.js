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

function getFragmentShaderSource() {
  const mobile = isMobile();
  const maxSteps = mobile ? 84 : 128;
  const stepSize = mobile ? 0.25 : 0.15;

  return `
    precision highp float;
    uniform vec2 resolution;
    uniform float time;
    uniform vec2 mouse;
    uniform float progress;
    uniform float schwarzschildRadius;

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

      if (r > influenceRadius && dot(rd, p) > 0.0) return col;

      for (int i = 0; i < MAX_STEPS; i++) {
        r = length(p);
        float adaptiveStepSize = STEP_SIZE * max(1.0, r * 0.1);
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
      }
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
      gl_FragColor = rayMarch(ro, rd, uv * aspect * 5.0, schwarzschildRadius * progress) * progress;
    }
  `;
}

const VS_SOURCE = `
  attribute vec4 aVertexPosition;
  void main() { gl_Position = aVertexPosition; }
`;

const BLIT_VS = `
  attribute vec4 aVertexPosition;
  varying vec2 vTexCoord;
  void main() {
    gl_Position = aVertexPosition;
    vTexCoord = aVertexPosition.xy * 0.5 + 0.5;
  }
`;

const BLIT_FS = `
  precision highp float;
  uniform sampler2D uTex;
  varying vec2 vTexCoord;
  void main() { gl_FragColor = texture2D(uTex, vTexCoord); }
`;

const LUM_SCALE = 1 / 3;
const E7 = 7 / 16;
const E3 = 3 / 16;
const E5 = 5 / 16;
const E1 = 1 / 16;
const RENDER_SCALE = 1;

let ditherWork, ditherOutput, ditherOut32, ditherSrc32;

function applyDithering(pixels, width, height) {
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
  const w = width;
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
            if (x > 0) work[i + w - 1] += oldVal * E3;
            work[i + w] += oldVal * E5;
            if (x < lastX) work[i + w + 1] += oldVal * E1;
          }
        }
      } else {
        const err = oldVal - 255;
        out32[i] = src32[i] | 0xff000000;
        if (x < lastX) work[i + 1] += err * E7;
        if (notLastY) {
          if (x > 0) work[i + w - 1] += err * E3;
          work[i + w] += err * E5;
          if (x < lastX) work[i + w + 1] += err * E1;
        }
      }
    }
  }

  return ditherOutput;
}

let framebuffer, sceneTexture, displayTexture, pixelBuffer;

function initFramebuffer(gl, width, height) {
  if (framebuffer) {
    gl.deleteFramebuffer(framebuffer);
    gl.deleteTexture(sceneTexture);
    gl.deleteTexture(displayTexture);
  }

  sceneTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, sceneTexture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  displayTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, displayTexture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  framebuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, sceneTexture, 0);
  pixelBuffer = new Uint8Array(width * height * 4);
  ditherSrc32 = null;
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

  const program = initShaderProgram(gl, VS_SOURCE, getFragmentShaderSource());
  const blitProgram = initShaderProgram(gl, BLIT_VS, BLIT_FS);
  const buffers = initBuffers(gl);

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
      initFramebuffer(gl, renderWidth, renderHeight);
    }

    if (!framebuffer && displayWidth > 0 && displayHeight > 0) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
      initFramebuffer(gl, renderWidth, renderHeight);
    }

    if (!framebuffer || canvas.width === 0 || canvas.height === 0) return;

    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.viewport(0, 0, renderWidth, renderHeight);
    gl.useProgram(programInfo.program);
    gl.uniform2f(programInfo.uniformLocations.resolution, renderWidth, renderHeight);
    gl.uniform1f(programInfo.uniformLocations.time, now * 0.001);
    gl.uniform1f(programInfo.uniformLocations.progress, easedProgress);
    gl.uniform2f(programInfo.uniformLocations.mouse, mouseX, mouseY);
    gl.uniform1f(programInfo.uniformLocations.schwarzschildRadius, schwarzschildRadius);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.readPixels(0, 0, renderWidth, renderHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixelBuffer);

    const dithered = applyDithering(pixelBuffer, renderWidth, renderHeight);

    gl.bindTexture(gl.TEXTURE_2D, displayTexture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, renderWidth, renderHeight, gl.RGBA, gl.UNSIGNED_BYTE, dithered);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, displayWidth, displayHeight);
    gl.useProgram(blitProgram);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  startLoop();
}

init();

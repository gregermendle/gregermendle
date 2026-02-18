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
  const warpSize = 0.25;

  return `
    precision highp float;
    uniform vec2 resolution;
    uniform float time;
    uniform vec2 mouse;
    uniform float invert;
    uniform float progress;
    uniform float stepSize;
    uniform float schwarzschildRadius;

    #define MAX_STEPS ${maxSteps}
    #define PI 3.14159265359
    #define WARP_SIZE ${warpSize.toFixed(2)}
    #define DISK_SCALE 1.

    vec4 diskColor(float dist, float radius) {
      float d = (dist - radius) / (radius * DISK_SCALE);
      float innerTemp = smoothstep(2.0, 4.0, d);
      float outerTemp = smoothstep(4.0, 8.0, d);
      float gray = mix(1.0, mix(0.4, 0.1, outerTemp), innerTemp);
      gray = mix(gray, 1.0 - gray, invert);
      float intensity = exp(-d * 0.25);
      return vec4(vec3(gray * intensity), 1.0);
    }

    float hash(vec2 p) {
      return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }

    vec4 rayMarch(vec3 ro, vec3 rd, vec2 uv, float radius, vec2 aspect) {
      float jitter = hash(uv) * stepSize;
      vec3 p = ro + rd * jitter;
      float r = length(p);
      vec4 col = vec4(0.);
      float totalDist = jitter;
      const float diskThickness = 0.12;
      const float diskColorFalloff = 0.05;
      const float rotationSpeed = 0.3;
      const float spiralFreq = 8.0;
      const float spiralAmp = 0.08;
      float rdDotP = dot(rd, normalize(p));
      float influenceRadius = radius * 50.0;

      if (r > influenceRadius && rdDotP > 0.0) return col;

      float baseStepSize = stepSize;
      float adaptiveStepSize;
      bool inInfluenceZone = false;
      bool nearDisk = false;

      for (int i = 0; i < MAX_STEPS; i++) {
        r = length(p);
        adaptiveStepSize = baseStepSize * max(1.0, r * 0.1);
        inInfluenceZone = r < influenceRadius;

        if (inInfluenceZone) {
          float force = radius / (r * r);
          vec3 gravity = normalize(p) * -force;
          rd += gravity * adaptiveStepSize;
          rd = normalize(rd);
        } else {
          return col;
        }

        p += rd * adaptiveStepSize;
        totalDist += adaptiveStepSize;
        nearDisk = abs(p.y) < diskThickness;

        if (nearDisk) {
          float diskRadius = length(vec2(p.x, p.z));
          if (diskRadius > radius) {
            vec4 diskCol = diskColor(diskRadius, radius) * exp(-totalDist * 0.07);
            float angle = atan(p.z, p.x) + time * rotationSpeed;
            diskCol *= 1.0 + spiralAmp * sin(angle * spiralFreq);
            angle = atan(rd.x, rd.y) + time * rotationSpeed;
            diskCol *= .75 + spiralAmp * sin(angle * spiralFreq);
            col += diskCol * exp(-totalDist * diskColorFalloff);
          }
        }

        if (r < radius || totalDist > 100.0 || length(col) > 10.0) return col;
      }
      return col;
    }

    void main() {
      vec2 uv = (gl_FragCoord.xy - 0.5 * resolution.xy) / min(resolution.x, resolution.y);
      vec2 aspect = resolution.xy / min(resolution.x, resolution.y);
      vec2 muv = (mouse.xy - 0.5) * aspect;
      float warp = 1.0 - smoothstep(0.0, WARP_SIZE, length(uv - muv));
      vec2 outward = normalize(muv - uv + 0.001) * warp * 2.0;
      vec3 ro = vec3(-1.0, 2.0, -10.0) + vec3(muv * 2., 0.) + vec3(outward, 0.);
      vec3 rd = normalize(vec3(uv, 1.0)) - vec3(0., 0.2, 0.);
      vec4 col = rayMarch(ro, rd, uv * aspect * 5., schwarzschildRadius * progress, aspect);
      gl_FragColor = col * progress;
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

const DITHER_THRESHOLD = 256;
const LUM_SCALE = 1 / 3;
const RENDER_SCALE = 0.2;

let ditherLum, ditherWork, ditherOut, ditherOutput;

function applyDitheringAndAberration(pixels, width, height, aberrationOff) {
  const n = width * height;
  const w = width;
  const w4 = w * 4;

  if (!ditherLum || ditherLum.length < n) {
    ditherLum = new Float32Array(n);
    ditherWork = new Float32Array(n);
    ditherOut = new Uint8Array(n);
    ditherOutput = new Uint8Array(n * 4);
  }

  const out = ditherOutput;
  const off = Math.max(0, aberrationOff | 0);
  const clampX = (x) => (x < 0 ? 0 : x >= w ? w - 1 : x);

  for (let i = 0; i < n; i++) {
    const i4 = i * 4;
    ditherLum[i] = (pixels[i4] + pixels[i4 + 1] + pixels[i4 + 2]) * LUM_SCALE;
  }
  ditherWork.set(ditherLum);

  const e7 = 7 / 16, e3 = 3 / 16, e5 = 5 / 16, e1 = 1 / 16;
  for (let y = 0; y < height; y++) {
    const rowBase = y * w;
    for (let x = 0; x < w; x++) {
      const i = rowBase + x;
      const oldVal = ditherWork[i];
      const newVal = oldVal === 0 ? 0 : oldVal < DITHER_THRESHOLD ? 0 : 255;
      const err = oldVal - newVal;
      ditherOut[i] = newVal;
      if (x + 1 < w) ditherWork[i + 1] += err * e7;
      if (y + 1 < height) {
        if (x > 0) ditherWork[i + w - 1] += err * e3;
        ditherWork[i + w] += err * e5;
        if (x + 1 < w) ditherWork[i + w + 1] += err * e1;
      }
    }
  }

  for (let i = 0; i < n; i++) {
    const y = (i / w) | 0;
    const x = i % w;
    const d = ditherOut[i];
    const oi = i * 4;
    out[oi + 3] = 255;
    if (d === 0) {
      out[oi] = out[oi + 1] = out[oi + 2] = 0;
    } else if (off > 0) {
      const row = y * w4;
      out[oi] = pixels[row + clampX(x + off) * 4];
      out[oi + 1] = pixels[row + x * 4 + 1];
      out[oi + 2] = pixels[row + clampX(x - off) * 4 + 2];
    } else {
      out[oi] = pixels[oi];
      out[oi + 1] = pixels[oi + 1];
      out[oi + 2] = pixels[oi + 2];
    }
  }
  return out;
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
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
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
}

function easeOutBack(x) {
  const c1 = 1.70158;
  const c3 = c1 + 1;
  return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2);
}

function init() {
  const canvas = document.getElementById("canvas");
  const gl = canvas.getContext("webgl");
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
      invert: gl.getUniformLocation(program, "invert"),
      progress: gl.getUniformLocation(program, "progress"),
      stepSize: gl.getUniformLocation(program, "stepSize"),
      schwarzschildRadius: gl.getUniformLocation(program, "schwarzschildRadius"),
    },
  };

  const blitProgramInfo = {
    program: blitProgram,
    attribLocations: { vertexPosition: gl.getAttribLocation(blitProgram, "aVertexPosition") },
    uniformLocations: { uTex: gl.getUniformLocation(blitProgram, "uTex") },
  };

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

  function toggleWebGL() {
    if (prefersReducedMotion) return;
    webglEnabled = !webglEnabled;
    localStorage.setItem("webgl", webglEnabled ? "1" : "0");
    canvas.style.display = webglEnabled ? "" : "none";
    webglToggleEl.classList.toggle("webgl-off", !webglEnabled);
    if (webglEnabled) requestAnimationFrame(render);
  }

  canvas.style.display = webglEnabled ? "" : "none";
  webglToggleEl.classList.toggle("webgl-off", !webglEnabled);

  webglToggleEl.addEventListener("click", toggleWebGL);

  const stepSize = isMobile() ? 0.25 : 0.15;
  let mouseX = 0, mouseY = 0;
  let nextMouseX = 0, nextMouseY = 0;
  let prevNow = 0, progress = 0;
  const minRadius = 0.25;
  const maxRadius = 1.2;
  let schwarzschildRadius = isMobile() ? 0.6 : 0.4;
  let targetRadius = schwarzschildRadius;
  let lastTouchY = 0;
  function adjustRadius(delta) {
    targetRadius = Math.max(minRadius, Math.min(maxRadius, targetRadius + delta));
  }

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
  });

  function render(now) {
    const dt = now - prevNow;
    prevNow = now;
    const timeScale = Math.abs(1 - (dt - 1));
    const displayWidth = canvas.clientWidth;
    const displayHeight = canvas.clientHeight;

    progress = Math.min(1, progress + 0.0025);

    mouseX += (nextMouseX - mouseX) * 0.01 * timeScale;
    mouseY += (nextMouseY - mouseY) * 0.01 * timeScale;
    schwarzschildRadius += (targetRadius - schwarzschildRadius) * 0.02 * timeScale;

    const renderWidth = Math.max(1, (displayWidth * RENDER_SCALE) | 0);
    const renderHeight = Math.max(1, (displayHeight * RENDER_SCALE) | 0);

    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
      gl.viewport(0, 0, canvas.width, canvas.height);
      initFramebuffer(gl, renderWidth, renderHeight);
    }

    if (!framebuffer && canvas.width > 0 && canvas.height > 0) {
      initFramebuffer(gl, renderWidth, renderHeight);
    }

    if (!framebuffer || canvas.width === 0 || canvas.height === 0) {
      requestAnimationFrame(render);
      return;
    }

    if (!webglEnabled) return;

    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.viewport(0, 0, renderWidth, renderHeight);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(programInfo.program);
    gl.uniform2f(programInfo.uniformLocations.resolution, renderWidth, renderHeight);
    gl.uniform1f(programInfo.uniformLocations.time, now * 0.001);
    gl.uniform1f(programInfo.uniformLocations.progress, easeOutBack(progress));
    gl.uniform2f(programInfo.uniformLocations.mouse, mouseX, mouseY);
    gl.uniform1f(programInfo.uniformLocations.invert, 0);
    gl.uniform1f(programInfo.uniformLocations.stepSize, stepSize);
    gl.uniform1f(programInfo.uniformLocations.schwarzschildRadius, schwarzschildRadius);

    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.position);
    gl.vertexAttribPointer(programInfo.attribLocations.vertexPosition, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(programInfo.attribLocations.vertexPosition);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.readPixels(0, 0, renderWidth, renderHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixelBuffer);

    const dithered = applyDitheringAndAberration(pixelBuffer, renderWidth, renderHeight, 0);

    gl.bindTexture(gl.TEXTURE_2D, displayTexture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, renderWidth, renderHeight, gl.RGBA, gl.UNSIGNED_BYTE, dithered);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, displayWidth, displayHeight);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(blitProgramInfo.program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, displayTexture);
    gl.uniform1i(blitProgramInfo.uniformLocations.uTex, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.position);
    gl.vertexAttribPointer(blitProgramInfo.attribLocations.vertexPosition, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(blitProgramInfo.attribLocations.vertexPosition);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);
}

init();

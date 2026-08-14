(function (scope) {
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
      new Float32Array([-1, -1, 3, -1, -1, 3]),
      gl.STATIC_DRAW
    );
    return { position: buffer };
  }

  const DISK_SIZE = 43;
  const MAX_STARS = 48;
  const MAX_LINES = 36;
  const STAR_RADIUS = 0.0145;

  function marchStepSource() {
    return `
        r = length(p);
        adaptiveStepSize = STEP_SIZE * max(1.0, min(r * 0.1, influenceRadius * 0.07));
        adaptiveStepSize = min(adaptiveStepSize, stepCap);
        if (r < influenceRadius) {
          rd = normalize(rd - p * (radius * adaptiveStepSize / (r * r * r)));
        }
        prevP = p;
        p += rd * adaptiveStepSize;
        totalDist += adaptiveStepSize;
        if (prevP.y * p.y <= 0.0 || abs(p.y) < max(0.12, adaptiveStepSize * 0.55)) {
          vec3 hit = p;
          if (abs(rd.y) > 1e-4 && prevP.y * p.y <= 0.0) {
            hit = prevP + rd * (-prevP.y / rd.y);
          }
          float diskRadius = length(hit.xz);
          if (diskRadius > radius) {
            float d = (diskRadius - radius) / radius;
            float innerTemp = smoothstep(DISK_SIZE * 0.25, DISK_SIZE * 0.5, d);
            float outerTemp = smoothstep(DISK_SIZE * 0.5, DISK_SIZE, d);
            float gray = mix(1.0, mix(0.4, 0.1, outerTemp), innerTemp);
            vec4 diskCol = vec4(vec3(gray * exp(-d * (1.7 / DISK_SIZE))), 1.0);
            diskCol *= 1.0 + 0.08 * sin((atan(hit.z, hit.x) + tRot) * 8.0);
            diskCol *= 0.75 + 0.08 * sin((atan(rd.x, rd.y) + tRot) * 8.0);
            col += diskCol;
            col.a = 1.0;
          }
        }
        if (r < radius || dot(col, col) > 100.0) return vec4(col.rgb, 1.0);
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
    uniform vec3 camPos;
    uniform vec3 camFwd;
    uniform vec3 camRight;
    uniform vec3 camUp;
    uniform vec4 uStars[${MAX_STARS}];
    uniform vec3 uLineA[${MAX_LINES}];
    uniform vec3 uLineB[${MAX_LINES}];
    uniform int uStarCount;
    uniform int uLineCount;
    uniform float uWarp;
    uniform vec2 uWarpCenter;
    uniform vec2 uViewHalf;
    ${fragOut}
    #define MAX_STEPS ${maxSteps}
    #define WARP_SIZE 0.25
    #define STEP_SIZE ${stepSize.toFixed(2)}
    #define DISK_SIZE ${DISK_SIZE.toFixed(1)}
    #define MAX_STARS ${MAX_STARS}
    #define MAX_LINES ${MAX_LINES}
    #define STAR_COMPACT 0.000004

    float hash(vec2 p) {
      return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }

    vec3 pullRay(vec3 rd, vec3 rel, float mass, float maxT) {
      float t = dot(rel, rd);
      if (t <= 0.0 || t >= maxT) return rd;
      vec3 closest = rd * t - rel;
      float b2 = dot(closest, closest);
      float reach = mass * 48.0;
      if (b2 > reach * reach) return rd;
      float minB = mass * 1.15;
      float invB = inversesqrt(max(b2, minB * minB));
      return normalize(rd - closest * (mass * invB * invB * invB));
    }

    vec3 deflectRay(vec3 ro, vec3 rd, float maxT) {
      for (int i = 0; i < MAX_STARS; i++) {
        if (float(i) + 0.5 > float(uStarCount)) break;
        vec4 s = uStars[i];
        rd = pullRay(rd, s.xyz - ro, s.w * STAR_COMPACT, maxT);
      }
      return rd;
    }

    float einstein2(float mass, float zL, float zS) {
      float dls = zS - zL;
      if (dls <= 0.0 || zL <= 1e-4) return 0.0;
      return 2.0 * mass * dls / max(zL * zS, 1e-8);
    }

    vec2 lensPull(vec2 uv, vec2 lp, float te2) {
      if (te2 <= 0.0) return vec2(0.0);
      vec2 d = uv - lp;
      float b2 = dot(d, d);
      return d * (te2 / max(b2, te2 * 0.08));
    }

    vec4 rayMarch(vec3 ro, vec3 rd, vec2 uv, float radius) {
      float influenceRadius = max(radius * 50.0, radius * DISK_SIZE + 4.0);
      float stepCap = max(radius * 0.2, STEP_SIZE * 1.8);
      float b = dot(ro, rd);
      float c = dot(ro, ro) - influenceRadius * influenceRadius;
      float h = b * b - c;
      if (h < 0.0) return vec4(0.0);
      float tExit = -b + sqrt(h);
      if (tExit < 0.0) return vec4(0.0);
      float tEnter = max(-b - sqrt(h), 0.0);
      float jitter = hash(uv) * STEP_SIZE;
      vec3 p = ro + rd * (tEnter + jitter);
      float r = length(p);
      vec4 col = vec4(0.0);
      float totalDist = tEnter + jitter;
      float tRot = time * 0.3;
      float adaptiveStepSize;
      vec3 prevP;

      ${loop}
      return col;
    }

    float holeBeacon(vec2 uv, vec3 ro, float radius, float minRes) {
      vec3 toH = -ro;
      float z = dot(toH, camFwd);
      if (z <= 1e-4) return 0.0;
      vec2 sp = vec2(dot(toH, camRight), dot(toH, camUp)) / z;
      float px = 1.0 / minRes;
      float ang = max(radius * 8.0 / z, px * 1.15);
      float pad = max(ang * 6.0, px * 4.0);
      if (abs(sp.x) > uViewHalf.x + pad || abs(sp.y) > uViewHalf.y + pad) return 0.0;
      float coreR = max(ang, px * 1.15);
      float glowR = max(ang * 5.5, px * 3.4);
      float d = length(uv - sp);
      float core = smoothstep(coreR, coreR * 0.42, d);
      float halo = exp(-d / max(glowR, 1e-5)) * 0.55;
      return core + halo;
    }

    vec2 projectStar(vec3 w, vec3 ro) {
      vec3 d = w - ro;
      float z = dot(d, camFwd);
      if (z <= 1e-4) return vec2(100.0);
      return vec2(dot(d, camRight) / z, dot(d, camUp) / z);
    }

    float starField(vec2 uv, vec3 ro, float radius, float minRes, float holeMask) {
      float glow = 0.0;
      float px = 1.0 / minRes;
      float holeZ = dot(-ro, camFwd);
      vec2 holeP = holeZ > 1e-4
        ? vec2(dot(-ro, camRight), dot(-ro, camUp)) / holeZ
        : vec2(100.0);
      for (int i = 0; i < MAX_STARS; i++) {
        if (float(i) + 0.5 > float(uStarCount)) break;
        vec4 s = uStars[i];
        vec3 toS = s.xyz - ro;
        float z = dot(toS, camFwd);
        if (z <= 1e-4) continue;
        if (holeZ > 1e-4 && z > holeZ && holeMask > 0.5) continue;
        vec2 sp = vec2(dot(toS, camRight), dot(toS, camUp)) / z;
        float ang = s.w / max(z, s.w * 0.35);
        float holeTe2 = einstein2(radius, holeZ, z);
        float ring = sqrt(max(holeTe2, 0.0));
        float pad = max(max(ang * 6.0, px * 4.0), ring * 2.4);
        bool nearHole = holeZ > 1e-4 && holeZ < z && length(uv - holeP) < pad;
        if (!nearHole && (abs(sp.x) > uViewHalf.x + pad || abs(sp.y) > uViewHalf.y + pad)) continue;
        vec2 src = uv;
        src -= lensPull(uv, holeP, holeTe2);
        float blocked = 0.0;
        for (int j = 0; j < MAX_STARS; j++) {
          if (float(j) + 0.5 > float(uStarCount)) break;
          if (i == j) continue;
          vec4 o = uStars[j];
          vec3 toO = o.xyz - ro;
          float zj = dot(toO, camFwd);
          if (zj <= 1e-4 || zj >= z) continue;
          vec2 lp = vec2(dot(toO, camRight), dot(toO, camUp)) / zj;
          float angj = o.w / max(zj, o.w * 0.35);
          if (length(uv - lp) < max(angj, px * 1.15)) blocked = 1.0;
          src -= lensPull(uv, lp, einstein2(o.w * STAR_COMPACT, zj, z));
        }
        if (blocked > 0.5) continue;
        float coreR = max(ang, px * 1.15);
        float glowR = max(ang * 5.5, px * 3.4);
        float d = length(src - sp);
        float core = smoothstep(coreR, coreR * 0.42, d);
        float near = clamp(ang / (px * 10.0), 0.0, 1.0);
        float halo = exp(-d / max(glowR, 1e-5)) * mix(0.16, 0.52, near);
        glow += core + halo;
      }
      for (int i = 0; i < MAX_LINES; i++) {
        if (float(i) + 0.5 > float(uLineCount)) break;
        vec3 wa = uLineA[i];
        vec3 wb = uLineB[i];
        vec3 da = wa - ro;
        vec3 db = wb - ro;
        float za = dot(da, camFwd);
        float zb = dot(db, camFwd);
        if (za <= 1e-4 && zb <= 1e-4) continue;
        float zLine = min(za > 1e-4 ? za : zb, zb > 1e-4 ? zb : za);
        if (holeZ > 1e-4 && zLine > holeZ && holeMask > 0.5) continue;
        vec2 a = vec2(dot(da, camRight), dot(da, camUp)) / max(za, 1e-4);
        vec2 b = vec2(dot(db, camRight), dot(db, camUp)) / max(zb, 1e-4);
        if (holeZ > 1e-4 && holeZ < zLine) {
          float te2 = einstein2(radius, holeZ, zLine);
          a -= lensPull(a, holeP, te2);
          b -= lensPull(b, holeP, te2);
        }
        float pad = 0.35;
        float minX = min(a.x, b.x);
        float maxX = max(a.x, b.x);
        float minY = min(a.y, b.y);
        float maxY = max(a.y, b.y);
        if (maxX < -uViewHalf.x - pad || minX > uViewHalf.x + pad) continue;
        if (maxY < -uViewHalf.y - pad || minY > uViewHalf.y + pad) continue;
        vec2 pa = uv - a;
        vec2 ba = b - a;
        float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-5), 0.0, 1.0);
        float d = length(pa - ba * h) * minRes;
        glow += (1.0 - smoothstep(0.35, 0.85, d)) * 0.16;
      }
      return glow;
    }

    void main() {
      float minRes = min(resolution.x, resolution.y);
      vec2 uv = (gl_FragCoord.xy - 0.5 * resolution.xy) / minRes;
      vec2 aspect = resolution.xy / minRes;
      float w = uWarp;
      vec2 from = uv - uWarpCenter;
      float r = length(from);
      vec2 dir = r > 1e-5 ? from / r : vec2(0.0);
      float coneR = mix(0.14, 0.82, w);
      float t = r / max(coneR, 1e-4);
      float cone = max(1.0 - t, 0.0);
      float rim = smoothstep(0.72, 0.96, t) * (1.0 - smoothstep(0.96, 1.12, t));
      vec2 wuv = uv + dir * w * coneR * (0.48 * cone + 0.2 * rim);
      vec3 ro = camPos;
      float radius = schwarzschildRadius * progress;
      vec3 rd0 = normalize(wuv.x * camRight + wuv.y * camUp + camFwd);
      float holeT = max(-dot(ro, rd0), 0.0);
      if (holeT < 1e-4) holeT = 1.0e20;
      vec3 rd = deflectRay(ro, rd0, holeT);
      vec4 marched = rayMarch(ro, rd, wuv * aspect * 5.0, radius);
      vec4 col = vec4(marched.rgb * progress, marched.a);
      col.rgb += vec3(starField(wuv, ro, radius, minRes, marched.a) * progress);
      float dist = length(ro);
      float beaconMix = smoothstep(50.0, 160.0, dist);
      float marchLum = dot(col.rgb, vec3(0.333));
      float fill = beaconMix * (1.0 - smoothstep(0.0, 0.12, marchLum));
      float beacon = holeBeacon(wuv, ro, radius, minRes) * progress * fill;
      col.rgb = max(col.rgb, vec3(beacon));
      ${writeColor} = col;
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
    void main() { ${writeColor} = vec4(vec3(${tex}(uTex, vTexCoord).r), 1.0); }`;

    return { vs, fs, blitVs, blitFs };
  }

  const E7 = 7 / 16;
  const E3 = 3 / 16;
  const E5 = 5 / 16;
  const E1 = 1 / 16;
  const RENDER_SCALE = 1;

  let wasmExports = null;

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

  // Floyd-Steinberg over a single-channel image. Only two rows of diffused
  // error are live at a time: `cur` holds the row being thresholded and
  // `carry`/`a1`/`a2` hold the neighbours still being accumulated.
  let ditherCur = null;
  let ditherNext = null;

  function ditherJS(gray, out, width, height) {
    const n = width * height;
    const fround = Math.fround;
    let cur = ditherCur;
    let next = ditherNext;
    if (!cur || cur.length < width + 4) {
      cur = new Float32Array(width + 4);
      next = new Float32Array(width + 4);
    }
    for (let x = 0; x < width; x++) cur[x + 1] = gray[x];
    let row = 0;
    for (let y = 0; y < height; y++) {
      const nextRow = row + width;
      let carry = 0;
      let a2 = 0;
      let a1 = nextRow < n ? gray[nextRow] : 0;
      for (let x = 0; x < width; x++) {
        const v = fround(cur[x + 1] + carry);
        const lit = v >= 256;
        out[row + x] = lit ? gray[row + x] : 0;
        carry = v * E7 - (lit ? 255 * E7 : 0);
        const e = v - (lit ? 255 : 0);
        next[x] = a2 + e * E3;
        a2 = fround(a1 + e * E5);
        const j = nextRow + x + 1;
        a1 = fround((j < n ? gray[j] : 0) + e * E1);
      }
      next[width] = a2;
      const swap = cur;
      cur = next;
      next = swap;
      row = nextRow;
    }
    ditherCur = cur;
    ditherNext = next;
  }

  function grayJS(rgba, gray, n) {
    for (let i = 0, j = 0; i < n; i++, j += 4) gray[i] = rgba[j];
  }

  function align16(v) {
    return (v + 15) & ~15;
  }

  function rowStride(width, bytesPerPixel) {
    return (width * bytesPerPixel + 3) & ~3;
  }

  function copyRows(src, dst, width, height, srcStride, dstStride) {
    if (srcStride === dstStride && src === dst) return;
    if (srcStride === dstStride) {
      dst.set(src.subarray(0, dstStride * height));
      return;
    }
    for (let y = 0, s = 0, d = 0; y < height; y++, s += srcStride, d += dstStride) {
      dst.set(src.subarray(s, s + width), d);
    }
  }

  function matchesPattern(buf, stride, expected, width) {
    for (let i = 0; i < expected.length; i++) {
      const x = i % width;
      const y = (i / width) | 0;
      if (buf[y * stride + x] !== expected[i]) return false;
    }
    return true;
  }

  function probeSingleChannelTarget(gl, webgl2) {
    while (gl.getError() !== gl.NO_ERROR) {}
    const w = 5;
    const h = 2;
    const pattern = new Uint8Array([11, 22, 33, 44, 55, 66, 77, 88, 99, 110]);
    const stride = rowStride(w, 1);
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, w, h, 0, gl.RED, gl.UNSIGNED_BYTE, pattern);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    const complete = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    const formatOk =
      complete &&
      gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_FORMAT) === gl.RED &&
      gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_TYPE) === gl.UNSIGNED_BYTE;
    let syncOk = false;
    let pboOk = false;
    if (formatOk) {
      gl.pixelStorei(gl.PACK_ALIGNMENT, 4);
      const packed = new Uint8Array(stride * h);
      gl.readPixels(0, 0, w, h, gl.RED, gl.UNSIGNED_BYTE, packed);
      syncOk = gl.getError() === gl.NO_ERROR && matchesPattern(packed, stride, pattern, w);
      if (syncOk && webgl2) {
        const pbo = gl.createBuffer();
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo);
        gl.bufferData(gl.PIXEL_PACK_BUFFER, stride * h, gl.STREAM_READ);
        gl.readPixels(0, 0, w, h, gl.RED, gl.UNSIGNED_BYTE, 0);
        const out = new Uint8Array(stride * h);
        gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, out);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        pboOk = gl.getError() === gl.NO_ERROR && matchesPattern(out, stride, pattern, w);
        gl.deleteBuffer(pbo);
      }
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(fb);
    gl.deleteTexture(tex);
    return { single: syncOk, pbo: pboOk };
  }

  function easeOutBack(x) {
    const c1 = 1.70158;
    const c3 = c1 + 1;
    return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2);
  }

  function createRenderer(canvas) {
    const glAttrs = {
      alpha: false,
      depth: false,
      stencil: false,
      antialias: false,
      powerPreference: "high-performance",
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
      desynchronized: false,
      failIfMajorPerformanceCaveat: false,
    };
    const gl =
      canvas.getContext("webgl2", glAttrs) || canvas.getContext("webgl", glAttrs);
    if (!gl) return null;

    const webgl2 =
      typeof WebGL2RenderingContext !== "undefined" &&
      gl instanceof WebGL2RenderingContext;
    const shaders = getShaders(webgl2);
    const program = initShaderProgram(gl, shaders.vs, shaders.fs);
    const blitProgram = initShaderProgram(gl, shaders.blitVs, shaders.blitFs);
    const buffers = initBuffers(gl);
    loadWasmDither();

    const starData = new Float32Array(MAX_STARS * 4);
    const lineAData = new Float32Array(MAX_LINES * 3);
    const lineBData = new Float32Array(MAX_LINES * 3);
    let clusters = [];
    let starCount = 0;
    let lineCount = 0;

    fetch("/js/clusters.json")
      .then((res) => res.json())
      .then((data) => {
        clusters = data.clusters || [];
      })
      .catch(() => {
        clusters = [];
      });

    function starInView(b, wx, wy, wz, starR, minRes, renderW, renderH) {
      const dx = wx - camX;
      const dy = wy - camY;
      const dz = wz - camZ;
      const z = dx * b.fx + dy * b.fy + dz * b.fz;
      if (z <= 0.05) return false;
      const sx = (dx * b.rx + dy * b.ry + dz * b.rz) / z;
      const sy = (dx * b.ux + dy * b.uy + dz * b.uz) / z;
      const pad = (starR / z) * 8 + 0.18;
      const halfX = renderW / (2 * minRes) + pad;
      const halfY = renderH / (2 * minRes) + pad;
      return Math.abs(sx) <= halfX && Math.abs(sy) <= halfY;
    }

    function packClusters(now, b, renderW, renderH) {
      const t = now * 0.001;
      const minRes = Math.min(renderW, renderH);
      const candidates = [];
      for (let c = 0; c < clusters.length; c++) {
        const cluster = clusters[c];
        const pts = cluster.points || [];
        if (!pts.length) continue;
        const origin = cluster.origin || [0, 0, 0];
        const glide = cluster.glide || {};
        const amp = glide.amp || [0, 0, 0];
        const speed = glide.speed || 0;
        const phase = glide.phase || 0;
        const ox = origin[0] + amp[0] * Math.sin(t * speed + phase);
        const oy = origin[1] + amp[1] * Math.sin(t * speed * 0.83 + phase + 1.1);
        const oz = origin[2] + amp[2] * Math.cos(t * speed * 0.71 + phase);
        const clusterR = cluster.radius || STAR_RADIUS;
        const visible = [];
        let minDist2 = Infinity;
        for (let i = 0; i < pts.length; i++) {
          const p = pts[i];
          const wx = ox + (p[0] || 0);
          const wy = oy + (p[1] || 0);
          const wz = oz + (p[2] || 0);
          const sr = p[3] || clusterR;
          if (!starInView(b, wx, wy, wz, sr, minRes, renderW, renderH)) continue;
          const dx = wx - camX;
          const dy = wy - camY;
          const dz = wz - camZ;
          const dist2 = dx * dx + dy * dy + dz * dz;
          if (dist2 < minDist2) minDist2 = dist2;
          visible.push({ wx, wy, wz, sr });
        }
        if (visible.length) candidates.push({ minDist2, visible });
      }
      candidates.sort((a, b) => a.minDist2 - b.minDist2);
      starCount = 0;
      lineCount = 0;
      for (let c = 0; c < candidates.length && starCount < MAX_STARS; c++) {
        const cluster = candidates[c];
        const first = starCount;
        for (let i = 0; i < cluster.visible.length && starCount < MAX_STARS; i++) {
          const star = cluster.visible[i];
          const o = starCount * 4;
          starData[o] = star.wx;
          starData[o + 1] = star.wy;
          starData[o + 2] = star.wz;
          starData[o + 3] = star.sr;
          starCount++;
        }
        for (let i = first + 1; i < starCount && lineCount < MAX_LINES; i++) {
          const a = (i - 1) * 4;
          const bi = i * 4;
          const lo = lineCount * 3;
          lineAData[lo] = starData[a];
          lineAData[lo + 1] = starData[a + 1];
          lineAData[lo + 2] = starData[a + 2];
          lineBData[lo] = starData[bi];
          lineBData[lo + 1] = starData[bi + 1];
          lineBData[lo + 2] = starData[bi + 2];
          lineCount++;
        }
      }
    }

    const readbackCaps = webgl2 ? probeSingleChannelTarget(gl, true) : { single: false, pbo: false };
    const singleTarget = readbackCaps.single;
    const usePbo = webgl2 && (singleTarget ? readbackCaps.pbo : true);
    const sceneInternal = singleTarget ? gl.R8 : gl.RGBA;
    const sceneFormat = singleTarget ? gl.RED : gl.RGBA;
    const displayInternal = webgl2 ? gl.R8 : gl.LUMINANCE;
    const displayFormat = webgl2 ? gl.RED : gl.LUMINANCE;
    const sceneBytes = singleTarget ? 1 : 4;

    const programInfo = {
      program,
      attribLocations: {
        vertexPosition: gl.getAttribLocation(program, "aVertexPosition"),
      },
      uniformLocations: {
        resolution: gl.getUniformLocation(program, "resolution"),
        time: gl.getUniformLocation(program, "time"),
        mouse: gl.getUniformLocation(program, "mouse"),
        progress: gl.getUniformLocation(program, "progress"),
        schwarzschildRadius: gl.getUniformLocation(program, "schwarzschildRadius"),
        camPos: gl.getUniformLocation(program, "camPos"),
        camFwd: gl.getUniformLocation(program, "camFwd"),
        camRight: gl.getUniformLocation(program, "camRight"),
        camUp: gl.getUniformLocation(program, "camUp"),
        stars: gl.getUniformLocation(program, "uStars[0]"),
        lineA: gl.getUniformLocation(program, "uLineA[0]"),
        lineB: gl.getUniformLocation(program, "uLineB[0]"),
        starCount: gl.getUniformLocation(program, "uStarCount"),
        lineCount: gl.getUniformLocation(program, "uLineCount"),
        warp: gl.getUniformLocation(program, "uWarp"),
        warpCenter: gl.getUniformLocation(program, "uWarpCenter"),
        viewHalf: gl.getUniformLocation(program, "uViewHalf"),
      },
    };

    const blitAttrib = gl.getAttribLocation(blitProgram, "aVertexPosition");
    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.position);
    gl.vertexAttribPointer(programInfo.attribLocations.vertexPosition, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(programInfo.attribLocations.vertexPosition);
    if (blitAttrib !== programInfo.attribLocations.vertexPosition) {
      gl.vertexAttribPointer(blitAttrib, 2, gl.FLOAT, false, 0, 0);
      gl.enableVertexAttribArray(blitAttrib);
    }

    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.pixelStorei(gl.PACK_ALIGNMENT, 4);
    gl.useProgram(blitProgram);
    gl.uniform1i(gl.getUniformLocation(blitProgram, "uTex"), 0);
    gl.activeTexture(gl.TEXTURE0);

    let framebuffer = null;
    let sceneTexture = null;
    let displayTexture = null;
    let pbos = null;
    let pboIndex = 0;
    let pboHasPrev = false;
    let fboWidth = 0;
    let fboHeight = 0;
    let packStride = 0;
    let uploadStride = 0;
    let hasPresented = false;

    let heap = null;
    let graySrc = null;
    let grayDst = null;
    let rgbaStage = null;
    let packDest = null;
    let uploadPack = null;
    let rowsPtr = 0;
    let srcPtr = 0;
    let dstPtr = 0;

    function createTexture(width, height, internal, format) {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, internal, width, height, 0, format, gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return tex;
    }

    function allocScratch(width, height) {
      const n = width * height;
      packStride = rowStride(width, sceneBytes);
      uploadStride = rowStride(width, 1);
      srcPtr = 0;
      dstPtr = align16(n + width + 16);
      rowsPtr = align16(dstPtr + n);
      const packPtr = align16(rowsPtr + 2 * (width + 4) * 4);
      const packBytes = packStride * height;
      const uploadPtr = align16(packPtr + packBytes);
      const uploadBytes = uploadStride === width ? 0 : uploadStride * height;
      const total = uploadPtr + uploadBytes;
      if (wasmExports) {
        wasmExports.ensure(total);
        heap = wasmExports.memory.buffer;
        graySrc = new Uint8Array(heap, srcPtr, n);
        grayDst = new Uint8Array(heap, dstPtr, n);
        packDest = new Uint8Array(heap, packPtr, packBytes);
        uploadPack = uploadBytes ? new Uint8Array(heap, uploadPtr, uploadBytes) : null;
        rgbaStage = singleTarget ? null : packDest;
      } else {
        heap = null;
        graySrc = new Uint8Array(n);
        grayDst = new Uint8Array(n);
        packDest = new Uint8Array(packBytes);
        uploadPack = uploadBytes ? new Uint8Array(uploadBytes) : null;
        rgbaStage = singleTarget ? null : packDest;
      }
    }

    function destroyPbos() {
      if (!pbos) return;
      gl.deleteBuffer(pbos[0]);
      gl.deleteBuffer(pbos[1]);
      pbos = null;
      pboHasPrev = false;
    }

    function initFramebuffer(width, height) {
      if (framebuffer) {
        gl.deleteFramebuffer(framebuffer);
        gl.deleteTexture(sceneTexture);
        gl.deleteTexture(displayTexture);
      }
      destroyPbos();

      sceneTexture = createTexture(width, height, sceneInternal, sceneFormat);
      displayTexture = createTexture(width, height, displayInternal, displayFormat);
      framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, sceneTexture, 0);
      fboWidth = width;
      fboHeight = height;
      hasPresented = false;
      allocScratch(width, height);

      if (usePbo) {
        const bytes = packStride * height;
        pbos = [gl.createBuffer(), gl.createBuffer()];
        for (let i = 0; i < 2; i++) {
          gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbos[i]);
          gl.bufferData(gl.PIXEL_PACK_BUFFER, bytes, gl.STREAM_READ);
        }
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        pboIndex = 0;
      }
    }

    function refreshViews() {
      if (wasmExports && heap !== wasmExports.memory.buffer) {
        allocScratch(fboWidth, fboHeight);
      }
    }

    function dither(width, height) {
      if (wasmExports) {
        wasmExports.dither(srcPtr, dstPtr, rowsPtr, width, height);
      } else {
        ditherJS(graySrc, grayDst, width, height);
      }
    }

    function toGray(n) {
      if (wasmExports) {
        wasmExports.gray(rgbaStage.byteOffset, srcPtr, n);
      } else {
        grayJS(rgbaStage, graySrc, n);
      }
    }

    let running = false;
    let hidden = false;
    let raf = 0;
    let mouseX = 0;
    let mouseY = 0;
    let nextMouseX = 0;
    let nextMouseY = 0;
    let prevNow = 0;
    let haveTime = false;
    let progress = 0;
    let easedProgress = 0;
    const minRadius = 0.08;
    const maxRadius = 1.2;
    let schwarzschildRadius = 0.25;
    let targetRadius = 0.25;
    let camX = -2;
    let camY = 6;
    let camZ = -26;
    let camYaw = 0;
    let camPitch = -0.18;
    let lookYaw = 0;
    let lookPitch = -0.18;
    let velX = 0;
    let velY = 0;
    let velZ = 0;
    let keyF = 0;
    let keyB = 0;
    let keyL = 0;
    let keyR = 0;
    let keyU = 0;
    let keyD = 0;
    let keyBoost = 0;
    let engine = 0;
    let screenWarp = 0;
    let warpCenterX = 0;
    let warpCenterY = 0;
    let displayWidth = 0;
    let displayHeight = 0;

    function camBasis() {
      const cp = Math.cos(camPitch);
      const sp = Math.sin(camPitch);
      const cy = Math.cos(camYaw);
      const sy = Math.sin(camYaw);
      const fx = sy * cp;
      const fy = sp;
      const fz = cy * cp;
      let rx = fz;
      let rz = -fx;
      const rl = Math.hypot(rx, rz) || 1;
      rx /= rl;
      rz /= rl;
      const ux = fy * rz;
      const uy = fz * rx - fx * rz;
      const uz = -fy * rx;
      return { fx, fy, fz, rx, ry: 0, rz, ux, uy, uz };
    }

    function keepOut() {
      if (!Number.isFinite(camX + camY + camZ + velX + velY + velZ)) {
        camX = -2;
        camY = 6;
        camZ = -26;
        velX = 0;
        velY = 0;
        velZ = 0;
      }
    }

    function renderScene(renderWidth, renderHeight, now) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.viewport(0, 0, renderWidth, renderHeight);
      gl.useProgram(programInfo.program);
      gl.uniform2f(programInfo.uniformLocations.resolution, renderWidth, renderHeight);
      gl.uniform1f(programInfo.uniformLocations.time, now * 0.001);
      gl.uniform1f(programInfo.uniformLocations.progress, easedProgress);
      gl.uniform2f(programInfo.uniformLocations.mouse, mouseX, mouseY);
      gl.uniform1f(programInfo.uniformLocations.schwarzschildRadius, schwarzschildRadius);
      const b = camBasis();
      gl.uniform3f(programInfo.uniformLocations.camPos, camX, camY, camZ);
      gl.uniform3f(programInfo.uniformLocations.camFwd, b.fx, b.fy, b.fz);
      gl.uniform3f(programInfo.uniformLocations.camRight, b.rx, b.ry, b.rz);
      gl.uniform3f(programInfo.uniformLocations.camUp, b.ux, b.uy, b.uz);
      const minRes = Math.min(renderWidth, renderHeight);
      gl.uniform2f(
        programInfo.uniformLocations.viewHalf,
        renderWidth / (2 * minRes),
        renderHeight / (2 * minRes)
      );
      packClusters(now, b, renderWidth, renderHeight);
      gl.uniform4fv(programInfo.uniformLocations.stars, starData);
      gl.uniform3fv(programInfo.uniformLocations.lineA, lineAData);
      gl.uniform3fv(programInfo.uniformLocations.lineB, lineBData);
      gl.uniform1i(programInfo.uniformLocations.starCount, starCount);
      gl.uniform1i(programInfo.uniformLocations.lineCount, lineCount);
      gl.uniform1f(programInfo.uniformLocations.warp, screenWarp);
      gl.uniform2f(programInfo.uniformLocations.warpCenter, warpCenterX, warpCenterY);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.flush();
    }

    function present(width, height, upload) {
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      if (gl.PIXEL_UNPACK_BUFFER) gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, null);
      gl.bindTexture(gl.TEXTURE_2D, displayTexture);
      if (upload) {
        let pixels = grayDst;
        if (uploadStride !== width) {
          copyRows(grayDst, uploadPack, width, height, width, uploadStride);
          pixels = uploadPack;
        }
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, displayFormat, gl.UNSIGNED_BYTE, pixels);
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, displayWidth, displayHeight);
      gl.useProgram(blitProgram);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      hasPresented = true;
    }

    function readDest(width) {
      return singleTarget && packStride === width ? graySrc : packDest;
    }

    function unpackRead(width, height) {
      if (singleTarget) {
        if (packStride !== width) copyRows(packDest, graySrc, width, height, packStride, width);
      } else {
        toGray(width * height);
      }
    }

    function readback(renderWidth, renderHeight) {
      const dest = readDest(renderWidth);
      if (pbos) {
        const write = pboIndex;
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbos[write]);
        gl.readPixels(0, 0, renderWidth, renderHeight, sceneFormat, gl.UNSIGNED_BYTE, 0);
        pboIndex ^= 1;
        if (!pboHasPrev) {
          pboHasPrev = true;
          gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
          return false;
        }
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbos[pboIndex]);
        gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, dest);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      } else {
        gl.readPixels(0, 0, renderWidth, renderHeight, sceneFormat, gl.UNSIGNED_BYTE, dest);
      }

      unpackRead(renderWidth, renderHeight);
      return true;
    }

    function render(now) {
      raf = 0;
      if (!running || hidden) return;
      raf = requestAnimationFrame(render);

      let timeScale;
      if (!haveTime) {
        haveTime = true;
        prevNow = now;
        timeScale = 15;
      } else {
        const dt = now - prevNow;
        prevNow = now;
        timeScale = Math.abs(1 - (dt - 1));
        if (timeScale > 32) timeScale = 32;
      }

      if (progress < 1) {
        progress = Math.min(1, progress + 0.0025);
        easedProgress = easeOutBack(progress);
      }

      mouseX += (nextMouseX - mouseX) * 0.01 * timeScale;
      mouseY += (nextMouseY - mouseY) * 0.01 * timeScale;
      schwarzschildRadius += (targetRadius - schwarzschildRadius) * 0.02 * timeScale;
      camYaw += (lookYaw - camYaw) * 0.14 * timeScale;
      camPitch += (lookPitch - camPitch) * 0.14 * timeScale;
      const b = camBasis();
      let wishX = b.fx * (keyF - keyB) + b.rx * (keyR - keyL) + b.ux * (keyU - keyD);
      let wishY = b.fy * (keyF - keyB) + b.ry * (keyR - keyL) + b.uy * (keyU - keyD);
      let wishZ = b.fz * (keyF - keyB) + b.rz * (keyR - keyL) + b.uz * (keyU - keyD);
      let wishLen = Math.hypot(wishX, wishY, wishZ);
      if (keyBoost && wishLen < 1e-6) {
        wishX = b.fx;
        wishY = b.fy;
        wishZ = b.fz;
        wishLen = 1;
      }
      const thrusting = wishLen > 1e-6;
      const targetEngine = thrusting ? (keyBoost ? 1 : 0.4) : 0;
      const spool = targetEngine > engine ? (keyBoost ? 0.0042 : 0.0024) : 0.0055;
      engine += (targetEngine - engine) * spool * timeScale;
      if (engine < 1e-4) engine = 0;
      if (thrusting) {
        const inv = 1 / wishLen;
        const accel = engine * (keyBoost ? 0.0001 : 0.0000085) * timeScale;
        velX += wishX * inv * accel;
        velY += wishY * inv * accel;
        velZ += wishZ * inv * accel;
      }
      const dragMul = Math.pow(thrusting ? 0.99992 : 0.99974, timeScale);
      velX *= dragMul;
      velY *= dragMul;
      velZ *= dragMul;
      const cruiseSpeed = 0.007;
      const warpSpeed = 0.075;
      const speed = Math.hypot(velX, velY, velZ);
      const cap = keyBoost ? warpSpeed : cruiseSpeed;
      if (speed > cap) {
        let next = cap;
        if (!keyBoost) {
          next = cap + (speed - cap) * Math.pow(0.985, timeScale);
        }
        const s = next / speed;
        velX *= s;
        velY *= s;
        velZ *= s;
      }
      camX += velX * timeScale;
      camY += velY * timeScale;
      camZ += velZ * timeScale;
      keepOut();
      const flySpeed = Math.hypot(velX, velY, velZ);
      const targetWarp = Math.min(1, flySpeed / warpSpeed);
      screenWarp += (targetWarp - screenWarp) * 0.028 * timeScale;
      if (screenWarp < 1e-4) screenWarp = 0;
      if (flySpeed > 1e-8) {
        const inv = 1 / flySpeed;
        const dx = velX * inv;
        const dy = velY * inv;
        const dz = velZ * inv;
        const vz = dx * b.fx + dy * b.fy + dz * b.fz;
        const vx = dx * b.rx + dy * b.ry + dz * b.rz;
        const vy = dx * b.ux + dy * b.uy + dz * b.uz;
        let nextX;
        let nextY;
        if (vz > 0.04) {
          nextX = vx / vz;
          nextY = vy / vz;
        } else {
          const sl = Math.hypot(vx, vy) || 1;
          nextX = (vx / sl) * 4;
          nextY = (vy / sl) * 4;
        }
        const follow = Math.min(1, 0.012 * timeScale);
        warpCenterX += (nextX - warpCenterX) * follow;
        warpCenterY += (nextY - warpCenterY) * follow;
      }

      if (displayWidth <= 0 || displayHeight <= 0) return;

      const renderWidth = Math.max(1, (displayWidth * RENDER_SCALE) | 0);
      const renderHeight = Math.max(1, (displayHeight * RENDER_SCALE) | 0);

      if (canvas.width !== displayWidth || canvas.height !== displayHeight || !framebuffer) {
        canvas.width = displayWidth;
        canvas.height = displayHeight;
        initFramebuffer(renderWidth, renderHeight);
      }

      refreshViews();
      renderScene(renderWidth, renderHeight, now);
      if (readback(renderWidth, renderHeight)) {
        dither(renderWidth, renderHeight);
        present(renderWidth, renderHeight, true);
      } else if (hasPresented) {
        present(renderWidth, renderHeight, false);
      }
    }

    function start() {
      if (!raf && running && !hidden) raf = requestAnimationFrame(render);
    }

    return {
      setSize(width, height) {
        displayWidth = width | 0;
        displayHeight = height | 0;
      },
      setPointer(x, y) {
        nextMouseX = x;
        nextMouseY = y;
      },
      adjustRadius(delta) {
        targetRadius = Math.max(minRadius, Math.min(maxRadius, targetRadius + delta));
      },
      look(dx, dy) {
        lookYaw += dx;
        lookPitch = Math.max(-1.2, Math.min(1.2, lookPitch - dy));
      },
      thrust(delta) {
        const b = camBasis();
        velX += b.fx * delta * 0.002;
        velY += b.fy * delta * 0.002;
        velZ += b.fz * delta * 0.002;
      },
      setKeys(keys) {
        keyF = keys.f ? 1 : 0;
        keyB = keys.b ? 1 : 0;
        keyL = keys.l ? 1 : 0;
        keyR = keys.r ? 1 : 0;
        keyU = keys.u ? 1 : 0;
        keyD = keys.d ? 1 : 0;
        keyBoost = keys.boost ? 1 : 0;
      },
      setHidden(value) {
        hidden = value;
        start();
      },
      setRunning(value) {
        running = value;
        start();
      },
    };
  }

  scope.createRenderer = createRenderer;
})(self);

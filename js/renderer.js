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

  const RS_REF = 0.25;
  const STAR_RADIUS = 0.32 * RS_REF;
  const VIEW_FRUSTUM_PAD = 0.32;
  const STAR_ANGULAR_PAD = 12;
  const STAR_BASE_PAD = 0.32;
  const STAR_GLOW_PAD = 9;
  const STAR_PX_PAD = 6;
  const MIN_LENS_Z = 2.5;
  const STAR_FAR_Z = 2800;
  const LENS_Z_MIN = 0.8;
  const ENABLE_DITHER = false;
  const LENS_FADE_BEHIND = -12;
  const LENS_FADE_FRONT = 0.6;
  const LENS_FAR_START = 80;
  const LENS_FAR_END = 320;
  const GRAV_G = 0.0011;
  const GRAV_SOFT2 = 2.25;
  const GRAV_STEP = 8.5;
  const GRAV_BH_MASS = 280000;
  const GRAV_STAR_MASS = 420;
  const GRAV_NEAR2 = 36 * 36;
  const GRAV_KICK_DT = 0.42;
  const GRAV_MAX_KICKS = 8;
  const GRAV_KICKS_MID = 5;
  const GRAV_KICKS_FAR = 3;
  const GRAV_CAM_MID = 100;
  const GRAV_CAM_FAR = 220;
  const GRAV_KICK_MUL = 24;
  const SIM_TIME_MIN = 0.001;
  const SIM_TIME_MAX = 1e12;
  const SIM_WARMUP_SCALE = 8;
  const SIM_WARMUP_SEC = 10;
  const DUST_GAIN_MUL = 0.85;
  const DUST_COUNT_MUL = 0.55;
  const GALAXY_OUT_SCALE = 2;

  function getShaders(webgl2) {
    const ver = webgl2 ? "#version 300 es\n" : "";
    const fragOut = webgl2 ? "out vec4 fragColor;\n" : "";
    const writeColor = webgl2 ? "fragColor" : "gl_FragColor";
    const attr = webgl2 ? "in" : "attribute";
    const varyOut = webgl2 ? "out" : "varying";
    const varyIn = webgl2 ? "in" : "varying";
    const tex = webgl2 ? "texture" : "texture2D";

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

    const maskCh = webgl2 ? "g" : "a";
    const lensLib = `
    float lensFade(float holeZ) {
      return 1.0;
    }
    float holeTe2Of(float mass, float holeZ, float z) {
      float fade = lensFade(holeZ);
      if (fade <= 0.0 || z <= 0.0) return 0.0;
      float zL = max(holeZ, ${LENS_Z_MIN.toFixed(2)});
      float dls = z - zL;
      if (dls <= 0.0) return 0.0;
      return fade * 2.0 * mass * dls / max(zL * z, 1e-8);
    }
    float lensMassOf(float rs) {
      return rs * rs / ${RS_REF.toFixed(2)};
    }
    vec2 holePos(vec3 ro) {
      float zProj = max(dot(-ro, camFwd), ${LENS_Z_MIN.toFixed(2)});
      return vec2(dot(-ro, camRight), dot(-ro, camUp)) / zProj;
    }`;
    const starVs = `${ver}${attr} vec2 aCorner;
    ${attr} vec4 aStar;
    ${attr} vec3 aColor;
    ${attr} float aKind;
    uniform vec2 resolution;
    uniform float progress;
    uniform float schwarzschildRadius;
    uniform vec3 camPos;
    uniform vec3 camFwd;
    uniform vec3 camRight;
    uniform vec3 camUp;
    uniform float uUseLens;
    ${varyOut} vec4 vStar;
    ${varyOut} vec3 vColor;
    ${varyOut} float vKind;
    ${lensLib}
    void main() {
      float minRes = min(resolution.x, resolution.y);
      float px = 1.0 / minRes;
      vec3 ro = camPos;
      float radius = schwarzschildRadius * progress;
      float holeZ = dot(-ro, camFwd);
      vec2 holeP = holePos(ro);
      float z = aStar.z;
      float ang = aStar.w;
      float holeTe2 = uUseLens > 0.5 ? holeTe2Of(lensMassOf(radius), holeZ, z) : 0.0;
      if (holeTe2 < px * px) holeTe2 = 0.0;
      float ring = sqrt(holeTe2);
      float glowR = max(ang * 5.5, px * 3.4);
      float pad = aKind > 1.5
        ? max(ang * 2.2, px * 3.0)
        : max(max(ang * ${STAR_GLOW_PAD.toFixed(1)}, px * ${STAR_PX_PAD.toFixed(1)}), glowR * 6.0);
      if (uUseLens > 0.5 && aKind < 1.5) pad = max(pad, ring * 2.4);
      vec2 center = (uUseLens > 0.5 && aKind > 0.5 && aKind < 1.5) ? holeP : aStar.xy;
      vec2 uv = center + aCorner * pad;
      gl_Position = vec4(uv.x * (2.0 * minRes / resolution.x), uv.y * (2.0 * minRes / resolution.y), 0.0, 1.0);
      vStar = aStar;
      vColor = aColor;
      vKind = aKind;
    }`;

    const starFs = `${ver}precision highp float;
    uniform vec2 resolution;
    uniform float progress;
    uniform float schwarzschildRadius;
    uniform vec3 camPos;
    uniform vec3 camFwd;
    uniform vec3 camRight;
    uniform vec3 camUp;
    uniform sampler2D uMask;
    uniform float uUseLens;
    uniform float uUseMask;
    uniform vec2 uMaskRes;
    ${varyIn} vec4 vStar;
    ${varyIn} vec3 vColor;
    ${varyIn} float vKind;
    ${fragOut}
    ${lensLib}
    vec2 lensPull(vec2 uv, vec2 lp, float te2) {
      if (te2 <= 0.0) return vec2(0.0);
      vec2 d = uv - lp;
      float b2 = dot(d, d);
      return d * (te2 / max(b2, te2 * 0.08));
    }
    void main() {
      float minRes = min(resolution.x, resolution.y);
      vec2 uv = (gl_FragCoord.xy - 0.5 * resolution.xy) / minRes;
      float px = 1.0 / minRes;
      vec3 ro = camPos;
      float radius = schwarzschildRadius * progress;
      float holeZ = dot(-ro, camFwd);
      vec2 holeP = holePos(ro);
      float holeMask = uUseMask > 0.5 ? ${tex}(uMask, gl_FragCoord.xy / uMaskRes).${maskCh} : 0.0;
      float z = vStar.z;
      if (z <= 1e-4 || (uUseMask > 0.5 && holeZ > 1e-4 && z > holeZ && holeMask > 0.4)) {
        ${writeColor} = vec4(0.0);
        return;
      }
      vec2 sp = vStar.xy;
      float ang = vStar.w;
      float holeTe2 = uUseLens > 0.5 ? holeTe2Of(lensMassOf(radius), holeZ, z) : 0.0;
      if (holeTe2 < px * px) holeTe2 = 0.0;
      vec2 src = uv;
      if (holeTe2 > 0.0) src -= lensPull(uv, holeP, holeTe2);
      float d = length(src - sp);
      float glow;
      if (vKind > 1.5) {
        float fall = exp(-dot(src - sp, src - sp) / max(ang * ang, 1e-8));
        glow = fall * (vKind - 2.0) * progress;
      } else {
        float coreR = max(ang, px * 1.15);
        float glowR = max(ang * 5.5, px * 3.4);
        float core = smoothstep(coreR, coreR * 0.42, d);
        float near = clamp(ang / (px * 10.0), 0.0, 1.0);
        float halo = exp(-d / max(glowR, 1e-5)) * mix(0.16, 0.52, near);
        glow = (core + halo) * progress;
      }
      ${writeColor} = vec4(vColor * glow, 1.0);
    }`;

    const lineVs = `${ver}${attr} vec2 aPos;
    ${attr} vec2 aMeta;
    uniform vec2 resolution;
    ${varyOut} vec2 vMeta;
    void main() {
      float minRes = min(resolution.x, resolution.y);
      gl_Position = vec4(aPos.x * (2.0 * minRes / resolution.x), aPos.y * (2.0 * minRes / resolution.y), 0.0, 1.0);
      vMeta = aMeta;
    }`;

    const lineFs = `${ver}precision highp float;
    uniform vec2 resolution;
    uniform float progress;
    uniform vec3 camPos;
    uniform vec3 camFwd;
    uniform sampler2D uMask;
    uniform float uUseMask;
    uniform vec2 uMaskRes;
    ${varyIn} vec2 vMeta;
    ${fragOut}
    void main() {
      float holeZ = dot(-camPos, camFwd);
      float holeMask = uUseMask > 0.5 ? ${tex}(uMask, gl_FragCoord.xy / uMaskRes).${maskCh} : 0.0;
      if (vMeta.x < 0.0 || (uUseMask > 0.5 && holeZ > 1e-4 && vMeta.x > holeZ && holeMask > 0.4)) {
        ${writeColor} = vec4(0.0);
        return;
      }
      float glow = (1.0 - smoothstep(0.35, 0.85, abs(vMeta.y))) * 0.16 * progress;
      ${writeColor} = vec4(vec3(glow), 1.0);
    }`;

    const dustVs = `${ver}${attr} vec2 aCorner;
    ${attr} vec4 aDust;
    ${attr} vec3 aColor;
    uniform vec2 resolution;
    ${varyOut} vec4 vDust;
    ${varyOut} vec3 vColor;
    void main() {
      float minRes = min(resolution.x, resolution.y);
      float px = 1.0 / minRes;
      float ang = aDust.w;
      float pad = max(ang * 2.2, px * 3.0);
      vec2 uv = aDust.xy + aCorner * pad;
      gl_Position = vec4(uv.x * (2.0 * minRes / resolution.x), uv.y * (2.0 * minRes / resolution.y), 0.0, 1.0);
      vDust = aDust;
      vColor = aColor;
    }`;

    const dustFs = `${ver}precision highp float;
    uniform vec2 resolution;
    uniform float progress;
    uniform vec3 camPos;
    uniform vec3 camFwd;
    uniform sampler2D uMask;
    uniform float uUseMask;
    uniform vec2 uMaskRes;
    ${varyIn} vec4 vDust;
    ${varyIn} vec3 vColor;
    ${fragOut}
    void main() {
      float minRes = min(resolution.x, resolution.y);
      vec2 uv = (gl_FragCoord.xy - 0.5 * resolution.xy) / minRes;
      float holeZ = dot(-camPos, camFwd);
      float holeMask = uUseMask > 0.5 ? ${tex}(uMask, gl_FragCoord.xy / uMaskRes).${maskCh} : 0.0;
      float z = vDust.z;
      if (z <= 1e-4 || (uUseMask > 0.5 && holeZ > 1e-4 && z > holeZ && holeMask > 0.75)) {
        ${writeColor} = vec4(0.0);
        return;
      }
      vec2 d = uv - vDust.xy;
      float fall = exp(-dot(d, d) / max(vDust.w * vDust.w, 1e-8));
      ${writeColor} = vec4(vColor * (fall * progress), 1.0);
    }`;

    const colorBlitFs = `${ver}precision highp float;
    uniform sampler2D uTex;
    ${varyIn} vec2 vTexCoord;
    ${fragOut}
    void main() { ${writeColor} = ${tex}(uTex, vTexCoord); }`;

    const colorLensFs = `${ver}precision highp float;
    uniform sampler2D uField;
    uniform vec2 resolution;
    uniform float progress;
    uniform float schwarzschildRadius;
    uniform vec3 camPos;
    uniform vec3 camFwd;
    uniform vec3 camRight;
    uniform vec3 camUp;
    ${varyIn} vec2 vTexCoord;
    ${fragOut}
    ${lensLib}
    vec2 lensPull(vec2 uv, vec2 lp, float te2) {
      if (te2 <= 0.0) return vec2(0.0);
      vec2 d = uv - lp;
      float b2 = dot(d, d);
      return d * (te2 / max(b2, te2 * 0.08));
    }
    void main() {
      float minRes = min(resolution.x, resolution.y);
      vec2 uv = (gl_FragCoord.xy - 0.5 * resolution.xy) / minRes;
      vec3 ro = camPos;
      float holeZ = dot(-ro, camFwd);
      float rs = schwarzschildRadius * progress;
      float te2 = holeTe2Of(lensMassOf(rs), holeZ, 1.0e6);
      vec2 src = uv;
      if (te2 > 0.0) src -= lensPull(uv, holePos(ro), te2);
      vec2 srcTex = clamp(src * minRes / resolution + 0.5, 0.0, 1.0);
      ${writeColor} = ${tex}(uField, srcTex);
    }`;

    const holeDiscVs = `${ver}${attr} vec2 aCorner;
    uniform vec2 resolution;
    uniform float progress;
    uniform float schwarzschildRadius;
    uniform vec3 camPos;
    uniform vec3 camFwd;
    uniform vec3 camRight;
    uniform vec3 camUp;
    ${varyOut} vec2 vLocal;
    ${lensLib}
    void main() {
      float minRes = min(resolution.x, resolution.y);
      float px = 1.0 / minRes;
      vec3 ro = camPos;
      float holeZ = dot(-ro, camFwd);
      if (holeZ <= ${LENS_Z_MIN.toFixed(2)}) {
        gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
        vLocal = vec2(2.0);
        return;
      }
      vec2 holeP = holePos(ro);
      float ang = schwarzschildRadius * progress / holeZ;
      float pad = max(ang, px * 2.0);
      vec2 uv = holeP + aCorner * pad;
      gl_Position = vec4(uv.x * (2.0 * minRes / resolution.x), uv.y * (2.0 * minRes / resolution.y), 0.0, 1.0);
      vLocal = aCorner;
    }`;

    const holeDiscFs = `${ver}precision highp float;
    ${varyIn} vec2 vLocal;
    ${fragOut}
    void main() {
      float d = length(vLocal);
      float a = 1.0 - smoothstep(0.94, 1.0, d);
      if (a <= 0.001) discard;
      ${writeColor} = vec4(a);
    }`;

    const lensFs = `${ver}precision highp float;
    uniform sampler2D uField;
    uniform vec2 resolution;
    uniform float progress;
    uniform float schwarzschildRadius;
    uniform vec3 camPos;
    uniform vec3 camFwd;
    uniform vec3 camRight;
    uniform vec3 camUp;
    ${varyIn} vec2 vTexCoord;
    ${fragOut}
    ${lensLib}
    vec2 lensPull(vec2 uv, vec2 lp, float te2) {
      if (te2 <= 0.0) return vec2(0.0);
      vec2 d = uv - lp;
      float b2 = dot(d, d);
      return d * (te2 / max(b2, te2 * 0.08));
    }
    void main() {
      float minRes = min(resolution.x, resolution.y);
      vec2 uv = (gl_FragCoord.xy - 0.5 * resolution.xy) / minRes;
      vec3 ro = camPos;
      float holeZ = dot(-ro, camFwd);
      float rs = schwarzschildRadius * progress;
      float te2 = holeTe2Of(lensMassOf(rs), holeZ, 1.0e6);
      vec2 src = uv;
      if (te2 > 0.0) src -= lensPull(uv, holePos(ro), te2);
      vec2 srcTex = clamp(src * minRes / resolution + 0.5, 0.0, 1.0);
      float glow = ${tex}(uField, srcTex).r;
      ${writeColor} = vec4(vec3(glow), 1.0);
    }`;

    return {
      blitVs,
      blitFs,
      starVs,
      starFs,
      lineVs,
      lineFs,
      holeDiscVs,
      holeDiscFs,
      lensFs,
      dustVs,
      dustFs,
      colorBlitFs,
      colorLensFs,
    };
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

  function createRenderer(canvas, emit) {
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
    const blitProgram = initShaderProgram(gl, shaders.blitVs, shaders.blitFs);
    const starProgram = initShaderProgram(gl, shaders.starVs, shaders.starFs);
    const lineProgram = initShaderProgram(gl, shaders.lineVs, shaders.lineFs);
    const lensProgram = initShaderProgram(gl, shaders.blitVs, shaders.lensFs);
    const holeDiscProgram = initShaderProgram(gl, shaders.holeDiscVs, shaders.holeDiscFs);
    const dustProgram = initShaderProgram(gl, shaders.dustVs, shaders.dustFs);
    const colorBlitProgram = initShaderProgram(gl, shaders.blitVs, shaders.colorBlitFs);
    const colorLensProgram = initShaderProgram(gl, shaders.blitVs, shaders.colorLensFs);
    const buffers = initBuffers(gl);
    loadWasmDither();

    let starVertCount = 0;
    let starFrontCount = 0;
    let lineVertCount = 0;
    let lineFrontCount = 0;
    let starGeom = new Float32Array(10 * 6 * 256);
    let starGeomFront = new Float32Array(10 * 6 * 64);
    let lineGeom = new Float32Array(4 * 6 * 256);
    let lineGeomFront = new Float32Array(4 * 6 * 64);
    let dustGeom = new Float32Array(9 * 6 * 128);
    let dustGeomFront = new Float32Array(9 * 6 * 32);
    let dustVertCount = 0;
    let dustFrontCount = 0;

    let clusterCount = 0;
    let clusterOx = new Float32Array(0);
    let clusterOy = new Float32Array(0);
    let clusterOz = new Float32Array(0);
    let clusterBound = new Float32Array(0);
    let clusterR = new Float32Array(0);
    let pointStart = new Int32Array(0);
    let pointCount = new Int32Array(0);
    let pointX = new Float32Array(0);
    let pointY = new Float32Array(0);
    let pointZ = new Float32Array(0);
    let pointR = new Float32Array(0);
    let pointCr = new Float32Array(0);
    let pointCg = new Float32Array(0);
    let pointCb = new Float32Array(0);
    let worldX = new Float32Array(0);
    let worldY = new Float32Array(0);
    let worldZ = new Float32Array(0);
    let worldVx = new Float32Array(0);
    let worldVy = new Float32Array(0);
    let worldVz = new Float32Array(0);
    let pointMass = new Float32Array(0);
    let starCluster = new Int16Array(0);
    let comX = new Float32Array(0);
    let comY = new Float32Array(0);
    let comZ = new Float32Array(0);
    let comMass = new Float32Array(0);
    let physFx = new Float32Array(0);
    let physFy = new Float32Array(0);
    let physFz = new Float32Array(0);
    let interFx = new Float32Array(0);
    let interFy = new Float32Array(0);
    let interFz = new Float32Array(0);
    let dustWx = new Float32Array(0);
    let dustWy = new Float32Array(0);
    let dustWz = new Float32Array(0);
    let dustVx = new Float32Array(0);
    let dustVy = new Float32Array(0);
    let dustVz = new Float32Array(0);
    let clusterLineMax = new Float32Array(0);
    let hashHead = new Int32Array(1024);
    let hashNext = new Int32Array(0);
    let totalStars = 0;
    let dustCount = new Int32Array(0);
    let dustRadius = new Float32Array(0);
    let dustSize = new Float32Array(0);
    let dustGain = new Float32Array(0);
    let dustH = new Float32Array(0);
    let dustStretch = new Float32Array(0);
    let dustR = new Float32Array(0);
    let dustG = new Float32Array(0);
    let dustB = new Float32Array(0);
    let dustBakeStart = new Int32Array(0);
    let dustBakeAlong = new Float32Array(0);
    let dustBakeOut = new Float32Array(0);
    let dustBakeUp = new Float32Array(0);
    let dustBakePuff = new Float32Array(0);
    let dustBakeGain = new Float32Array(0);
    function setClusters(next) {
      const src = Array.isArray(next) ? next : [];
      clusterCount = src.length;
      clusterOx = new Float32Array(clusterCount);
      clusterOy = new Float32Array(clusterCount);
      clusterOz = new Float32Array(clusterCount);
      clusterBound = new Float32Array(clusterCount);
      clusterR = new Float32Array(clusterCount);
      pointStart = new Int32Array(clusterCount);
      pointCount = new Int32Array(clusterCount);
      dustCount = new Int32Array(clusterCount);
      dustRadius = new Float32Array(clusterCount);
      dustSize = new Float32Array(clusterCount);
      dustGain = new Float32Array(clusterCount);
      dustH = new Float32Array(clusterCount);
      dustStretch = new Float32Array(clusterCount);
      dustR = new Float32Array(clusterCount);
      dustG = new Float32Array(clusterCount);
      dustB = new Float32Array(clusterCount);
      let totalPts = 0;
      let totalDustPuffs = 0;
      for (let i = 0; i < clusterCount; i++) {
        const cluster = src[i] || {};
        const pts = cluster.points;
        totalPts += pts ? pts.length : 0;
        const dust = Array.isArray(cluster.dust) ? cluster.dust[0] : cluster.dust;
        if (dust && (dust.count | 0) > 0) totalDustPuffs += dust.count | 0;
      }
      dustBakeStart = new Int32Array(clusterCount);
      dustBakeAlong = new Float32Array(totalDustPuffs);
      dustBakeOut = new Float32Array(totalDustPuffs);
      dustBakeUp = new Float32Array(totalDustPuffs);
      dustBakePuff = new Float32Array(totalDustPuffs);
      dustBakeGain = new Float32Array(totalDustPuffs);
      let dustBakePtr = 0;
      pointX = new Float32Array(totalPts);
      pointY = new Float32Array(totalPts);
      pointZ = new Float32Array(totalPts);
      pointR = new Float32Array(totalPts);
      pointCr = new Float32Array(totalPts);
      pointCg = new Float32Array(totalPts);
      pointCb = new Float32Array(totalPts);
      worldX = new Float32Array(totalPts);
      worldY = new Float32Array(totalPts);
      worldZ = new Float32Array(totalPts);
      worldVx = new Float32Array(totalPts);
      worldVy = new Float32Array(totalPts);
      worldVz = new Float32Array(totalPts);
      pointMass = new Float32Array(totalPts);
      starCluster = new Int16Array(totalPts);
      comX = new Float32Array(clusterCount);
      comY = new Float32Array(clusterCount);
      comZ = new Float32Array(clusterCount);
      comMass = new Float32Array(clusterCount);
      physFx = new Float32Array(totalPts);
      physFy = new Float32Array(totalPts);
      physFz = new Float32Array(totalPts);
      interFx = new Float32Array(totalPts);
      interFy = new Float32Array(totalPts);
      interFz = new Float32Array(totalPts);
      dustWx = new Float32Array(clusterCount);
      dustWy = new Float32Array(clusterCount);
      dustWz = new Float32Array(clusterCount);
      dustVx = new Float32Array(clusterCount);
      dustVy = new Float32Array(clusterCount);
      dustVz = new Float32Array(clusterCount);
      clusterLineMax = new Float32Array(clusterCount);
      hashNext = new Int32Array(clusterCount);
      totalStars = totalPts;
      const bhMassRef = RS_REF * RS_REF * RS_REF * GRAV_BH_MASS;
      let p = 0;
      for (let c = 0; c < clusterCount; c++) {
        const cluster = src[c] || {};
        const origin = cluster.origin || [0, 0, 0];
        clusterOx[c] = (origin[0] || 0) * GALAXY_OUT_SCALE;
        clusterOy[c] = (origin[1] || 0) * GALAXY_OUT_SCALE;
        clusterOz[c] = (origin[2] || 0) * GALAXY_OUT_SCALE;
        const cr = cluster.radius || STAR_RADIUS;
        clusterR[c] = cr;
        const pts = cluster.points || [];
        pointStart[c] = p;
        pointCount[c] = pts.length;
        let maxExt = 0;
        for (let i = 0; i < pts.length; i++) {
          const pt = pts[i] || [];
          const x = pt[0] || 0;
          const y = pt[1] || 0;
          const z = pt[2] || 0;
          const r = pt[3] || cr;
          pointX[p] = x * GALAXY_OUT_SCALE;
          pointY[p] = y * GALAXY_OUT_SCALE;
          pointZ[p] = z * GALAXY_OUT_SCALE;
          pointR[p] = r;
          pointCr[p] = pt[4] == null ? 1 : pt[4];
          pointCg[p] = pt[5] == null ? 1 : pt[5];
          pointCb[p] = pt[6] == null ? 1 : pt[6];
          const wx = (clusterOx[c] || 0) + x;
          const wy = (clusterOy[c] || 0) + y;
          const wz = (clusterOz[c] || 0) + z;
          worldX[p] = wx;
          worldY[p] = wy;
          worldZ[p] = wz;
          starCluster[p] = c;
          const rm = r * r * r;
          pointMass[p] = rm * GRAV_STAR_MASS;
          const ext = Math.hypot(x, y, z) + r;
          if (ext > maxExt) maxExt = ext;
          p++;
        }
        const start = pointStart[c];
        const end = start + pts.length;
        const muRef = GRAV_G * bhMassRef;
        let maxSpan = 0;
        for (let i = start; i < end; i++) {
          diskOrbitVel(worldX[i], worldY[i], worldZ[i], muRef);
          worldVx[i] = orbitScratch[0];
          worldVy[i] = orbitScratch[1];
          worldVz[i] = orbitScratch[2];
          if (i > start) {
            const span = Math.hypot(
              worldX[i] - worldX[i - 1],
              worldY[i] - worldY[i - 1],
              worldZ[i] - worldZ[i - 1]
            );
            if (span > maxSpan) maxSpan = span;
          }
        }
        clusterLineMax[c] = Math.min(16, maxSpan * 1.2 + 1.6);
        dustWx[c] = clusterOx[c];
        dustWy[c] = clusterOy[c];
        dustWz[c] = clusterOz[c];
        const dust = Array.isArray(cluster.dust) ? cluster.dust[0] : cluster.dust;
        if (dust && (dust.count | 0) > 0) {
          dustCount[c] = dust.count | 0;
          diskOrbitVel(dustWx[c], dustWy[c], dustWz[c], muRef);
          dustVx[c] = orbitScratch[0];
          dustVy[c] = orbitScratch[1];
          dustVz[c] = orbitScratch[2];
          dustRadius[c] = (dust.radius || 5) * GALAXY_OUT_SCALE;
          dustSize[c] = dust.size || 1.6;
          dustGain[c] = Math.max(0, dust.opacity == null ? 0.04 : dust.opacity) * DUST_GAIN_MUL;
          dustH[c] = dust.height == null ? 0.12 : Math.max(0.03, dust.height);
          dustStretch[c] = dust.stretch == null ? 1.4 : Math.max(0.4, dust.stretch);
          dustR[c] = 1;
          dustG[c] = 1;
          dustB[c] = 1;
          const rad = dustRadius[c];
          const size = dustSize[c];
          const height = dustH[c];
          const stretch = dustStretch[c];
          dustBakeStart[c] = dustBakePtr;
          const dn = dustCount[c];
          for (let i = 0; i < dn; i++) {
            const u = dustHash(c, i, 0);
            const v = dustHash(c, i, 1);
            const w = dustHash(c, i, 2);
            dustBakeAlong[dustBakePtr] = (u - 0.5) * 2 * rad * stretch;
            dustBakeOut[dustBakePtr] = (w - 0.5) * 2 * rad * 0.42;
            dustBakeUp[dustBakePtr] = (v - 0.5) * 2 * rad * height;
            dustBakePuff[dustBakePtr] = size * (0.55 + dustHash(c, i, 4) * 0.85);
            dustBakeGain[dustBakePtr] = 0.7 + dustHash(c, i, 5) * 0.3;
            dustBakePtr++;
          }
          const dustExt = rad * Math.max(1, stretch) + size * 1.4;
          if (dustExt > maxExt) maxExt = dustExt;
        }
        clusterBound[c] = maxExt;
      }
      warmSim(SIM_WARMUP_SEC, SIM_WARMUP_SCALE);
    }

    fetch("/js/clusters.json")
      .then((res) => res.json())
      .then((data) => {
        if (!clusterCount) setClusters(data.clusters || []);
      })
      .catch(() => {});

    function viewHalf(renderW, renderH, minRes) {
      return {
        x: renderW / (2 * minRes) + VIEW_FRUSTUM_PAD,
        y: renderH / (2 * minRes) + VIEW_FRUSTUM_PAD,
      };
    }

    function starInView(b, wx, wy, wz, starR, half) {
      const dx = wx - camX;
      const dy = wy - camY;
      const dz = wz - camZ;
      const z = dx * b.fx + dy * b.fy + dz * b.fz;
      if (z <= 0.05 || z > STAR_FAR_Z) return false;
      const sx = (dx * b.rx + dy * b.ry + dz * b.rz) / z;
      const sy = (dx * b.ux + dy * b.uy + dz * b.uz) / z;
      const pad = (starR / z) * STAR_ANGULAR_PAD + STAR_BASE_PAD;
      return Math.abs(sx) <= half.x + pad && Math.abs(sy) <= half.y + pad;
    }

    function clusterInView(b, ox, oy, oz, boundR, half) {
      const dx = ox - camX;
      const dy = oy - camY;
      const dz = oz - camZ;
      const z = dx * b.fx + dy * b.fy + dz * b.fz;
      if (z + boundR < 0.05 || z - boundR > STAR_FAR_Z) return false;
      if (z <= boundR * 2 + 0.05) return true;
      const sx = (dx * b.rx + dy * b.ry + dz * b.rz) / z;
      const sy = (dx * b.ux + dy * b.uy + dz * b.uz) / z;
      const pad = (boundR / z) * STAR_ANGULAR_PAD + STAR_BASE_PAD;
      return Math.abs(sx) <= half.x + pad && Math.abs(sy) <= half.y + pad;
    }

    function smoothstepJS(edge0, edge1, x) {
      const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
      return t * t * (3 - 2 * t);
    }

    function lensFadeJS(holeZ) {
      return 1;
    }

    function einstein2JS(mass, zL, zS) {
      const fade = lensFadeJS(zL);
      if (fade <= 0 || zS <= 0) return 0;
      const zUse = Math.max(zL, LENS_Z_MIN);
      const dls = zS - zUse;
      if (dls <= 0) return 0;
      return (fade * 2 * mass * dls) / Math.max(zUse * zS, 1e-8);
    }

    function lensPullJS(uvx, uvy, lpx, lpy, te2) {
      if (te2 <= 0) return [0, 0];
      const dx = uvx - lpx;
      const dy = uvy - lpy;
      const s = te2 / Math.max(dx * dx + dy * dy, te2 * 0.08);
      return [dx * s, dy * s];
    }

    function growFloat(buf, need) {
      if (buf.length >= need) return buf;
      let n = buf.length || 256;
      while (n < need) n *= 2;
      return new Float32Array(n);
    }

    function emitStarTo(bufName, countName, spx, spy, z, ang, kind, cr, cg, cb) {
      const count = countName === "front" ? starFrontCount : starVertCount;
      const need = count * 10 + 60;
      if (countName === "front") starGeomFront = growFloat(starGeomFront, need);
      else starGeom = growFloat(starGeom, need);
      const geom = countName === "front" ? starGeomFront : starGeom;
      const corners = [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1];
      for (let i = 0; i < 6; i++) {
        const o = count * 10 + i * 10;
        geom[o] = corners[i * 2];
        geom[o + 1] = corners[i * 2 + 1];
        geom[o + 2] = spx;
        geom[o + 3] = spy;
        geom[o + 4] = z;
        geom[o + 5] = ang;
        geom[o + 6] = cr;
        geom[o + 7] = cg;
        geom[o + 8] = cb;
        geom[o + 9] = kind;
      }
      if (countName === "front") starFrontCount += 6;
      else starVertCount += 6;
    }

    function emitStarQuad(spx, spy, z, ang, kind, cr, cg, cb) {
      emitStarTo("starGeom", "back", spx, spy, z, ang, kind, cr, cg, cb);
    }

    function dustHash(c, i, k) {
      let n = Math.imul(c + 1, 374761393) ^ Math.imul(i + 1, 668265263) ^ Math.imul(k + 1, 1442695041);
      n = Math.imul(n ^ (n >>> 15), 2246822519);
      n = Math.imul(n ^ (n >>> 13), 3266489917);
      return ((n ^ (n >>> 16)) >>> 0) / 4294967296;
    }

    function emitDustTo(front, spx, spy, z, ang, cr, cg, cb) {
      const count = front ? dustFrontCount : dustVertCount;
      const need = count * 9 + 54;
      if (front) dustGeomFront = growFloat(dustGeomFront, need);
      else dustGeom = growFloat(dustGeom, need);
      const geom = front ? dustGeomFront : dustGeom;
      const corners = [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1];
      for (let i = 0; i < 6; i++) {
        const o = count * 9 + i * 9;
        geom[o] = corners[i * 2];
        geom[o + 1] = corners[i * 2 + 1];
        geom[o + 2] = spx;
        geom[o + 3] = spy;
        geom[o + 4] = z;
        geom[o + 5] = ang;
        geom[o + 6] = cr;
        geom[o + 7] = cg;
        geom[o + 8] = cb;
      }
      if (front) dustFrontCount += 6;
      else dustVertCount += 6;
    }

    function emitClusterDust(c, ox, oy, oz, b, minRes, half, holeZ) {
      const baseN = dustCount[c];
      if (!baseN) return;
      let n = Math.max(3, (baseN * DUST_COUNT_MUL) | 0);
      if (isMobile()) n = Math.max(6, (n * 0.48) | 0);
      const dustStep = Math.max(1, (baseN / n) | 0);
      const gain = dustGain[c];
      const zCut = Math.max(holeZ, LENS_Z_MIN);
      const tlen = Math.hypot(ox, oz) || 1;
      const rx = ox / tlen;
      const rz = oz / tlen;
      const tx = -rz;
      const tz = rx;
      const bakeStart = dustBakeStart[c];
      for (let i = 0; i < baseN; i += dustStep) {
        const bi = bakeStart + i;
        const along = dustBakeAlong[bi];
        const out = dustBakeOut[bi];
        const up = dustBakeUp[bi];
        const wx = ox + tx * along + rx * out;
        const wy = oy + up;
        const wz = oz + tz * along + rz * out;
        const puff = dustBakePuff[bi];
        if (!starInView(b, wx, wy, wz, puff, half)) continue;
        const dx = wx - camX;
        const dy = wy - camY;
        const dz = wz - camZ;
        const z = dx * b.fx + dy * b.fy + dz * b.fz;
        const spx = (dx * b.rx + dy * b.ry + dz * b.rz) / z;
        const spy = (dx * b.ux + dy * b.uy + dz * b.uz) / z;
        const ang = puff / Math.max(z, puff * 0.35);
        const puffGain = gain * dustBakeGain[bi];
        emitDustTo(z <= zCut, spx, spy, z, ang, puffGain, puffGain, puffGain);
      }
    }

    function emitLineQuad(ax, ay, bx, by, z, minRes, front) {
      const dx = bx - ax;
      const dy = by - ay;
      const len = Math.hypot(dx, dy) || 1e-5;
      const hw = 0.85 / minRes;
      const nx = (-dy / len) * hw;
      const ny = (dx / len) * hw;
      const x0 = ax - nx;
      const y0 = ay - ny;
      const x1 = ax + nx;
      const y1 = ay + ny;
      const x2 = bx - nx;
      const y2 = by - ny;
      const x3 = bx + nx;
      const y3 = by + ny;
      const count = front ? lineFrontCount : lineVertCount;
      const need = count * 4 + 24;
      if (front) lineGeomFront = growFloat(lineGeomFront, need);
      else lineGeom = growFloat(lineGeom, need);
      const geom = front ? lineGeomFront : lineGeom;
      const verts = [x0, y0, -0.85, x1, y1, 0.85, x2, y2, -0.85, x2, y2, -0.85, x1, y1, 0.85, x3, y3, 0.85];
      for (let i = 0; i < 6; i++) {
        const o = count * 4 + i * 4;
        geom[o] = verts[i * 3];
        geom[o + 1] = verts[i * 3 + 1];
        geom[o + 2] = z;
        geom[o + 3] = verts[i * 3 + 2];
      }
      if (front) lineFrontCount += 6;
      else lineVertCount += 6;
    }

    const orbitScratch = new Float32Array(3);

    function diskOrbitVel(px, py, pz, mu) {
      const sr = Math.hypot(px, py, pz) || 1;
      const speed = Math.sqrt(Math.max(1e-8, mu / sr));
      let tx = -pz;
      let tz = px;
      const tLen = Math.hypot(tx, tz) || 1;
      tx /= tLen;
      tz /= tLen;
      orbitScratch[0] = tx * speed;
      orbitScratch[1] = 0;
      orbitScratch[2] = tz * speed;
    }

    function bhMu() {
      const rs = schwarzschildRadius;
      return GRAV_G * rs * rs * rs * GRAV_BH_MASS;
    }

    function stumpff(psi, out) {
      if (psi > 1e-8) {
        const s = Math.sqrt(psi);
        out[0] = (1 - Math.cos(s)) / psi;
        out[1] = (s - Math.sin(s)) / (s * psi);
        return;
      }
      if (psi < -1e-8) {
        const s = Math.sqrt(-psi);
        out[0] = (Math.cosh(s) - 1) / -psi;
        out[1] = (Math.sinh(s) - s) / (s * -psi);
        return;
      }
      out[0] = 0.5 - psi / 24;
      out[1] = 1 / 6 - psi / 120;
    }

    const keplerStump = new Float64Array(2);
    const keplerOut = new Float64Array(6);

    function keplerStep(px, py, pz, vx, vy, vz, dt, mu, out) {
      const r0 = Math.hypot(px, py, pz);
      if (r0 < 1e-6 || dt === 0) {
        out[0] = px;
        out[1] = py;
        out[2] = pz;
        out[3] = vx;
        out[4] = vy;
        out[5] = vz;
        return true;
      }

      const v2 = vx * vx + vy * vy + vz * vz;
      const rdotv = px * vx + py * vy + pz * vz;
      const alpha = 2 / r0 - v2 / mu;
      if (alpha > 1e-12) {
        const a = 1 / alpha;
        const period = Math.PI * 2 * Math.sqrt((a * a * a) / mu);
        dt -= period * Math.round(dt / period);
        if (dt === 0) {
          out[0] = px;
          out[1] = py;
          out[2] = pz;
          out[3] = vx;
          out[4] = vy;
          out[5] = vz;
          return true;
        }
      }

      const sqrtMu = Math.sqrt(mu);
      let chi;
      if (alpha > 1e-12) {
        chi = sqrtMu * dt * alpha;
      } else if (alpha < -1e-12) {
        const sign = dt >= 0 ? 1 : -1;
        const den = rdotv * alpha + sign * Math.sqrt(-mu * alpha) * (1 - r0 * alpha);
        chi = sign * Math.sqrt(-1 / alpha) * Math.log(Math.max(1e-12, (-2 * mu * alpha * dt) / den));
        if (!Number.isFinite(chi)) chi = sqrtMu * dt * Math.abs(alpha);
      } else {
        chi = (sqrtMu * dt) / r0;
      }

      const sqrtMuDt = sqrtMu * dt;
      const cs = keplerStump;
      let converged = false;
      for (let k = 0; k < 14; k++) {
        stumpff(alpha * chi * chi, cs);
        const C = cs[0];
        const S = cs[1];
        const chi2 = chi * chi;
        const chi3 = chi2 * chi;
        const F = (rdotv / sqrtMu) * chi2 * C + (1 - alpha * r0) * chi3 * S + r0 * chi - sqrtMuDt;
        const dF = (rdotv / sqrtMu) * chi * (1 - alpha * chi2 * S) + (1 - alpha * r0) * chi2 * C + r0;
        if (Math.abs(dF) < 1e-18) break;
        const dchi = F / dF;
        chi -= dchi;
        if (Math.abs(dchi) < 1e-12 * (1 + Math.abs(chi))) {
          converged = true;
          break;
        }
      }

      stumpff(alpha * chi * chi, cs);
      const C = cs[0];
      const S = cs[1];
      const chi2 = chi * chi;
      const chi3 = chi2 * chi;
      const f = 1 - (chi2 / r0) * C;
      const g = dt - (chi3 / sqrtMu) * S;
      const nx = f * px + g * vx;
      const ny = f * py + g * vy;
      const nz = f * pz + g * vz;
      const r = Math.hypot(nx, ny, nz);
      if (!converged || !(r > 1e-8) || !Number.isFinite(r)) return false;

      const fdot = (sqrtMu / (r * r0)) * (alpha * chi3 * S - chi);
      const gdot = 1 - (chi2 / r) * C;
      const nvx = fdot * px + gdot * vx;
      const nvy = fdot * py + gdot * vy;
      const nvz = fdot * pz + gdot * vz;
      if (!Number.isFinite(nvx + nvy + nvz)) return false;

      out[0] = nx;
      out[1] = ny;
      out[2] = nz;
      out[3] = nvx;
      out[4] = nvy;
      out[5] = nvz;
      return true;
    }

    function newtonBhStep(i, dt, mu, wx, wy, wz, vx, vy, vz) {
      const px = wx[i];
      const py = wy[i];
      const pz = wz[i];
      const r2 = px * px + py * py + pz * pz + 1e-8;
      const invR3 = mu / (r2 * Math.sqrt(r2));
      vx[i] -= px * invR3 * dt;
      vy[i] -= py * invR3 * dt;
      vz[i] -= pz * invR3 * dt;
      wx[i] += vx[i] * dt;
      wy[i] += vy[i] * dt;
      wz[i] += vz[i] * dt;
    }

    function keplerBody(px, py, pz, vx, vy, vz, dt, mu) {
      const out = keplerOut;
      if (keplerStep(px, py, pz, vx, vy, vz, dt, mu, out)) {
        return out;
      }
      const r2 = px * px + py * py + pz * pz + 1e-8;
      const invR3 = mu / (r2 * Math.sqrt(r2));
      const nvx = vx - px * invR3 * dt;
      const nvy = vy - py * invR3 * dt;
      const nvz = vz - pz * invR3 * dt;
      out[0] = px + nvx * dt;
      out[1] = py + nvy * dt;
      out[2] = pz + nvz * dt;
      out[3] = nvx;
      out[4] = nvy;
      out[5] = nvz;
      return out;
    }

    function refreshCom(n, cc, wx, wy, wz, mass, cx, cy, cz, cm) {
      cx.fill(0);
      cy.fill(0);
      cz.fill(0);
      cm.fill(0);
      for (let i = 0; i < n; i++) {
        const m = mass[i];
        const c = starCluster[i];
        cx[c] += wx[i] * m;
        cy[c] += wy[i] * m;
        cz[c] += wz[i] * m;
        cm[c] += m;
      }
      for (let c = 0; c < cc; c++) {
        const m = cm[c];
        if (m <= 0) continue;
        cx[c] /= m;
        cy[c] /= m;
        cz[c] /= m;
      }
    }

    function computeInterCluster(n, cc, wx, wy, wz, mass, g, eps2, cx, cy, cz, cm) {
      interFx.fill(0);
      interFy.fill(0);
      interFz.fill(0);
      if (cc < 2) return;
      refreshCom(n, cc, wx, wy, wz, mass, cx, cy, cz, cm);
      const near2 = GRAV_NEAR2;
      const invCell = 1 / 40;
      hashHead.fill(-1);
      for (let c = 0; c < cc; c++) {
        if (pointCount[c] < 1) continue;
        const ix = (Math.floor(cx[c] * invCell) + 512) & 31;
        const iz = (Math.floor(cz[c] * invCell) + 512) & 31;
        const key = ix | (iz << 5);
        hashNext[c] = hashHead[key];
        hashHead[key] = c;
      }
      for (let c = 0; c < cc; c++) {
        const c1s = pointStart[c];
        const c1e = c1s + pointCount[c];
        if (c1e <= c1s) continue;
        const c1x = cx[c];
        const c1y = cy[c];
        const c1z = cz[c];
        const m1 = cm[c];
        if (m1 <= 0) continue;
        const ix0 = (Math.floor(c1x * invCell) + 512) & 31;
        const iz0 = (Math.floor(c1z * invCell) + 512) & 31;
        for (let oz = -1; oz <= 1; oz++) {
          for (let ox = -1; ox <= 1; ox++) {
            const key = ((ix0 + ox) & 31) | (((iz0 + oz) & 31) << 5);
            let d = hashHead[key];
            while (d >= 0) {
              if (d > c) {
                const c2s = pointStart[d];
                const c2e = c2s + pointCount[d];
                if (c2e > c2s) {
                  const m2 = cm[d];
                  if (m2 <= 0) {
                    d = hashNext[d];
                    continue;
                  }
                  const dx0 = cx[d] - c1x;
                  const dy0 = cy[d] - c1y;
                  const dz0 = cz[d] - c1z;
                  if (dx0 * dx0 + dy0 * dy0 + dz0 * dz0 <= near2) {
                    const r2 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 + eps2;
                    const invR3 = g / (r2 * Math.sqrt(r2));
                    const fx = dx0 * invR3;
                    const fy = dy0 * invR3;
                    const fz = dz0 * invR3;
                    for (let i = c1s; i < c1e; i++) {
                      interFx[i] += fx * m2;
                      interFy[i] += fy * m2;
                      interFz[i] += fz * m2;
                    }
                    for (let j = c2s; j < c2e; j++) {
                      interFx[j] -= fx * m1;
                      interFy[j] -= fy * m1;
                      interFz[j] -= fz * m1;
                    }
                  }
                }
              }
              d = hashNext[d];
            }
          }
        }
      }
    }

    function kickStars(h, n, cc, wx, wy, wz, vx, vy, vz, mass, g, eps2, ax, ay, az) {
      ax.fill(0);
      ay.fill(0);
      az.fill(0);

      for (let c = 0; c < cc; c++) {
        const start = pointStart[c];
        const end = start + pointCount[c];
        for (let i = start; i < end; i++) {
          const ix = wx[i];
          const iy = wy[i];
          const iz = wz[i];
          const mi = mass[i];
          for (let j = i + 1; j < end; j++) {
            const dx = wx[j] - ix;
            const dy = wy[j] - iy;
            const dz = wz[j] - iz;
            const r2 = dx * dx + dy * dy + dz * dz + eps2;
            const invR3 = g / (r2 * Math.sqrt(r2));
            const mj = mass[j];
            ax[i] += dx * invR3 * mj;
            ay[i] += dy * invR3 * mj;
            az[i] += dz * invR3 * mj;
            ax[j] -= dx * invR3 * mi;
            ay[j] -= dy * invR3 * mi;
            az[j] -= dz * invR3 * mi;
          }
        }
      }

      for (let i = 0; i < n; i++) {
        vx[i] += (ax[i] + interFx[i]) * h;
        vy[i] += (ay[i] + interFy[i]) * h;
        vz[i] += (az[i] + interFz[i]) * h;
      }
    }

    function driftStars(dt, n, cc, mu, wx, wy, wz, vx, vy, vz, dustStride) {
      const out = keplerOut;
      for (let i = 0; i < n; i++) {
        if (keplerStep(wx[i], wy[i], wz[i], vx[i], vy[i], vz[i], dt, mu, out)) {
          wx[i] = out[0];
          wy[i] = out[1];
          wz[i] = out[2];
          vx[i] = out[3];
          vy[i] = out[4];
          vz[i] = out[5];
        } else {
          newtonBhStep(i, dt, mu, wx, wy, wz, vx, vy, vz);
        }
      }
      if (dustStride > 1) {
        const dustDt = dt * dustStride;
        for (let c = 0; c < cc; c++) {
          if (!dustCount[c] || (c + simFrame) % dustStride !== 0) continue;
          const next = keplerBody(dustWx[c], dustWy[c], dustWz[c], dustVx[c], dustVy[c], dustVz[c], dustDt, mu);
          dustWx[c] = next[0];
          dustWy[c] = next[1];
          dustWz[c] = next[2];
          dustVx[c] = next[3];
          dustVy[c] = next[4];
          dustVz[c] = next[5];
        }
        return;
      }
      for (let c = 0; c < cc; c++) {
        if (!dustCount[c]) continue;
        const next = keplerBody(dustWx[c], dustWy[c], dustWz[c], dustVx[c], dustVy[c], dustVz[c], dt, mu);
        dustWx[c] = next[0];
        dustWy[c] = next[1];
        dustWz[c] = next[2];
        dustVx[c] = next[3];
        dustVy[c] = next[4];
        dustVz[c] = next[5];
      }
    }

    function updateGravity(frameMs, timeMul) {
      const n = totalStars;
      const cc = clusterCount;
      if (!cc) return;

      const wx = worldX;
      const wy = worldY;
      const wz = worldZ;
      const vx = worldVx;
      const vy = worldVy;
      const vz = worldVz;
      const mass = pointMass;
      const cx = comX;
      const cy = comY;
      const cz = comZ;
      const cm = comMass;
      const ax = physFx;
      const ay = physFy;
      const az = physFz;
      const mu = bhMu();
      const orbitDt = frameMs * 0.001 * GRAV_STEP * timeMul;
      const useKicks = timeMul <= GRAV_KICK_MUL;
      const camDist = Math.hypot(camX, camY, camZ);
      const kickCap =
        camDist > GRAV_CAM_FAR ? GRAV_KICKS_FAR : camDist > GRAV_CAM_MID ? GRAV_KICKS_MID : GRAV_MAX_KICKS;
      const dustStride = camDist > GRAV_CAM_FAR ? 2 : 1;
      const refreshBounds = (simFrame++ & 3) === 0;

      if (n) computeInterCluster(n, cc, wx, wy, wz, mass, GRAV_G, GRAV_SOFT2, cx, cy, cz, cm);

      if (n && useKicks) {
        const nKick = Math.min(kickCap, Math.max(1, Math.ceil(orbitDt / GRAV_KICK_DT)));
        const h = orbitDt / nKick;
        kickStars(h * 0.5, n, cc, wx, wy, wz, vx, vy, vz, mass, GRAV_G, GRAV_SOFT2, ax, ay, az);
        for (let s = 0; s < nKick; s++) {
          driftStars(h, n, cc, mu, wx, wy, wz, vx, vy, vz, dustStride);
          kickStars(
            s + 1 === nKick ? h * 0.5 : h,
            n,
            cc,
            wx,
            wy,
            wz,
            vx,
            vy,
            vz,
            mass,
            GRAV_G,
            GRAV_SOFT2,
            ax,
            ay,
            az
          );
        }
      } else {
        driftStars(orbitDt, n, cc, mu, wx, wy, wz, vx, vy, vz, dustStride);
        if (n) {
          for (let i = 0; i < n; i++) {
            vx[i] += interFx[i] * orbitDt;
            vy[i] += interFy[i] * orbitDt;
            vz[i] += interFz[i] * orbitDt;
          }
        }
      }

      refreshCom(n, cc, wx, wy, wz, mass, cx, cy, cz, cm);
      for (let c = 0; c < cc; c++) {
        const m = cm[c];
        if (m <= 0) {
          clusterOx[c] = dustWx[c];
          clusterOy[c] = dustWy[c];
          clusterOz[c] = dustWz[c];
          continue;
        }
        clusterOx[c] = cx[c];
        clusterOy[c] = cy[c];
        clusterOz[c] = cz[c];
        if (!refreshBounds) continue;
        const start = pointStart[c];
        const end = start + pointCount[c];
        let maxExt = 0;
        for (let i = start; i < end; i++) {
          pointX[i] = wx[i] - cx[c];
          pointY[i] = wy[i] - cy[c];
          pointZ[i] = wz[i] - cz[c];
          const ext = Math.hypot(pointX[i], pointY[i], pointZ[i]);
          if (ext > maxExt) maxExt = ext;
        }
        clusterBound[c] = maxExt;
      }
    }

    function warmSim(seconds, timeMul) {
      if (!clusterCount) return;
      const frameMs = 250;
      const steps = Math.ceil((seconds * 1000) / frameMs);
      for (let i = 0; i < steps; i++) updateGravity(frameMs, timeMul);
    }

    function packScene(b, renderW, renderH, radius) {
      const minRes = Math.min(renderW, renderH);
      const half = viewHalf(renderW, renderH, minRes);
      const holeZ = -camX * b.fx - camY * b.fy - camZ * b.fz;
      starVertCount = 0;
      starFrontCount = 0;
      lineVertCount = 0;
      lineFrontCount = 0;
      dustVertCount = 0;
      dustFrontCount = 0;
      for (let c = 0; c < clusterCount; c++) {
        const npts = pointCount[c];
        const hasDust = dustCount[c] > 0;
        if (!npts && !hasDust) continue;
        const starVisible =
          npts > 0 &&
          clusterInView(b, clusterOx[c], clusterOy[c], clusterOz[c], clusterBound[c], half);
        const dustVisible =
          hasDust &&
          clusterInView(
            b,
            dustWx[c],
            dustWy[c],
            dustWz[c],
            dustRadius[c] * Math.max(1, dustStretch[c]) + dustSize[c] * 1.4,
            half
          );
        if (!starVisible && !dustVisible) continue;
        if (dustVisible) emitClusterDust(c, dustWx[c], dustWy[c], dustWz[c], b, minRes, half, holeZ);
        if (!starVisible) continue;
        const start = pointStart[c];
        const maxSpan = clusterLineMax[c];
        let prevPi = -1;
        let prevAx = 0;
        let prevAy = 0;
        let prevWx = 0;
        let prevWy = 0;
        let prevWz = 0;
        let prevZ = 0;
        let prevBehind = false;
        for (let i = 0; i < npts; i++) {
          const pi = start + i;
          const wx = worldX[pi];
          const wy = worldY[pi];
          const wz = worldZ[pi];
          const sr = pointR[pi];
          const displaySr = sr * (schwarzschildRadius / RS_REF);
          const scr = pointCr[pi];
          const scg = pointCg[pi];
          const scb = pointCb[pi];
          if (!starInView(b, wx, wy, wz, displaySr, half)) {
            prevPi = -1;
            continue;
          }
          const dx = wx - camX;
          const dy = wy - camY;
          const dz = wz - camZ;
          const z = dx * b.fx + dy * b.fy + dz * b.fz;
          const spx = (dx * b.rx + dy * b.ry + dz * b.rz) / z;
          const spy = (dx * b.ux + dy * b.uy + dz * b.uz) / z;
          const ang = displaySr / Math.max(z, displaySr * 0.35);
          const behind = z > Math.max(holeZ, LENS_Z_MIN);
          if (!behind) emitStarTo("starGeomFront", "front", spx, spy, z, ang, 0, scr, scg, scb);
          else emitStarQuad(spx, spy, z, ang, 0, scr, scg, scb);
          if (prevPi >= 0 && pi === prevPi + 1 && behind === prevBehind) {
            const span = Math.hypot(wx - prevWx, wy - prevWy, wz - prevWz);
            if (span <= maxSpan) {
              emitLineQuad(prevAx, prevAy, spx, spy, Math.min(prevZ, z), minRes, !behind);
            }
          }
          prevPi = pi;
          prevAx = spx;
          prevAy = spy;
          prevWx = wx;
          prevWy = wy;
          prevWz = wz;
          prevZ = z;
          prevBehind = behind;
        }
      }
    }

    const readbackCaps = webgl2 ? probeSingleChannelTarget(gl, true) : { single: false, pbo: false };
    const singleTarget = true;
    const usePbo = webgl2 && readbackCaps.pbo;
    const sceneInternal = webgl2 ? gl.RG8 : gl.RGBA;
    const sceneFormat = webgl2 ? gl.RG : gl.RGBA;
    const displayInternal = webgl2 ? gl.R8 : gl.LUMINANCE;
    const displayFormat = webgl2 ? gl.RED : gl.LUMINANCE;
    const sceneBytes = 1;

    const starAttribs = {
      corner: gl.getAttribLocation(starProgram, "aCorner"),
      star: gl.getAttribLocation(starProgram, "aStar"),
      color: gl.getAttribLocation(starProgram, "aColor"),
      kind: gl.getAttribLocation(starProgram, "aKind"),
    };
    const starUniforms = {
      resolution: gl.getUniformLocation(starProgram, "resolution"),
      progress: gl.getUniformLocation(starProgram, "progress"),
      schwarzschildRadius: gl.getUniformLocation(starProgram, "schwarzschildRadius"),
      camPos: gl.getUniformLocation(starProgram, "camPos"),
      camFwd: gl.getUniformLocation(starProgram, "camFwd"),
      camRight: gl.getUniformLocation(starProgram, "camRight"),
      camUp: gl.getUniformLocation(starProgram, "camUp"),
      mask: gl.getUniformLocation(starProgram, "uMask"),
      useLens: gl.getUniformLocation(starProgram, "uUseLens"),
      useMask: gl.getUniformLocation(starProgram, "uUseMask"),
      maskRes: gl.getUniformLocation(starProgram, "uMaskRes"),
    };
    const lineAttribs = {
      pos: gl.getAttribLocation(lineProgram, "aPos"),
      meta: gl.getAttribLocation(lineProgram, "aMeta"),
    };
    const lineUniforms = {
      resolution: gl.getUniformLocation(lineProgram, "resolution"),
      progress: gl.getUniformLocation(lineProgram, "progress"),
      camPos: gl.getUniformLocation(lineProgram, "camPos"),
      camFwd: gl.getUniformLocation(lineProgram, "camFwd"),
      mask: gl.getUniformLocation(lineProgram, "uMask"),
      useMask: gl.getUniformLocation(lineProgram, "uUseMask"),
      maskRes: gl.getUniformLocation(lineProgram, "uMaskRes"),
    };
    const lensUniforms = {
      field: gl.getUniformLocation(lensProgram, "uField"),
      resolution: gl.getUniformLocation(lensProgram, "resolution"),
      progress: gl.getUniformLocation(lensProgram, "progress"),
      schwarzschildRadius: gl.getUniformLocation(lensProgram, "schwarzschildRadius"),
      camPos: gl.getUniformLocation(lensProgram, "camPos"),
      camFwd: gl.getUniformLocation(lensProgram, "camFwd"),
      camRight: gl.getUniformLocation(lensProgram, "camRight"),
      camUp: gl.getUniformLocation(lensProgram, "camUp"),
    };
    const holeDiscAttribs = {
      corner: gl.getAttribLocation(holeDiscProgram, "aCorner"),
    };
    const holeDiscUniforms = {
      resolution: gl.getUniformLocation(holeDiscProgram, "resolution"),
      progress: gl.getUniformLocation(holeDiscProgram, "progress"),
      schwarzschildRadius: gl.getUniformLocation(holeDiscProgram, "schwarzschildRadius"),
      camPos: gl.getUniformLocation(holeDiscProgram, "camPos"),
      camFwd: gl.getUniformLocation(holeDiscProgram, "camFwd"),
      camRight: gl.getUniformLocation(holeDiscProgram, "camRight"),
      camUp: gl.getUniformLocation(holeDiscProgram, "camUp"),
    };
    const dustAttribs = {
      corner: gl.getAttribLocation(dustProgram, "aCorner"),
      dust: gl.getAttribLocation(dustProgram, "aDust"),
      color: gl.getAttribLocation(dustProgram, "aColor"),
    };
    const dustUniforms = {
      resolution: gl.getUniformLocation(dustProgram, "resolution"),
      progress: gl.getUniformLocation(dustProgram, "progress"),
      camPos: gl.getUniformLocation(dustProgram, "camPos"),
      camFwd: gl.getUniformLocation(dustProgram, "camFwd"),
      mask: gl.getUniformLocation(dustProgram, "uMask"),
      useMask: gl.getUniformLocation(dustProgram, "uUseMask"),
      maskRes: gl.getUniformLocation(dustProgram, "uMaskRes"),
    };
    const colorLensUniforms = {
      field: gl.getUniformLocation(colorLensProgram, "uField"),
      resolution: gl.getUniformLocation(colorLensProgram, "resolution"),
      progress: gl.getUniformLocation(colorLensProgram, "progress"),
      schwarzschildRadius: gl.getUniformLocation(colorLensProgram, "schwarzschildRadius"),
      camPos: gl.getUniformLocation(colorLensProgram, "camPos"),
      camFwd: gl.getUniformLocation(colorLensProgram, "camFwd"),
      camRight: gl.getUniformLocation(colorLensProgram, "camRight"),
      camUp: gl.getUniformLocation(colorLensProgram, "camUp"),
    };
    const colorBlitAttrib = gl.getAttribLocation(colorBlitProgram, "aVertexPosition");
    const colorLensAttrib = gl.getAttribLocation(colorLensProgram, "aVertexPosition");
    const starBuffer = gl.createBuffer();
    const lineBuffer = gl.createBuffer();
    const dustBuffer = gl.createBuffer();
    const holeDiscBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, holeDiscBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW
    );

    const blitAttrib = gl.getAttribLocation(blitProgram, "aVertexPosition");
    const lensAttrib = gl.getAttribLocation(lensProgram, "aVertexPosition");
    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.position);
    gl.vertexAttribPointer(blitAttrib, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(blitAttrib);
    if (lensAttrib >= 0 && lensAttrib !== blitAttrib) {
      gl.vertexAttribPointer(lensAttrib, 2, gl.FLOAT, false, 0, 0);
      gl.enableVertexAttribArray(lensAttrib);
    }

    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.pixelStorei(gl.PACK_ALIGNMENT, 4);
    gl.useProgram(blitProgram);
    gl.uniform1i(gl.getUniformLocation(blitProgram, "uTex"), 0);
    gl.useProgram(colorBlitProgram);
    gl.uniform1i(gl.getUniformLocation(colorBlitProgram, "uTex"), 0);
    gl.activeTexture(gl.TEXTURE0);

    let compositeFb = null;
    let fieldFb = null;
    let compositeTexture = null;
    let fieldTexture = null;
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

    const colorInternal = webgl2 ? gl.RGBA8 : gl.RGBA;
    const colorFormat = gl.RGBA;

    function createTexture(width, height, internal, format, filter) {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, internal, width, height, 0, format, gl.UNSIGNED_BYTE, null);
      const mag = filter || gl.NEAREST;
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, mag);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, mag);
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
      if (compositeFb) {
        gl.deleteFramebuffer(compositeFb);
        gl.deleteFramebuffer(fieldFb);
        gl.deleteTexture(compositeTexture);
        gl.deleteTexture(fieldTexture);
        gl.deleteTexture(displayTexture);
      }
      destroyPbos();

      compositeTexture = createTexture(width, height, displayInternal, displayFormat);
      fieldTexture = createTexture(width, height, displayInternal, displayFormat);
      displayTexture = createTexture(width, height, displayInternal, displayFormat);
      compositeFb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, compositeFb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, compositeTexture, 0);
      fieldFb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fieldFb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, fieldTexture, 0);
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
    let fpsFrames = 0;
    let fpsLast = 0;
    let simTimeScale = 1;
    let simClock = 0;
    let simFrame = 0;
    let progress = 0;
    let easedProgress = 0;
    const minRadius = 0.16;
    const maxRadius = 2.4;
    let schwarzschildRadius = 0.5;
    let targetRadius = 0.5;
    let camX = 135;
    let camY = 135;
    let camZ = 203;
    let camYaw = -2.553;
    let camPitch = -0.505;
    let lookYaw = -2.553;
    let lookPitch = -0.505;
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
        camX = 135;
        camY = 135;
        camZ = 203;
        velX = 0;
        velY = 0;
        velZ = 0;
      }
    }

    function disableAttrib(loc) {
      if (loc >= 0) gl.disableVertexAttribArray(loc);
    }

    function bindFullscreen() {
      disableAttrib(starAttribs.corner);
      disableAttrib(starAttribs.star);
      disableAttrib(starAttribs.color);
      disableAttrib(starAttribs.kind);
      disableAttrib(lineAttribs.pos);
      disableAttrib(lineAttribs.meta);
      disableAttrib(dustAttribs.corner);
      disableAttrib(dustAttribs.dust);
      disableAttrib(dustAttribs.color);
      disableAttrib(holeDiscAttribs.corner);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffers.position);
      gl.vertexAttribPointer(blitAttrib, 2, gl.FLOAT, false, 0, 0);
      gl.enableVertexAttribArray(blitAttrib);
      if (lensAttrib >= 0 && lensAttrib !== blitAttrib) {
        gl.vertexAttribPointer(lensAttrib, 2, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(lensAttrib);
      }
    }

    function drawStarGeom(geom, count, renderWidth, renderHeight, b, useLens, useMask) {
      if (!count) return;
      disableAttrib(dustAttribs.corner);
      disableAttrib(dustAttribs.dust);
      disableAttrib(dustAttribs.color);
      gl.useProgram(starProgram);
      gl.uniform2f(starUniforms.resolution, renderWidth, renderHeight);
      gl.uniform1f(starUniforms.progress, easedProgress);
      gl.uniform1f(starUniforms.schwarzschildRadius, schwarzschildRadius);
      gl.uniform3f(starUniforms.camPos, camX, camY, camZ);
      gl.uniform3f(starUniforms.camFwd, b.fx, b.fy, b.fz);
      gl.uniform3f(starUniforms.camRight, b.rx, b.ry, b.rz);
      gl.uniform3f(starUniforms.camUp, b.ux, b.uy, b.uz);
      gl.uniform1i(starUniforms.mask, 0);
      gl.uniform1f(starUniforms.useLens, useLens ? 1 : 0);
      gl.uniform1f(starUniforms.useMask, useMask ? 1 : 0);
      gl.uniform2f(starUniforms.maskRes, fboWidth, fboHeight);
      gl.bindBuffer(gl.ARRAY_BUFFER, starBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, geom.subarray(0, count * 10), gl.STREAM_DRAW);
      const stride = 40;
      gl.vertexAttribPointer(starAttribs.corner, 2, gl.FLOAT, false, stride, 0);
      gl.enableVertexAttribArray(starAttribs.corner);
      gl.vertexAttribPointer(starAttribs.star, 4, gl.FLOAT, false, stride, 8);
      gl.enableVertexAttribArray(starAttribs.star);
      gl.vertexAttribPointer(starAttribs.color, 3, gl.FLOAT, false, stride, 24);
      gl.enableVertexAttribArray(starAttribs.color);
      gl.vertexAttribPointer(starAttribs.kind, 1, gl.FLOAT, false, stride, 36);
      gl.enableVertexAttribArray(starAttribs.kind);
      gl.drawArrays(gl.TRIANGLES, 0, count);
    }

    function drawLineGeom(geom, count, renderWidth, renderHeight, b, useMask) {
      if (!count) return;
      disableAttrib(starAttribs.corner);
      disableAttrib(starAttribs.star);
      disableAttrib(starAttribs.color);
      disableAttrib(starAttribs.kind);
      gl.useProgram(lineProgram);
      gl.uniform2f(lineUniforms.resolution, renderWidth, renderHeight);
      gl.uniform1f(lineUniforms.progress, easedProgress);
      gl.uniform3f(lineUniforms.camPos, camX, camY, camZ);
      gl.uniform3f(lineUniforms.camFwd, b.fx, b.fy, b.fz);
      gl.uniform1i(lineUniforms.mask, 0);
      gl.uniform1f(lineUniforms.useMask, useMask ? 1 : 0);
      gl.uniform2f(lineUniforms.maskRes, fboWidth, fboHeight);
      gl.bindBuffer(gl.ARRAY_BUFFER, lineBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, geom.subarray(0, count * 4), gl.STREAM_DRAW);
      const stride = 16;
      gl.vertexAttribPointer(lineAttribs.pos, 2, gl.FLOAT, false, stride, 0);
      gl.enableVertexAttribArray(lineAttribs.pos);
      gl.vertexAttribPointer(lineAttribs.meta, 2, gl.FLOAT, false, stride, 8);
      gl.enableVertexAttribArray(lineAttribs.meta);
      gl.drawArrays(gl.TRIANGLES, 0, count);
    }

    function clearComposite(renderWidth, renderHeight) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, compositeFb);
      gl.viewport(0, 0, renderWidth, renderHeight);
      gl.disable(gl.BLEND);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }

    function drawHoleDisc(renderWidth, renderHeight, b) {
      if (easedProgress <= 0) return;
      disableAttrib(starAttribs.corner);
      disableAttrib(starAttribs.star);
      disableAttrib(starAttribs.color);
      disableAttrib(starAttribs.kind);
      disableAttrib(lineAttribs.pos);
      disableAttrib(lineAttribs.meta);
      disableAttrib(dustAttribs.corner);
      disableAttrib(dustAttribs.dust);
      disableAttrib(dustAttribs.color);
      gl.useProgram(holeDiscProgram);
      gl.uniform2f(holeDiscUniforms.resolution, renderWidth, renderHeight);
      gl.uniform1f(holeDiscUniforms.progress, easedProgress);
      gl.uniform1f(holeDiscUniforms.schwarzschildRadius, schwarzschildRadius);
      gl.uniform3f(holeDiscUniforms.camPos, camX, camY, camZ);
      gl.uniform3f(holeDiscUniforms.camFwd, b.fx, b.fy, b.fz);
      gl.uniform3f(holeDiscUniforms.camRight, b.rx, b.ry, b.rz);
      gl.uniform3f(holeDiscUniforms.camUp, b.ux, b.uy, b.uz);
      gl.bindBuffer(gl.ARRAY_BUFFER, holeDiscBuffer);
      gl.vertexAttribPointer(holeDiscAttribs.corner, 2, gl.FLOAT, false, 0, 0);
      gl.enableVertexAttribArray(holeDiscAttribs.corner);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ZERO, gl.ONE_MINUS_SRC_COLOR);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    function renderStars(renderWidth, renderHeight, b) {
      if (starVertCount || lineVertCount) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, fieldFb);
        gl.viewport(0, 0, renderWidth, renderHeight);
        gl.disable(gl.BLEND);
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE);
        drawStarGeom(starGeom, starVertCount, renderWidth, renderHeight, b, false, false);
        drawLineGeom(lineGeom, lineVertCount, renderWidth, renderHeight, b, false);
      }
      clearComposite(renderWidth, renderHeight);
      gl.bindFramebuffer(gl.FRAMEBUFFER, compositeFb);
      gl.viewport(0, 0, renderWidth, renderHeight);
      if (starVertCount || lineVertCount) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE);
        gl.useProgram(lensProgram);
        bindFullscreen();
        gl.uniform2f(lensUniforms.resolution, renderWidth, renderHeight);
        gl.uniform1f(lensUniforms.progress, easedProgress);
        gl.uniform1f(lensUniforms.schwarzschildRadius, schwarzschildRadius);
        gl.uniform3f(lensUniforms.camPos, camX, camY, camZ);
        gl.uniform3f(lensUniforms.camFwd, b.fx, b.fy, b.fz);
        gl.uniform3f(lensUniforms.camRight, b.rx, b.ry, b.rz);
        gl.uniform3f(lensUniforms.camUp, b.ux, b.uy, b.uz);
        gl.uniform1i(lensUniforms.field, 0);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, fieldTexture);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }
      drawHoleDisc(renderWidth, renderHeight, b);
      if (starFrontCount || lineFrontCount) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE);
        drawStarGeom(starGeomFront, starFrontCount, renderWidth, renderHeight, b, false, false);
        drawLineGeom(lineGeomFront, lineFrontCount, renderWidth, renderHeight, b, false);
      }
      gl.disable(gl.BLEND);
    }

    function drawDustGeom(geom, count, renderWidth, renderHeight, b, useMask) {
      if (!count) return;
      disableAttrib(starAttribs.corner);
      disableAttrib(starAttribs.star);
      disableAttrib(starAttribs.color);
      disableAttrib(starAttribs.kind);
      disableAttrib(lineAttribs.pos);
      disableAttrib(lineAttribs.meta);
      gl.useProgram(dustProgram);
      gl.uniform2f(dustUniforms.resolution, renderWidth, renderHeight);
      gl.uniform1f(dustUniforms.progress, easedProgress);
      gl.uniform3f(dustUniforms.camPos, camX, camY, camZ);
      gl.uniform3f(dustUniforms.camFwd, b.fx, b.fy, b.fz);
      gl.uniform1i(dustUniforms.mask, 0);
      gl.uniform1f(dustUniforms.useMask, useMask ? 1 : 0);
      gl.uniform2f(dustUniforms.maskRes, fboWidth, fboHeight);
      gl.bindBuffer(gl.ARRAY_BUFFER, dustBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, geom.subarray(0, count * 9), gl.STREAM_DRAW);
      const stride = 36;
      gl.vertexAttribPointer(dustAttribs.corner, 2, gl.FLOAT, false, stride, 0);
      gl.enableVertexAttribArray(dustAttribs.corner);
      gl.vertexAttribPointer(dustAttribs.dust, 4, gl.FLOAT, false, stride, 8);
      gl.enableVertexAttribArray(dustAttribs.dust);
      gl.vertexAttribPointer(dustAttribs.color, 3, gl.FLOAT, false, stride, 24);
      gl.enableVertexAttribArray(dustAttribs.color);
      gl.drawArrays(gl.TRIANGLES, 0, count);
    }

    function renderDust(renderWidth, renderHeight, b) {
      if (!dustVertCount && !dustFrontCount) return;
      gl.bindFramebuffer(gl.FRAMEBUFFER, compositeFb);
      gl.viewport(0, 0, renderWidth, renderHeight);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      if (dustVertCount) drawDustGeom(dustGeom, dustVertCount, renderWidth, renderHeight, b, false);
      if (dustFrontCount) drawDustGeom(dustGeomFront, dustFrontCount, renderWidth, renderHeight, b, false);
      gl.disable(gl.BLEND);
    }

    function renderScene(renderWidth, renderHeight, now, b) {
      renderStars(renderWidth, renderHeight, b);
      renderDust(renderWidth, renderHeight, b);
      gl.bindFramebuffer(gl.FRAMEBUFFER, compositeFb);
      gl.flush();
    }

    function present(width, height, upload) {
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      if (gl.PIXEL_UNPACK_BUFFER) gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, null);
      bindFullscreen();
      gl.bindTexture(gl.TEXTURE_2D, displayTexture);
      if (upload) {
        let pixels = ENABLE_DITHER ? grayDst : graySrc;
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

    function pullPrevRead(renderWidth, renderHeight) {
      if (!pbos || !pboHasPrev) return false;
      const dest = readDest(renderWidth);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbos[pboIndex ^ 1]);
      gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, dest);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      unpackRead(renderWidth, renderHeight);
      return true;
    }

    function packCurrent(renderWidth, renderHeight) {
      if (pbos) {
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbos[pboIndex]);
        gl.readPixels(0, 0, renderWidth, renderHeight, displayFormat, gl.UNSIGNED_BYTE, 0);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        pboIndex ^= 1;
        pboHasPrev = true;
        return false;
      }
      const dest = readDest(renderWidth);
      gl.readPixels(0, 0, renderWidth, renderHeight, displayFormat, gl.UNSIGNED_BYTE, dest);
      unpackRead(renderWidth, renderHeight);
      return true;
    }

    function render(now) {
      raf = 0;
      if (!running || hidden) return;
      raf = requestAnimationFrame(render);
      const t = typeof now === "number" && now > 0 ? now : performance.now();
      fpsFrames += 1;
      if (!fpsLast) fpsLast = t;
      const elapsed = t - fpsLast;
      if (elapsed >= 500) {
        if (emit) emit({ type: "fps", v: Math.round((fpsFrames * 1000) / elapsed), time: simTimeScale });
        fpsFrames = 0;
        fpsLast = t;
      }

      let frameMs = 16;
      let timeScale;
      if (!haveTime) {
        haveTime = true;
        prevNow = now;
        timeScale = 15;
      } else {
        frameMs = Math.min(50, Math.max(1, now - prevNow));
        prevNow = now;
        timeScale = Math.abs(1 - (frameMs - 1));
        if (timeScale > 32) timeScale = 32;
      }
      simClock += frameMs * 0.001 * simTimeScale;

      if (progress < 1) {
        progress = Math.min(1, progress + 0.0025);
        easedProgress = easeOutBack(progress);
      }

      mouseX += (nextMouseX - mouseX) * 0.01 * timeScale;
      mouseY += (nextMouseY - mouseY) * 0.01 * timeScale;
      schwarzschildRadius += (targetRadius - schwarzschildRadius) * 0.02 * timeScale;
      camYaw = lookYaw;
      camPitch = lookPitch;
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
      updateGravity(frameMs, simTimeScale);

      if (displayWidth <= 0 || displayHeight <= 0) return;

      const renderWidth = Math.max(1, (displayWidth * RENDER_SCALE) | 0);
      const renderHeight = Math.max(1, (displayHeight * RENDER_SCALE) | 0);

      if (canvas.width !== displayWidth || canvas.height !== displayHeight || !compositeFb) {
        canvas.width = displayWidth;
        canvas.height = displayHeight;
        initFramebuffer(renderWidth, renderHeight);
      }

      refreshViews();
      const prevReady = pullPrevRead(renderWidth, renderHeight);
      packScene(b, renderWidth, renderHeight, schwarzschildRadius * easedProgress);
      renderScene(renderWidth, renderHeight, now, b);
      const syncReady = packCurrent(renderWidth, renderHeight);
      if (prevReady || syncReady) {
        if (ENABLE_DITHER) dither(renderWidth, renderHeight);
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
      adjustTimeScale(factor) {
        simTimeScale = Math.max(SIM_TIME_MIN, Math.min(SIM_TIME_MAX, simTimeScale * factor));
      },
      resetTimeScale() {
        simTimeScale = 1;
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
      setClusters,
    };
  }

  scope.createRenderer = createRenderer;
})(self);

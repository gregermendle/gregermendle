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
    uniform vec4 uStarProj[${MAX_STARS}];
    uniform vec3 uLineA[${MAX_LINES}];
    uniform vec3 uLineB[${MAX_LINES}];
    uniform int uStarCount;
    uniform int uLineCount;
    uniform int uDeflectCount;
    uniform int uMicro;
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
      if (t <= 0.0 || t >= maxT || t > 90.0) return rd;
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
        if (float(i) + 0.5 > float(uDeflectCount)) break;
        vec4 s = uStars[i];
        rd = pullRay(rd, s.xyz - ro, s.w * STAR_COMPACT, maxT);
      }
      return rd;
    }

    float einstein2(float mass, float zL, float zS) {
      float dls = zS - zL;
      if (dls <= 0.0 || zL <= 1e-4 || zL > 220.0) return 0.0;
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
      float R2 = influenceRadius * influenceRadius;
      vec3 closest = cross(rd, cross(ro, rd));
      float d2 = dot(closest, closest);
      if (d2 > R2) return vec4(0.0);
      float halfChord = sqrt(R2 - d2);
      float tClosest = -dot(ro, rd);
      if (tClosest + halfChord < 0.0) return vec4(0.0);
      float jitter = hash(uv) * STEP_SIZE;
      vec3 p = closest - rd * (halfChord - jitter);
      if (dot(ro, ro) < R2) p = ro + rd * jitter;
      float r = length(p);
      vec4 col = vec4(0.0);
      float totalDist = max(tClosest - halfChord, 0.0) + jitter;
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

    float starField(vec2 uv, vec3 ro, float radius, float minRes, float holeMask) {
      float glow = 0.0;
      float px = 1.0 / minRes;
      float holeZ = dot(-ro, camFwd);
      vec2 holeP = holeZ > 1e-4
        ? vec2(dot(-ro, camRight), dot(-ro, camUp)) / holeZ
        : vec2(100.0);
      float coverZ = 1.0e20;
      for (int j = 0; j < MAX_STARS; j++) {
        if (float(j) + 0.5 > float(uStarCount)) break;
        vec4 p = uStarProj[j];
        vec2 cd = uv - p.xy;
        float cr = max(p.w, px * 1.15);
        if (p.z > 1e-4 && dot(cd, cd) < cr * cr) {
          coverZ = min(coverZ, p.z);
        }
      }
      for (int i = 0; i < MAX_STARS; i++) {
        if (float(i) + 0.5 > float(uStarCount)) break;
        vec4 p = uStarProj[i];
        float z = p.z;
        if (z <= 1e-4 || z > coverZ) continue;
        if (holeZ > 1e-4 && z > holeZ && holeMask > 0.5) continue;
        vec2 sp = p.xy;
        float ang = p.w;
        float holeTe2 = einstein2(radius, holeZ, z);
        if (holeTe2 < px * px) holeTe2 = 0.0;
        float ring = sqrt(holeTe2);
        float pad = max(max(ang * 6.0, px * 4.0), ring * 2.4);
        vec2 holeD = uv - holeP;
        bool nearHole = holeTe2 > 0.0 && dot(holeD, holeD) < pad * pad;
        if (!nearHole && (abs(sp.x) > uViewHalf.x + pad || abs(sp.y) > uViewHalf.y + pad)) continue;
        vec2 src = uv;
        if (holeTe2 > 0.0) src -= lensPull(uv, holeP, holeTe2);
        if (uMicro > 0) {
          for (int j = 0; j < MAX_STARS; j++) {
            if (float(j) + 0.5 > float(uStarCount)) break;
            if (i == j) continue;
            vec4 o = uStarProj[j];
            float zj = o.z;
            if (zj <= 1e-4 || zj >= z || zj > 90.0) continue;
            float starTe2 = einstein2(uStars[j].w * STAR_COMPACT, zj, z);
            if (starTe2 >= px * px) src -= lensPull(uv, o.xy, starTe2);
          }
        }
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
        vec3 la = uLineA[i];
        vec3 lb = uLineB[i];
        float zLine = la.z;
        if (zLine < 0.0) continue;
        if (holeZ > 1e-4 && zLine > holeZ && holeMask > 0.5) continue;
        vec2 a = la.xy;
        vec2 b = lb.xy;
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
      vec3 ro = camPos;
      float radius = schwarzschildRadius * progress;
      vec3 rd0 = normalize(uv.x * camRight + uv.y * camUp + camFwd);
      float holeT = max(-dot(ro, rd0), 0.0);
      if (holeT < 1e-4) holeT = 1.0e20;
      vec3 rd = (holeT > 220.0 || uDeflectCount < 1) ? rd0 : deflectRay(ro, rd0, holeT);
      vec4 marched = rayMarch(ro, rd, uv * aspect * 5.0, radius);
      vec4 col = vec4(marched.rgb * progress, marched.a);
      if (uStarCount > 0 || uLineCount > 0) {
        col.rgb += vec3(starField(uv, ro, radius, minRes, marched.a) * progress);
      }
      float dist = length(ro);
      float beaconMix = smoothstep(50.0, 160.0, dist);
      float marchLum = dot(col.rgb, vec3(0.333));
      float fill = beaconMix * (1.0 - smoothstep(0.0, 0.12, marchLum));
      float beacon = holeBeacon(uv, ro, radius, minRes) * progress * fill;
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
    const program = initShaderProgram(gl, shaders.vs, shaders.fs);
    const blitProgram = initShaderProgram(gl, shaders.blitVs, shaders.blitFs);
    const buffers = initBuffers(gl);
    loadWasmDither();

    const STAR_COMPACT = 0.000004;
    const starData = new Float32Array(MAX_STARS * 4);
    const starProjData = new Float32Array(MAX_STARS * 4);
    const lineAData = new Float32Array(MAX_LINES * 3);
    const lineBData = new Float32Array(MAX_LINES * 3);
    let starCount = 0;
    let lineCount = 0;
    let deflectCount = 0;
    let micro = 0;

    let clusterCount = 0;
    let clusterOx = new Float32Array(0);
    let clusterOy = new Float32Array(0);
    let clusterOz = new Float32Array(0);
    let clusterAmpX = new Float32Array(0);
    let clusterAmpY = new Float32Array(0);
    let clusterAmpZ = new Float32Array(0);
    let clusterSpeed = new Float32Array(0);
    let clusterPhase = new Float32Array(0);
    let clusterBound = new Float32Array(0);
    let clusterAmpR = new Float32Array(0);
    let clusterR = new Float32Array(0);
    let pointStart = new Int32Array(0);
    let pointCount = new Int32Array(0);
    let pointX = new Float32Array(0);
    let pointY = new Float32Array(0);
    let pointZ = new Float32Array(0);
    let pointR = new Float32Array(0);
    let visWX = new Float32Array(0);
    let visWY = new Float32Array(0);
    let visWZ = new Float32Array(0);
    let visWR = new Float32Array(0);
    let candMinD2 = new Float32Array(0);
    let candStart = new Int32Array(0);
    let candNpts = new Int32Array(0);
    let candOrder = new Int32Array(0);

    function setClusters(next) {
      const src = Array.isArray(next) ? next : [];
      clusterCount = src.length;
      clusterOx = new Float32Array(clusterCount);
      clusterOy = new Float32Array(clusterCount);
      clusterOz = new Float32Array(clusterCount);
      clusterAmpX = new Float32Array(clusterCount);
      clusterAmpY = new Float32Array(clusterCount);
      clusterAmpZ = new Float32Array(clusterCount);
      clusterSpeed = new Float32Array(clusterCount);
      clusterPhase = new Float32Array(clusterCount);
      clusterBound = new Float32Array(clusterCount);
      clusterAmpR = new Float32Array(clusterCount);
      clusterR = new Float32Array(clusterCount);
      pointStart = new Int32Array(clusterCount);
      pointCount = new Int32Array(clusterCount);
      candMinD2 = new Float32Array(clusterCount);
      candStart = new Int32Array(clusterCount);
      candNpts = new Int32Array(clusterCount);
      candOrder = new Int32Array(clusterCount);
      let totalPts = 0;
      for (let i = 0; i < clusterCount; i++) {
        const pts = src[i] && src[i].points;
        totalPts += pts ? pts.length : 0;
      }
      pointX = new Float32Array(totalPts);
      pointY = new Float32Array(totalPts);
      pointZ = new Float32Array(totalPts);
      pointR = new Float32Array(totalPts);
      visWX = new Float32Array(totalPts);
      visWY = new Float32Array(totalPts);
      visWZ = new Float32Array(totalPts);
      visWR = new Float32Array(totalPts);
      let p = 0;
      for (let c = 0; c < clusterCount; c++) {
        const cluster = src[c] || {};
        const origin = cluster.origin || [0, 0, 0];
        const glide = cluster.glide || {};
        const amp = glide.amp || [0, 0, 0];
        clusterOx[c] = origin[0] || 0;
        clusterOy[c] = origin[1] || 0;
        clusterOz[c] = origin[2] || 0;
        clusterAmpX[c] = amp[0] || 0;
        clusterAmpY[c] = amp[1] || 0;
        clusterAmpZ[c] = amp[2] || 0;
        clusterSpeed[c] = glide.speed || 0;
        clusterPhase[c] = glide.phase || 0;
        clusterAmpR[c] = Math.hypot(clusterAmpX[c], clusterAmpY[c], clusterAmpZ[c]);
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
          pointX[p] = x;
          pointY[p] = y;
          pointZ[p] = z;
          pointR[p] = r;
          const ext = Math.hypot(x, y, z) + r;
          if (ext > maxExt) maxExt = ext;
          p++;
        }
        clusterBound[c] = maxExt;
      }
    }

    fetch("/js/clusters.json")
      .then((res) => res.json())
      .then((data) => {
        setClusters(data.clusters || []);
      })
      .catch(() => {
        setClusters([]);
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

    function clusterInView(b, ox, oy, oz, boundR, minRes, renderW, renderH) {
      const dx = ox - camX;
      const dy = oy - camY;
      const dz = oz - camZ;
      const z = dx * b.fx + dy * b.fy + dz * b.fz;
      if (z + boundR < 0.05) return false;
      if (z <= boundR * 2 + 0.05) return true;
      const sx = (dx * b.rx + dy * b.ry + dz * b.rz) / z;
      const sy = (dx * b.ux + dy * b.uy + dz * b.uz) / z;
      const pad = (boundR / z) * 8 + 0.18;
      const halfX = renderW / (2 * minRes) + pad;
      const halfY = renderH / (2 * minRes) + pad;
      return Math.abs(sx) <= halfX && Math.abs(sy) <= halfY;
    }

    function einstein2JS(mass, zL, zS) {
      const dls = zS - zL;
      if (dls <= 0 || zL <= 1e-4 || zL > 220) return 0;
      return (2 * mass * dls) / Math.max(zL * zS, 1e-8);
    }

    function lensPullJS(uvx, uvy, lpx, lpy, te2) {
      if (te2 <= 0) return [0, 0];
      const dx = uvx - lpx;
      const dy = uvy - lpy;
      const s = te2 / Math.max(dx * dx + dy * dy, te2 * 0.08);
      return [dx * s, dy * s];
    }

    function packScene(now, b, renderW, renderH, radius) {
      const t = now * 0.001;
      const minRes = Math.min(renderW, renderH);
      let visN = 0;
      let candN = 0;
      for (let c = 0; c < clusterCount; c++) {
        const npts = pointCount[c];
        if (!npts) continue;
        if (
          !clusterInView(
            b,
            clusterOx[c],
            clusterOy[c],
            clusterOz[c],
            clusterBound[c] + clusterAmpR[c],
            minRes,
            renderW,
            renderH
          )
        ) {
          continue;
        }
        const speed = clusterSpeed[c];
        const phase = clusterPhase[c];
        const ox = clusterOx[c] + clusterAmpX[c] * Math.sin(t * speed + phase);
        const oy = clusterOy[c] + clusterAmpY[c] * Math.sin(t * speed * 0.83 + phase + 1.1);
        const oz = clusterOz[c] + clusterAmpZ[c] * Math.cos(t * speed * 0.71 + phase);
        if (!clusterInView(b, ox, oy, oz, clusterBound[c], minRes, renderW, renderH)) continue;
        const start = pointStart[c];
        const visStart = visN;
        let minDist2 = 1e20;
        for (let i = 0; i < npts; i++) {
          const pi = start + i;
          const wx = ox + pointX[pi];
          const wy = oy + pointY[pi];
          const wz = oz + pointZ[pi];
          const sr = pointR[pi];
          if (!starInView(b, wx, wy, wz, sr, minRes, renderW, renderH)) continue;
          const dx = wx - camX;
          const dy = wy - camY;
          const dz = wz - camZ;
          const dist2 = dx * dx + dy * dy + dz * dz;
          if (dist2 < minDist2) minDist2 = dist2;
          visWX[visN] = wx;
          visWY[visN] = wy;
          visWZ[visN] = wz;
          visWR[visN] = sr;
          visN++;
        }
        if (visN !== visStart) {
          candMinD2[candN] = minDist2;
          candStart[candN] = visStart;
          candNpts[candN] = visN - visStart;
          candOrder[candN] = candN;
          candN++;
        }
      }
      for (let i = 1; i < candN; i++) {
        const id = candOrder[i];
        const d = candMinD2[id];
        let j = i - 1;
        while (j >= 0 && candMinD2[candOrder[j]] > d) {
          candOrder[j + 1] = candOrder[j];
          j--;
        }
        candOrder[j + 1] = id;
      }
      starCount = 0;
      lineCount = 0;
      for (let ci = 0; ci < candN && starCount < MAX_STARS; ci++) {
        const c = candOrder[ci];
        const first = starCount;
        const start = candStart[c];
        const n = candNpts[c];
        for (let i = 0; i < n && starCount < MAX_STARS; i++) {
          const s = start + i;
          const o = starCount * 4;
          starData[o] = visWX[s];
          starData[o + 1] = visWY[s];
          starData[o + 2] = visWZ[s];
          starData[o + 3] = visWR[s];
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

      const px = 1 / minRes;
      const px2 = px * px;
      const holeZ = -camX * b.fx - camY * b.fy - camZ * b.fz;
      const holePx = holeZ > 1e-4 ? (-camX * b.rx - camY * b.ry - camZ * b.rz) / holeZ : 100;
      const holePy = holeZ > 1e-4 ? (-camX * b.ux - camY * b.uy - camZ * b.uz) / holeZ : 100;
      deflectCount = 0;
      micro = 0;
      for (let i = 0; i < starCount; i++) {
        const o = i * 4;
        const dx = starData[o] - camX;
        const dy = starData[o + 1] - camY;
        const dz = starData[o + 2] - camZ;
        const z = dx * b.fx + dy * b.fy + dz * b.fz;
        const sr = starData[o + 3];
        let spx = 100;
        let spy = 100;
        let ang = 0;
        if (z > 1e-4) {
          spx = (dx * b.rx + dy * b.ry + dz * b.rz) / z;
          spy = (dx * b.ux + dy * b.uy + dz * b.uz) / z;
          ang = sr / Math.max(z, sr * 0.35);
        }
        starProjData[o] = spx;
        starProjData[o + 1] = spy;
        starProjData[o + 2] = z;
        starProjData[o + 3] = ang;
        const mass = sr * STAR_COMPACT;
        if (z > 1e-4 && (mass * 48) / z > px * 0.5) deflectCount = starCount;
      }
      for (let i = 0; i < starCount && !micro; i++) {
        const z = starProjData[i * 4 + 2];
        for (let j = 0; j < starCount; j++) {
          if (i === j) continue;
          const zj = starProjData[j * 4 + 2];
          if (zj <= 1e-4 || zj >= z || zj > 90) continue;
          if (einstein2JS(starData[j * 4 + 3] * STAR_COMPACT, zj, z) >= px2) {
            micro = 1;
            break;
          }
        }
      }
      for (let i = 0; i < lineCount; i++) {
        const lo = i * 3;
        const dax = lineAData[lo] - camX;
        const day = lineAData[lo + 1] - camY;
        const daz = lineAData[lo + 2] - camZ;
        const dbx = lineBData[lo] - camX;
        const dby = lineBData[lo + 1] - camY;
        const dbz = lineBData[lo + 2] - camZ;
        const za = dax * b.fx + day * b.fy + daz * b.fz;
        const zb = dbx * b.fx + dby * b.fy + dbz * b.fz;
        if (za <= 1e-4 && zb <= 1e-4) {
          lineAData[lo] = 0;
          lineAData[lo + 1] = 0;
          lineAData[lo + 2] = -1;
          lineBData[lo] = 0;
          lineBData[lo + 1] = 0;
          lineBData[lo + 2] = 0;
          continue;
        }
        const zLine = Math.min(za > 1e-4 ? za : zb, zb > 1e-4 ? zb : za);
        let ax = (dax * b.rx + day * b.ry + daz * b.rz) / Math.max(za, 1e-4);
        let ay = (dax * b.ux + day * b.uy + daz * b.uz) / Math.max(za, 1e-4);
        let bx = (dbx * b.rx + dby * b.ry + dbz * b.rz) / Math.max(zb, 1e-4);
        let by = (dbx * b.ux + dby * b.uy + dbz * b.uz) / Math.max(zb, 1e-4);
        if (holeZ > 1e-4 && holeZ < zLine) {
          const te2 = einstein2JS(radius, holeZ, zLine);
          if (te2 >= px2) {
            const pa = lensPullJS(ax, ay, holePx, holePy, te2);
            const pb = lensPullJS(bx, by, holePx, holePy, te2);
            ax -= pa[0];
            ay -= pa[1];
            bx -= pb[0];
            by -= pb[1];
          }
        }
        lineAData[lo] = ax;
        lineAData[lo + 1] = ay;
        lineAData[lo + 2] = zLine;
        lineBData[lo] = bx;
        lineBData[lo + 1] = by;
        lineBData[lo + 2] = 0;
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
        starProj: gl.getUniformLocation(program, "uStarProj[0]"),
        lineA: gl.getUniformLocation(program, "uLineA[0]"),
        lineB: gl.getUniformLocation(program, "uLineB[0]"),
        starCount: gl.getUniformLocation(program, "uStarCount"),
        lineCount: gl.getUniformLocation(program, "uLineCount"),
        deflectCount: gl.getUniformLocation(program, "uDeflectCount"),
        micro: gl.getUniformLocation(program, "uMicro"),
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
    let fpsFrames = 0;
    let fpsLast = 0;
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
      packScene(now, b, renderWidth, renderHeight, schwarzschildRadius * easedProgress);
      if (starCount) {
        if (deflectCount || micro) {
          gl.uniform4fv(programInfo.uniformLocations.stars, starData.subarray(0, starCount * 4));
        }
        gl.uniform4fv(programInfo.uniformLocations.starProj, starProjData.subarray(0, starCount * 4));
      }
      if (lineCount) {
        gl.uniform3fv(programInfo.uniformLocations.lineA, lineAData.subarray(0, lineCount * 3));
        gl.uniform3fv(programInfo.uniformLocations.lineB, lineBData.subarray(0, lineCount * 3));
      }
      gl.uniform1i(programInfo.uniformLocations.starCount, starCount);
      gl.uniform1i(programInfo.uniformLocations.lineCount, lineCount);
      gl.uniform1i(programInfo.uniformLocations.deflectCount, deflectCount);
      gl.uniform1i(programInfo.uniformLocations.micro, micro);
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
        gl.readPixels(0, 0, renderWidth, renderHeight, sceneFormat, gl.UNSIGNED_BYTE, 0);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        pboIndex ^= 1;
        pboHasPrev = true;
        return false;
      }
      const dest = readDest(renderWidth);
      gl.readPixels(0, 0, renderWidth, renderHeight, sceneFormat, gl.UNSIGNED_BYTE, dest);
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
        if (emit) emit({ type: "fps", v: Math.round((fpsFrames * 1000) / elapsed) });
        fpsFrames = 0;
        fpsLast = t;
      }

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

      if (displayWidth <= 0 || displayHeight <= 0) return;

      const renderWidth = Math.max(1, (displayWidth * RENDER_SCALE) | 0);
      const renderHeight = Math.max(1, (displayHeight * RENDER_SCALE) | 0);

      if (canvas.width !== displayWidth || canvas.height !== displayHeight || !framebuffer) {
        canvas.width = displayWidth;
        canvas.height = displayHeight;
        initFramebuffer(renderWidth, renderHeight);
      }

      refreshViews();
      const prevReady = pullPrevRead(renderWidth, renderHeight);
      renderScene(renderWidth, renderHeight, now);
      const syncReady = packCurrent(renderWidth, renderHeight);
      if (prevReady || syncReady) {
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
      setClusters,
    };
  }

  scope.createRenderer = createRenderer;
})(self);

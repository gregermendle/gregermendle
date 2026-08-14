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

  const DISK_SIZE = 23;
  const MAX_DEFLECT = 8;
  const STAR_RADIUS = 0.145;
  const VIEW_FRUSTUM_PAD = 0.32;
  const STAR_ANGULAR_PAD = 12;
  const STAR_BASE_PAD = 0.32;
  const STAR_GLOW_PAD = 9;
  const STAR_PX_PAD = 6;
  const MIN_LENS_Z = 2.5;
  const STAR_FAR_Z = 2800;
  const LENS_Z_MIN = 0.8;
  const LENS_FADE_BEHIND = -12;
  const LENS_FADE_FRONT = 0.6;
  const LENS_FAR_START = 80;
  const LENS_FAR_END = 320;
  const DISK_OCCLUDE_LUMA = 0.2;

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
          }
        }
        if (r > influenceRadius && dot(p, rd) > 0.0) return col;
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
    uniform vec4 uStars[${MAX_DEFLECT}];
    uniform int uDeflectCount;
    uniform vec2 uViewHalf;
    ${fragOut}
    #define MAX_STEPS ${maxSteps}
    #define WARP_SIZE 0.25
    #define STEP_SIZE ${stepSize.toFixed(2)}
    #define DISK_SIZE ${DISK_SIZE.toFixed(1)}
    #define MAX_DEFLECT ${MAX_DEFLECT}
    #define STAR_COMPACT 0.000004
    #define MIN_LENS_Z ${MIN_LENS_Z.toFixed(2)}

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
      for (int i = 0; i < MAX_DEFLECT; i++) {
        if (float(i) + 0.5 > float(uDeflectCount)) break;
        vec4 s = uStars[i];
        vec3 rel = s.xyz - ro;
        float z = dot(rel, camFwd);
        if (z < MIN_LENS_Z || z > 90.0) continue;
        rd = pullRay(rd, rel, s.w * STAR_COMPACT, maxT);
      }
      return rd;
    }

    float lensFade(float holeZ) {
      return smoothstep(${LENS_FADE_BEHIND.toFixed(1)}, ${LENS_FADE_FRONT.toFixed(1)}, holeZ)
        * (1.0 - smoothstep(${LENS_FAR_START.toFixed(1)}, ${LENS_FAR_END.toFixed(1)}, holeZ));
    }

    float holeTe2Of(float mass, float holeZ, float z) {
      float fade = lensFade(holeZ);
      if (fade <= 0.0 || z <= 0.0) return 0.0;
      float zL = max(holeZ, ${LENS_Z_MIN.toFixed(2)});
      float dls = z - zL;
      if (dls <= 0.0) return 0.0;
      return fade * 2.0 * mass * dls / max(zL * z, 1e-8);
    }

    vec2 holePos(vec3 ro) {
      float zProj = max(dot(-ro, camFwd), ${LENS_Z_MIN.toFixed(2)});
      return vec2(dot(-ro, camRight), dot(-ro, camUp)) / zProj;
    }

    float einstein2(float mass, float zL, float zS) {
      return holeTe2Of(mass, zL, zS);
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

    void main() {
      float minRes = min(resolution.x, resolution.y);
      vec2 uv = (gl_FragCoord.xy - 0.5 * resolution.xy) / minRes;
      vec2 aspect = resolution.xy / minRes;
      vec3 ro = camPos;
      float radius = schwarzschildRadius * progress;
      vec3 rd0 = normalize(uv.x * camRight + uv.y * camUp + camFwd);
      float holeT = max(-dot(ro, rd0), 0.0);
      if (holeT < 1e-4) holeT = 1.0e20;
      vec3 rd = uDeflectCount < 1 ? rd0 : deflectRay(ro, rd0, holeT);
      vec4 marched = rayMarch(ro, rd, uv * aspect * 5.0, radius);
      vec4 col = vec4(marched.rgb * progress, marched.a);
      float dist = length(ro);
      float beaconMix = smoothstep(50.0, 160.0, dist);
      float marchLum = dot(col.rgb, vec3(0.333));
      float fill = beaconMix * (1.0 - smoothstep(0.0, 0.12, marchLum));
      float beacon = holeBeacon(uv, ro, radius, minRes) * progress * fill;
      col.rgb = max(col.rgb, vec3(beacon));
      float mask = max(marched.a, step(${DISK_OCCLUDE_LUMA.toFixed(2)}, marched.r * progress));
      ${webgl2 ? `${writeColor} = vec4(col.r, mask, 0.0, 1.0);` : `${writeColor} = vec4(col.rgb, mask);`}
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

    const maskCh = webgl2 ? "g" : "a";
    const lensLib = `
    float lensFade(float holeZ) {
      return smoothstep(${LENS_FADE_BEHIND.toFixed(1)}, ${LENS_FADE_FRONT.toFixed(1)}, holeZ)
        * (1.0 - smoothstep(${LENS_FAR_START.toFixed(1)}, ${LENS_FAR_END.toFixed(1)}, holeZ));
    }
    float holeTe2Of(float mass, float holeZ, float z) {
      float fade = lensFade(holeZ);
      if (fade <= 0.0 || z <= 0.0) return 0.0;
      float zL = max(holeZ, ${LENS_Z_MIN.toFixed(2)});
      float dls = z - zL;
      if (dls <= 0.0) return 0.0;
      return fade * 2.0 * mass * dls / max(zL * z, 1e-8);
    }
    vec2 holePos(vec3 ro) {
      float zProj = max(dot(-ro, camFwd), ${LENS_Z_MIN.toFixed(2)});
      return vec2(dot(-ro, camRight), dot(-ro, camUp)) / zProj;
    }`;
    const starVs = `${ver}${attr} vec2 aCorner;
    ${attr} vec4 aStar;
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
      float holeTe2 = uUseLens > 0.5 ? holeTe2Of(radius, holeZ, z) : 0.0;
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
    ${varyIn} vec4 vStar;
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
      float holeMask = uUseMask > 0.5 ? ${tex}(uMask, gl_FragCoord.xy / resolution).${maskCh} : 0.0;
      float z = vStar.z;
      if (z <= 1e-4 || (uUseMask > 0.5 && holeZ > 1e-4 && z > holeZ && holeMask > 0.5)) {
        ${writeColor} = vec4(0.0);
        return;
      }
      vec2 sp = vStar.xy;
      float ang = vStar.w;
      float holeTe2 = uUseLens > 0.5 ? holeTe2Of(radius, holeZ, z) : 0.0;
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
      ${writeColor} = vec4(vec3(glow), 1.0);
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
    ${varyIn} vec2 vMeta;
    ${fragOut}
    void main() {
      float holeZ = dot(-camPos, camFwd);
      float holeMask = uUseMask > 0.5 ? ${tex}(uMask, gl_FragCoord.xy / resolution).${maskCh} : 0.0;
      if (vMeta.x < 0.0 || (uUseMask > 0.5 && holeZ > 1e-4 && vMeta.x > holeZ && holeMask > 0.5)) {
        ${writeColor} = vec4(0.0);
        return;
      }
      float glow = (1.0 - smoothstep(0.35, 0.85, abs(vMeta.y))) * 0.16 * progress;
      ${writeColor} = vec4(vec3(glow), 1.0);
    }`;

    const lensFs = `${ver}precision highp float;
    uniform sampler2D uField;
    uniform sampler2D uMask;
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
      float holeMask = ${tex}(uMask, vTexCoord).${maskCh};
      if (holeMask > 0.5) {
        ${writeColor} = vec4(0.0);
        return;
      }
      float te2 = holeTe2Of(schwarzschildRadius * progress, holeZ, 1.0e6);
      vec2 src = uv;
      if (te2 > 0.0) src -= lensPull(uv, holePos(ro), te2);
      vec2 srcTex = src * minRes / resolution + 0.5;
      if (srcTex.x < 0.0 || srcTex.y < 0.0 || srcTex.x > 1.0 || srcTex.y > 1.0) {
        ${writeColor} = vec4(0.0);
        return;
      }
      float glow = ${tex}(uField, srcTex).r;
      ${writeColor} = vec4(vec3(glow), 1.0);
    }`;

    return { vs, fs, blitVs, blitFs, starVs, starFs, lineVs, lineFs, lensFs };
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
    const starProgram = initShaderProgram(gl, shaders.starVs, shaders.starFs);
    const lineProgram = initShaderProgram(gl, shaders.lineVs, shaders.lineFs);
    const lensProgram = initShaderProgram(gl, shaders.blitVs, shaders.lensFs);
    const buffers = initBuffers(gl);
    loadWasmDither();

    const STAR_COMPACT = 0.000004;
    const starData = new Float32Array(MAX_DEFLECT * 4);
    let deflectCount = 0;
    let starVertCount = 0;
    let starFrontCount = 0;
    let lineVertCount = 0;
    let starGeom = new Float32Array(7 * 6 * 256);
    let starGeomFront = new Float32Array(7 * 6 * 64);
    let lineGeom = new Float32Array(4 * 6 * 256);

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
    let dustCount = new Int32Array(0);
    let dustRadius = new Float32Array(0);
    let dustSize = new Float32Array(0);
    let dustGain = new Float32Array(0);
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
      dustCount = new Int32Array(clusterCount);
      dustRadius = new Float32Array(clusterCount);
      dustSize = new Float32Array(clusterCount);
      dustGain = new Float32Array(clusterCount);
      let totalPts = 0;
      for (let i = 0; i < clusterCount; i++) {
        const pts = src[i] && src[i].points;
        totalPts += pts ? pts.length : 0;
      }
      pointX = new Float32Array(totalPts);
      pointY = new Float32Array(totalPts);
      pointZ = new Float32Array(totalPts);
      pointR = new Float32Array(totalPts);
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
        const dust = Array.isArray(cluster.dust) ? cluster.dust[0] : cluster.dust;
        if (dust && (dust.count | 0) > 0) {
          const col = dust.color || [1, 1, 1];
          const luma = Math.max(0, (col[0] || 0) * 0.333 + (col[1] || 0) * 0.333 + (col[2] || 0) * 0.333);
          dustCount[c] = dust.count | 0;
          dustRadius[c] = dust.radius || 3;
          dustSize[c] = dust.size || 0.7;
          dustGain[c] = Math.max(0, luma * (dust.opacity == null ? 0.04 : dust.opacity));
          const dustExt = dustRadius[c] + dustSize[c];
          if (dustExt > maxExt) maxExt = dustExt;
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

    function viewHalf(renderW, renderH, minRes) {
      return {
        x: renderW / (2 * minRes) + VIEW_FRUSTUM_PAD,
        y: renderH / (2 * minRes) + VIEW_FRUSTUM_PAD,
      };
    }

    function starInView(b, wx, wy, wz, starR, minRes, renderW, renderH) {
      const dx = wx - camX;
      const dy = wy - camY;
      const dz = wz - camZ;
      const z = dx * b.fx + dy * b.fy + dz * b.fz;
      if (z <= 0.05 || z > STAR_FAR_Z) return false;
      const sx = (dx * b.rx + dy * b.ry + dz * b.rz) / z;
      const sy = (dx * b.ux + dy * b.uy + dz * b.uz) / z;
      const pad = (starR / z) * STAR_ANGULAR_PAD + STAR_BASE_PAD;
      const half = viewHalf(renderW, renderH, minRes);
      return Math.abs(sx) <= half.x + pad && Math.abs(sy) <= half.y + pad;
    }

    function clusterInView(b, ox, oy, oz, boundR, minRes, renderW, renderH) {
      const dx = ox - camX;
      const dy = oy - camY;
      const dz = oz - camZ;
      const z = dx * b.fx + dy * b.fy + dz * b.fz;
      if (z + boundR < 0.05 || z - boundR > STAR_FAR_Z) return false;
      if (z <= boundR * 2 + 0.05) return true;
      const sx = (dx * b.rx + dy * b.ry + dz * b.rz) / z;
      const sy = (dx * b.ux + dy * b.uy + dz * b.uz) / z;
      const pad = (boundR / z) * STAR_ANGULAR_PAD + STAR_BASE_PAD;
      const half = viewHalf(renderW, renderH, minRes);
      return Math.abs(sx) <= half.x + pad && Math.abs(sy) <= half.y + pad;
    }

    function smoothstepJS(edge0, edge1, x) {
      const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
      return t * t * (3 - 2 * t);
    }

    function lensFadeJS(holeZ) {
      return (
        smoothstepJS(LENS_FADE_BEHIND, LENS_FADE_FRONT, holeZ) *
        (1 - smoothstepJS(LENS_FAR_START, LENS_FAR_END, holeZ))
      );
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

    function emitStarTo(bufName, countName, spx, spy, z, ang, kind) {
      const count = countName === "front" ? starFrontCount : starVertCount;
      const need = count * 7 + 42;
      if (countName === "front") starGeomFront = growFloat(starGeomFront, need);
      else starGeom = growFloat(starGeom, need);
      const geom = countName === "front" ? starGeomFront : starGeom;
      const corners = [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1];
      for (let i = 0; i < 6; i++) {
        const o = count * 7 + i * 7;
        geom[o] = corners[i * 2];
        geom[o + 1] = corners[i * 2 + 1];
        geom[o + 2] = spx;
        geom[o + 3] = spy;
        geom[o + 4] = z;
        geom[o + 5] = ang;
        geom[o + 6] = kind;
      }
      if (countName === "front") starFrontCount += 6;
      else starVertCount += 6;
    }

    function emitStarQuad(spx, spy, z, ang, kind) {
      emitStarTo("starGeom", "back", spx, spy, z, ang, kind);
    }

    function dustHash(c, i, k) {
      let n = Math.imul(c + 1, 374761393) ^ Math.imul(i + 1, 668265263) ^ Math.imul(k + 1, 1442695041);
      n = Math.imul(n ^ (n >>> 15), 2246822519);
      n = Math.imul(n ^ (n >>> 13), 3266489917);
      return ((n ^ (n >>> 16)) >>> 0) / 4294967296;
    }

    function emitClusterDust(c, ox, oy, oz, b, minRes, renderW, renderH, holeZ) {
      const n = dustCount[c];
      if (!n) return;
      const rad = dustRadius[c];
      const size = dustSize[c];
      const kind = 2 + dustGain[c];
      const zCut = Math.max(holeZ, LENS_Z_MIN);
      for (let i = 0; i < n; i++) {
        const u = dustHash(c, i, 0);
        const v = dustHash(c, i, 1);
        const w = dustHash(c, i, 2);
        const theta = u * Math.PI * 2;
        const zN = v * 2 - 1;
        const rxy = Math.sqrt(Math.max(0, 1 - zN * zN)) * Math.cbrt(w);
        const r = rad * rxy;
        const wx = ox + Math.cos(theta) * r;
        const wy = oy + zN * rad * 0.28;
        const wz = oz + Math.sin(theta) * r;
        if (!starInView(b, wx, wy, wz, size, minRes, renderW, renderH)) continue;
        const dx = wx - camX;
        const dy = wy - camY;
        const dz = wz - camZ;
        const z = dx * b.fx + dy * b.fy + dz * b.fz;
        const spx = (dx * b.rx + dy * b.ry + dz * b.rz) / z;
        const spy = (dx * b.ux + dy * b.uy + dz * b.uz) / z;
        const ang = size / Math.max(z, size * 0.35);
        if (z > zCut) emitStarQuad(spx, spy, z, ang, kind);
        else emitStarTo("starGeomFront", "front", spx, spy, z, ang, kind);
      }
    }

    function emitLineQuad(ax, ay, bx, by, z, minRes) {
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
      const need = lineVertCount * 4 + 24;
      lineGeom = growFloat(lineGeom, need);
      const verts = [x0, y0, -0.85, x1, y1, 0.85, x2, y2, -0.85, x2, y2, -0.85, x1, y1, 0.85, x3, y3, 0.85];
      for (let i = 0; i < 6; i++) {
        const o = lineVertCount * 4;
        lineGeom[o] = verts[i * 3];
        lineGeom[o + 1] = verts[i * 3 + 1];
        lineGeom[o + 2] = z;
        lineGeom[o + 3] = verts[i * 3 + 2];
        lineVertCount++;
      }
    }

    function considerDeflect(wx, wy, wz, sr, z, px) {
      if (z < MIN_LENS_Z || z > 90) return;
      const mass = sr * STAR_COMPACT;
      if ((mass * 48) / z <= px * 0.5) return;
      if (deflectCount < MAX_DEFLECT) {
        const o = deflectCount * 4;
        starData[o] = wx;
        starData[o + 1] = wy;
        starData[o + 2] = wz;
        starData[o + 3] = sr;
        deflectCount++;
        return;
      }
      let far = 0;
      let farZ = -1;
      for (let i = 0; i < MAX_DEFLECT; i++) {
        const dx = starData[i * 4] - camX;
        const dy = starData[i * 4 + 1] - camY;
        const dz = starData[i * 4 + 2] - camZ;
        const zi = dx * dx + dy * dy + dz * dz;
        if (zi > farZ) {
          farZ = zi;
          far = i;
        }
      }
      const dist2 = (wx - camX) ** 2 + (wy - camY) ** 2 + (wz - camZ) ** 2;
      if (dist2 < farZ) {
        const o = far * 4;
        starData[o] = wx;
        starData[o + 1] = wy;
        starData[o + 2] = wz;
        starData[o + 3] = sr;
      }
    }

    function packScene(now, b, renderW, renderH, radius) {
      const t = now * 0.001;
      const minRes = Math.min(renderW, renderH);
      const px = 1 / minRes;
      const holeZ = -camX * b.fx - camY * b.fy - camZ * b.fz;
      starVertCount = 0;
      starFrontCount = 0;
      lineVertCount = 0;
      deflectCount = 0;
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
        emitClusterDust(c, ox, oy, oz, b, minRes, renderW, renderH, holeZ);
        const start = pointStart[c];
        let prevOn = false;
        let prevAx = 0;
        let prevAy = 0;
        let prevZ = 0;
        for (let i = 0; i < npts; i++) {
          const pi = start + i;
          const wx = ox + pointX[pi];
          const wy = oy + pointY[pi];
          const wz = oz + pointZ[pi];
          const sr = pointR[pi];
          if (!starInView(b, wx, wy, wz, sr, minRes, renderW, renderH)) {
            prevOn = false;
            continue;
          }
          const dx = wx - camX;
          const dy = wy - camY;
          const dz = wz - camZ;
          const z = dx * b.fx + dy * b.fy + dz * b.fz;
          const spx = (dx * b.rx + dy * b.ry + dz * b.rz) / z;
          const spy = (dx * b.ux + dy * b.uy + dz * b.uz) / z;
          const ang = sr / Math.max(z, sr * 0.35);
          const behind = z > Math.max(holeZ, LENS_Z_MIN);
          if (!behind) emitStarTo("starGeomFront", "front", spx, spy, z, ang, 0);
          else emitStarQuad(spx, spy, z, ang, 0);
          considerDeflect(wx, wy, wz, sr, z, px);
          if (prevOn) {
            emitLineQuad(prevAx, prevAy, spx, spy, Math.min(prevZ, z), minRes);
          }
          prevOn = true;
          prevAx = spx;
          prevAy = spy;
          prevZ = z;
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
        deflectCount: gl.getUniformLocation(program, "uDeflectCount"),
        viewHalf: gl.getUniformLocation(program, "uViewHalf"),
      },
    };

    const starAttribs = {
      corner: gl.getAttribLocation(starProgram, "aCorner"),
      star: gl.getAttribLocation(starProgram, "aStar"),
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
    };
    const lensUniforms = {
      field: gl.getUniformLocation(lensProgram, "uField"),
      mask: gl.getUniformLocation(lensProgram, "uMask"),
      resolution: gl.getUniformLocation(lensProgram, "resolution"),
      progress: gl.getUniformLocation(lensProgram, "progress"),
      schwarzschildRadius: gl.getUniformLocation(lensProgram, "schwarzschildRadius"),
      camPos: gl.getUniformLocation(lensProgram, "camPos"),
      camFwd: gl.getUniformLocation(lensProgram, "camFwd"),
      camRight: gl.getUniformLocation(lensProgram, "camRight"),
      camUp: gl.getUniformLocation(lensProgram, "camUp"),
    };
    const starBuffer = gl.createBuffer();
    const lineBuffer = gl.createBuffer();

    const blitAttrib = gl.getAttribLocation(blitProgram, "aVertexPosition");
    const lensAttrib = gl.getAttribLocation(lensProgram, "aVertexPosition");
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
    let compositeFb = null;
    let fieldFb = null;
    let sceneTexture = null;
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
        gl.deleteFramebuffer(compositeFb);
        gl.deleteFramebuffer(fieldFb);
        gl.deleteTexture(sceneTexture);
        gl.deleteTexture(compositeTexture);
        gl.deleteTexture(fieldTexture);
        gl.deleteTexture(displayTexture);
      }
      destroyPbos();

      sceneTexture = createTexture(width, height, sceneInternal, sceneFormat);
      framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, sceneTexture, 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        gl.deleteTexture(sceneTexture);
        sceneTexture = createTexture(width, height, gl.RGBA, gl.RGBA);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, sceneTexture, 0);
      }
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

    function disableAttrib(loc) {
      if (loc >= 0) gl.disableVertexAttribArray(loc);
    }

    function bindFullscreen() {
      disableAttrib(starAttribs.corner);
      disableAttrib(starAttribs.star);
      disableAttrib(starAttribs.kind);
      disableAttrib(lineAttribs.pos);
      disableAttrib(lineAttribs.meta);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffers.position);
      gl.vertexAttribPointer(programInfo.attribLocations.vertexPosition, 2, gl.FLOAT, false, 0, 0);
      gl.enableVertexAttribArray(programInfo.attribLocations.vertexPosition);
      if (blitAttrib !== programInfo.attribLocations.vertexPosition) {
        gl.vertexAttribPointer(blitAttrib, 2, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(blitAttrib);
      }
    }

    function drawStarGeom(geom, count, renderWidth, renderHeight, b, useLens, useMask) {
      if (!count) return;
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
      gl.bindBuffer(gl.ARRAY_BUFFER, starBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, geom.subarray(0, count * 7), gl.STREAM_DRAW);
      const stride = 28;
      gl.vertexAttribPointer(starAttribs.corner, 2, gl.FLOAT, false, stride, 0);
      gl.enableVertexAttribArray(starAttribs.corner);
      gl.vertexAttribPointer(starAttribs.star, 4, gl.FLOAT, false, stride, 8);
      gl.enableVertexAttribArray(starAttribs.star);
      gl.vertexAttribPointer(starAttribs.kind, 1, gl.FLOAT, false, stride, 24);
      gl.enableVertexAttribArray(starAttribs.kind);
      gl.drawArrays(gl.TRIANGLES, 0, count);
    }

    function drawLineGeom(renderWidth, renderHeight, b, useMask) {
      if (!lineVertCount) return;
      disableAttrib(starAttribs.corner);
      disableAttrib(starAttribs.star);
      disableAttrib(starAttribs.kind);
      gl.useProgram(lineProgram);
      gl.uniform2f(lineUniforms.resolution, renderWidth, renderHeight);
      gl.uniform1f(lineUniforms.progress, easedProgress);
      gl.uniform3f(lineUniforms.camPos, camX, camY, camZ);
      gl.uniform3f(lineUniforms.camFwd, b.fx, b.fy, b.fz);
      gl.uniform1i(lineUniforms.mask, 0);
      gl.uniform1f(lineUniforms.useMask, useMask ? 1 : 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, lineBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, lineGeom.subarray(0, lineVertCount * 4), gl.STREAM_DRAW);
      const stride = 16;
      gl.vertexAttribPointer(lineAttribs.pos, 2, gl.FLOAT, false, stride, 0);
      gl.enableVertexAttribArray(lineAttribs.pos);
      gl.vertexAttribPointer(lineAttribs.meta, 2, gl.FLOAT, false, stride, 8);
      gl.enableVertexAttribArray(lineAttribs.meta);
      gl.drawArrays(gl.TRIANGLES, 0, lineVertCount);
    }

    function blitScene(renderWidth, renderHeight) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, compositeFb);
      gl.viewport(0, 0, renderWidth, renderHeight);
      gl.disable(gl.BLEND);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, sceneTexture);
      gl.useProgram(blitProgram);
      bindFullscreen();
      gl.drawArrays(gl.TRIANGLES, 0, 3);
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
        drawLineGeom(renderWidth, renderHeight, b, false);
      }
      blitScene(renderWidth, renderHeight);
      if (!starVertCount && !lineVertCount && !starFrontCount) return;
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      if (starVertCount || lineVertCount) {
        gl.useProgram(lensProgram);
        bindFullscreen();
        if (lensAttrib !== blitAttrib && lensAttrib >= 0) {
          gl.vertexAttribPointer(lensAttrib, 2, gl.FLOAT, false, 0, 0);
          gl.enableVertexAttribArray(lensAttrib);
        }
        gl.uniform2f(lensUniforms.resolution, renderWidth, renderHeight);
        gl.uniform1f(lensUniforms.progress, easedProgress);
        gl.uniform1f(lensUniforms.schwarzschildRadius, schwarzschildRadius);
        gl.uniform3f(lensUniforms.camPos, camX, camY, camZ);
        gl.uniform3f(lensUniforms.camFwd, b.fx, b.fy, b.fz);
        gl.uniform3f(lensUniforms.camRight, b.rx, b.ry, b.rz);
        gl.uniform3f(lensUniforms.camUp, b.ux, b.uy, b.uz);
        gl.uniform1i(lensUniforms.mask, 0);
        gl.uniform1i(lensUniforms.field, 1);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, sceneTexture);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, fieldTexture);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, null);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, sceneTexture);
      }
      drawStarGeom(starGeomFront, starFrontCount, renderWidth, renderHeight, b, false, true);
      gl.disable(gl.BLEND);
    }

    function renderScene(renderWidth, renderHeight, now) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.viewport(0, 0, renderWidth, renderHeight);
      gl.disable(gl.BLEND);
      gl.useProgram(programInfo.program);
      bindFullscreen();
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
      const half = viewHalf(renderWidth, renderHeight, minRes);
      gl.uniform2f(programInfo.uniformLocations.viewHalf, half.x, half.y);
      packScene(now, b, renderWidth, renderHeight, schwarzschildRadius * easedProgress);
      if (deflectCount) {
        gl.uniform4fv(programInfo.uniformLocations.stars, starData.subarray(0, deflectCount * 4));
      }
      gl.uniform1i(programInfo.uniformLocations.deflectCount, deflectCount);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      renderStars(renderWidth, renderHeight, b);
      gl.flush();
    }

    function present(width, height, upload) {
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      if (gl.PIXEL_UNPACK_BUFFER) gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, null);
      bindFullscreen();
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

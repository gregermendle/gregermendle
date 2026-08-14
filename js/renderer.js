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

  function marchStepSource() {
    return `
        stepCount += 1.0;
        if (stepCount > marchBudget) return col * marchGain;
        r = length(p);
        adaptiveStepSize = STEP_SIZE * max(1.0, r * 0.1) * stepMul;
        if (r < influenceRadius) {
          rd = normalize(rd - p * (radius * adaptiveStepSize / (r * r * r)));
        }
        prevP = p;
        p += rd * adaptiveStepSize;
        totalDist += adaptiveStepSize;
        if (prevP.y * p.y <= 0.0 || abs(p.y) < 0.12) {
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
        if (r < radius || totalDist > 100.0 || dot(col, col) > 100.0) return col * marchGain;
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
    uniform vec3 uStars[24];
    uniform vec3 uLineA[20];
    uniform vec3 uLineB[20];
    uniform int uStarCount;
    uniform int uLineCount;
    ${fragOut}
    #define MAX_STEPS ${maxSteps}
    #define WARP_SIZE 0.25
    #define STEP_SIZE ${stepSize.toFixed(2)}
    #define DISK_SIZE ${DISK_SIZE.toFixed(1)}
    #define MAX_STARS 24
    #define MAX_LINES 20

    float hash(vec2 p) {
      return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }

    vec4 rayMarch(vec3 ro, vec3 rd, vec2 uv, float radius) {
      float influenceRadius = max(radius * 50.0, radius * DISK_SIZE + 4.0);
      float b = dot(ro, rd);
      float c = dot(ro, ro) - influenceRadius * influenceRadius;
      float h = b * b - c;
      if (h < 0.0) return vec4(0.0);
      float tExit = -b + sqrt(h);
      if (tExit < 0.0) return vec4(0.0);
      float tEnter = max(-b - sqrt(h), 0.0);
      float lod = smoothstep(10.0, 36.0, length(ro));
      float marchBudget = mix(float(MAX_STEPS), float(MAX_STEPS) * 0.35, lod);
      float stepMul = mix(1.0, 2.4, lod);
      float marchGain = mix(1.0, 0.55, lod);
      float jitter = hash(uv) * STEP_SIZE;
      vec3 p = ro + rd * (tEnter + jitter);
      float r = length(p);
      vec4 col = vec4(0.0);
      float totalDist = tEnter + jitter;
      float tRot = time * 0.3;
      float adaptiveStepSize;
      vec3 prevP;
      float stepCount = 0.0;

      ${loop}
      return col * marchGain;
    }

    vec2 projectStar(vec3 w, vec3 ro) {
      vec3 d = w - ro;
      float z = dot(d, camFwd);
      if (z <= 1e-4) return vec2(100.0);
      return vec2(dot(d, camRight) / z, dot(d, camUp) / z);
    }

    float starField(vec2 uv, vec3 ro, float radius, float minRes) {
      float glow = 0.0;
      for (int i = 0; i < MAX_STARS; i++) {
        float on = step(float(i) + 0.5, float(uStarCount));
        vec3 w = uStars[i];
        vec3 toS = w - ro;
        float denom = dot(toS, toS);
        float t = -dot(ro, toS) / max(denom, 1e-5);
        vec3 closest = ro + clamp(t, 0.0, 1.0) * toS;
        on *= step(radius * radius, dot(closest, closest));
        float d = length(uv - projectStar(w, ro)) * minRes;
        glow += on * (exp(-d * d * 0.55) * 1.2 + exp(-d * 0.22) * 0.14);
      }
      for (int i = 0; i < MAX_LINES; i++) {
        float on = step(float(i) + 0.5, float(uLineCount));
        vec2 a = projectStar(uLineA[i], ro);
        vec2 b = projectStar(uLineB[i], ro);
        vec2 pa = uv - a;
        vec2 ba = b - a;
        float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-5), 0.0, 1.0);
        float d = length(pa - ba * h) * minRes;
        glow += on * (1.0 - smoothstep(0.35, 0.85, d)) * 0.22;
      }
      return glow;
    }

    void main() {
      float minRes = min(resolution.x, resolution.y);
      vec2 uv = (gl_FragCoord.xy - 0.5 * resolution.xy) / minRes;
      vec2 aspect = resolution.xy / minRes;
      vec3 ro = camPos;
      vec3 rd = normalize(uv.x * camRight + uv.y * camUp + camFwd);
      float radius = schwarzschildRadius * progress;
      vec4 col = rayMarch(ro, rd, uv * aspect * 5.0, radius) * progress;
      col += vec4(vec3(starField(uv, ro, radius, minRes) * progress), 0.0);
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

    const MAX_STARS = 24;
    const MAX_LINES = 20;
    const starData = new Float32Array(MAX_STARS * 3);
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

    function packClusters(now) {
      const t = now * 0.001;
      starCount = 0;
      lineCount = 0;
      for (let c = 0; c < clusters.length; c++) {
        const cluster = clusters[c];
        const pts = cluster.points || [];
        if (!pts.length || starCount >= MAX_STARS) break;
        const origin = cluster.origin || [0, 0, 0];
        const glide = cluster.glide || {};
        const amp = glide.amp || [0, 0, 0];
        const speed = glide.speed || 0;
        const phase = glide.phase || 0;
        const ox = origin[0] + amp[0] * Math.sin(t * speed + phase);
        const oy = origin[1] + amp[1] * Math.sin(t * speed * 0.83 + phase + 1.1);
        const oz = origin[2] + amp[2] * Math.cos(t * speed * 0.71 + phase);
        for (let i = 0; i < pts.length && starCount < MAX_STARS; i++) {
          const p = pts[i];
          const o = starCount * 3;
          starData[o] = ox + (p[0] || 0);
          starData[o + 1] = 0;
          starData[o + 2] = oz + (p[2] || 0);
          if (i > 0 && lineCount < MAX_LINES) {
            const a = (starCount - 1) * 3;
            const b = starCount * 3;
            const lo = lineCount * 3;
            lineAData[lo] = starData[a];
            lineAData[lo + 1] = starData[a + 1];
            lineAData[lo + 2] = starData[a + 2];
            lineBData[lo] = starData[b];
            lineBData[lo + 1] = starData[b + 1];
            lineBData[lo + 2] = starData[b + 2];
            lineCount++;
          }
          starCount++;
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
    const mapBounds = 40;
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
      const r = Math.hypot(camX, camY, camZ);
      const min = schwarzschildRadius * 3 + 1.2;
      if (r < min && r > 1e-5) {
        const s = min / r;
        camX *= s;
        camY *= s;
        camZ *= s;
      }
      camX = Math.max(-mapBounds, Math.min(mapBounds, camX));
      camY = Math.max(0.4, Math.min(24, camY));
      camZ = Math.max(-mapBounds, Math.min(mapBounds, camZ));
    }

    function emitHud() {
      if (!emit || displayWidth <= 0 || displayHeight <= 0) return;
      emit({
        type: "hud",
        map: {
          camX,
          camZ,
          yaw: camYaw,
          bounds: mapBounds,
          stars: Array.from({ length: starCount }, (_, i) => [
            starData[i * 3],
            starData[i * 3 + 2],
          ]),
        },
      });
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
      packClusters(now);
      gl.uniform3fv(programInfo.uniformLocations.stars, starData);
      gl.uniform3fv(programInfo.uniformLocations.lineA, lineAData);
      gl.uniform3fv(programInfo.uniformLocations.lineB, lineBData);
      gl.uniform1i(programInfo.uniformLocations.starCount, starCount);
      gl.uniform1i(programInfo.uniformLocations.lineCount, lineCount);
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
      const wishX = b.fx * (keyF - keyB) + b.rx * (keyR - keyL) + b.ux * (keyU - keyD);
      const wishY = b.fy * (keyF - keyB) + b.ry * (keyR - keyL) + b.uy * (keyU - keyD);
      const wishZ = b.fz * (keyF - keyB) + b.rz * (keyR - keyL) + b.uz * (keyU - keyD);
      const accel = 0.0004 * timeScale;
      velX += wishX * accel;
      velY += wishY * accel;
      velZ += wishZ * accel;
      const damp = Math.pow(0.993, timeScale);
      velX *= damp;
      velY *= damp;
      velZ *= damp;
      const speed = Math.hypot(velX, velY, velZ);
      if (speed > 0.007) {
        const s = 0.007 / speed;
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
      renderScene(renderWidth, renderHeight, now);
      emitHud();
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
      },
      setNav(x, z) {
        camX = x;
        camZ = z;
        lookYaw = Math.atan2(-camX, -camZ);
        camYaw = lookYaw;
        keepOut();
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

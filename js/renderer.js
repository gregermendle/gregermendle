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
        r = length(p);
        adaptiveStepSize = STEP_SIZE * max(1.0, r * 0.1);
        if (r >= influenceRadius) {
          if (dot(rd, p) > 0.0) return col;
          adaptiveStepSize = max(adaptiveStepSize, (r - influenceRadius) * 0.25);
        } else {
          rd = normalize(rd - p * (radius * adaptiveStepSize / (r * r * r)));
        }
        p += rd * adaptiveStepSize;
        totalDist += adaptiveStepSize;
        if (abs(p.y) < 0.12) {
          float diskRadius = length(p.xz);
          if (diskRadius > radius) {
            float d = (diskRadius - radius) / radius;
            float innerTemp = smoothstep(DISK_SIZE * 0.25, DISK_SIZE * 0.5, d);
            float outerTemp = smoothstep(DISK_SIZE * 0.5, DISK_SIZE, d);
            float gray = mix(1.0, mix(0.4, 0.1, outerTemp), innerTemp);
            vec4 diskCol = vec4(vec3(gray * exp(-d * (1.7 / DISK_SIZE))), 1.0) * exp(-totalDist * 0.12);
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
    #define DISK_SIZE ${DISK_SIZE.toFixed(1)}

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
    let displayWidth = 0;
    let displayHeight = 0;
    let hoverPaused = false;
    let spin = 0;
    let prevHudNow = 0;

    function projectWorld(wx, wy, wz, minRes, rox, roy, roz) {
      const dx = wx - rox;
      const dy = wy - roy;
      const dz = wz - roz;
      const len = Math.hypot(dx, dy, dz);
      const ndy = dy / len;
      const ndz = dz / len;
      const k = (-0.4 * ndy + Math.sqrt(0.16 * ndy * ndy + 3.84)) * 0.5;
      const sz = k * ndz;
      if (sz <= 1e-5) return null;
      return {
        x: ((k * (dx / len)) / sz) * minRes + 0.5 * displayWidth,
        y: displayHeight - (((k * ndy + 0.2) / sz) * minRes + 0.5 * displayHeight),
        len,
      };
    }

    const ORBIT_COUNT = 8;

    function occluded(wx, wy, wz, rox, roy, roz, radius) {
      const dx = wx - rox;
      const dy = wy - roy;
      const dz = wz - roz;
      const denom = dx * dx + dy * dy + dz * dz;
      if (denom < 1e-8) return false;
      const t = -(rox * dx + roy * dy + roz * dz) / denom;
      if (t <= 0.04 || t >= 0.96) return false;
      const cx = rox + t * dx;
      const cy = roy + t * dy;
      const cz = roz + t * dz;
      return cx * cx + cy * cy + cz * cz < radius * radius;
    }

    function emitHud(now) {
      if (!emit || displayWidth <= 0 || displayHeight <= 0) return;
      const minRes = Math.min(displayWidth, displayHeight);
      const rox = -1 + (mouseX - 0.5) * (displayWidth / minRes) * 2;
      const roy = 2 + (mouseY - 0.5) * (displayHeight / minRes) * 2;
      const roz = -10;
      const center = projectWorld(0, 0, 0, minRes, rox, roy, roz);
      const top = projectWorld(0, schwarzschildRadius, 0, minRes, rox, roy, roz);
      if (!center || !top) {
        emit({ type: "hud", a: 0 });
        return;
      }
      const holeR = Math.hypot(top.x - center.x, top.y - center.y);
      if (!prevHudNow) prevHudNow = now;
      if (!hoverPaused) spin += (now - prevHudNow) * 0.001 * 0.3;
      prevHudNow = now;
      const orbits = [];
      for (let i = 0; i < ORBIT_COUNT; i++) {
        const angle = spin + (i / ORBIT_COUNT) * Math.PI * 2;
        const rad = schwarzschildRadius * easedProgress * (DISK_SIZE + 2.5);
        const wx = Math.cos(angle) * rad;
        const wz = Math.sin(angle) * rad;
        const pt = projectWorld(wx, 0, wz, minRes, rox, roy, roz);
        if (!pt) {
          orbits.push({ a: 0 });
          continue;
        }
        const behind = occluded(wx, 0, wz, rox, roy, roz, schwarzschildRadius);
        const depth = Math.min(1.2, Math.max(0.55, 10 / pt.len));
        orbits.push({
          x: pt.x,
          y: pt.y,
          s: easedProgress * depth,
          a: easedProgress * (behind ? 0.12 : Math.min(1, 12 / pt.len)),
          z: Math.round(2000 - pt.len * 40),
        });
      }
      emit({
        type: "hud",
        x: top.x,
        y: top.y - holeR * 0.55 - 28,
        s: easedProgress * (10 / top.len),
        a: easedProgress,
        orbits,
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
      emitHud(now);
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
      setHidden(value) {
        hidden = value;
        if (hidden && emit) emit({ type: "hud", a: 0 });
        start();
      },
      setRunning(value) {
        running = value;
        if (!running && emit) emit({ type: "hud", a: 0 });
        start();
      },
      setHover(value) {
        hoverPaused = value;
      },
    };
  }

  scope.createRenderer = createRenderer;
})(self);

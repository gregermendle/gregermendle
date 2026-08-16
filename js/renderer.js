(function (scope) {
  const NOISE_SIZE = 256;
  const NOISE3_SIZE = 128;
  const BLUE_SIZE = 128;
  const SIM_SIZE = 256;
  const SIM_MIP = 8;
  const JACOBI_STEPS = 40;
  const MARCH_SCALE = 0.7;
  const MAX_MARCH = 900;
  const DITHER_LEVELS = 2;

  const CAM_Y = 22.0;
  const SPREAD = 0.42;
  const CLOUD_BOT = 0.0;
  const CLOUD_TOP = 1.1;
  const DOMAIN = 13.0;

  function loadShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error(gl.getShaderInfoLog(shader) || "shader compile failed");
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  function mulberry32(seed) {
    return () => {
      seed |= 0;
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function makeNoiseData() {
    const rng = mulberry32(0x6e6f6973);
    const data = new Uint8Array(NOISE_SIZE * NOISE_SIZE * 4);
    for (let i = 0; i < NOISE_SIZE * NOISE_SIZE; i++) {
      const j = i * 4;
      data[j] = (rng() * 256) | 0;
      data[j + 1] = (rng() * 256) | 0;
      data[j + 2] = (rng() * 256) | 0;
      data[j + 3] = 255;
    }
    return data;
  }

  function makeBlueNoiseData() {
    const n = BLUE_SIZE * BLUE_SIZE;
    const values = new Float32Array(n);
    const rng = mulberry32(0x626c7565);
    for (let i = 0; i < n; i++) values[i] = rng();
    const tmp = new Float32Array(n);
    for (let pass = 0; pass < 8; pass++) {
      tmp.set(values);
      for (let y = 0; y < BLUE_SIZE; y++) {
        for (let x = 0; x < BLUE_SIZE; x++) {
          let acc = 0;
          let wsum = 0;
          for (let oy = -2; oy <= 2; oy++) {
            for (let ox = -2; ox <= 2; ox++) {
              if (!ox && !oy) continue;
              const w = 1 / (ox * ox + oy * oy);
              const ix = (x + ox + BLUE_SIZE) % BLUE_SIZE;
              const iy = (y + oy + BLUE_SIZE) % BLUE_SIZE;
              acc += tmp[iy * BLUE_SIZE + ix] * w;
              wsum += w;
            }
          }
          values[y * BLUE_SIZE + x] = tmp[y * BLUE_SIZE + x] - 0.85 * (acc / wsum - 0.5);
        }
      }
    }
    const ranked = Array.from(values, (v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
    const data = new Uint8Array(n * 4);
    for (let r = 0; r < n; r++) {
      const i = ranked[r][1] * 4;
      const v = r < 8 ? 255 : (r / (n - 1)) * 255;
      data[i] = data[i + 1] = data[i + 2] = v;
      data[i + 3] = 255;
    }
    return data;
  }

  function createRenderer(canvas) {
    const gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: false,
      depth: false,
      preserveDrawingBuffer: true,
      powerPreference: "high-performance",
    });
    if (!gl) return null;
    gl.getExtension("EXT_color_buffer_float");
    gl.getExtension("EXT_color_buffer_half_float");

    function makeProgram(vsSrc, fsSrc, uniformNames) {
      const p = gl.createProgram();
      const v = loadShader(gl, gl.VERTEX_SHADER, vsSrc);
      const f = loadShader(gl, gl.FRAGMENT_SHADER, fsSrc);
      if (!v || !f) return null;
      gl.attachShader(p, v);
      gl.attachShader(p, f);
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
        console.error(gl.getProgramInfoLog(p) || "program link failed");
        return null;
      }
      const u = {};
      for (const name of uniformNames) u[name] = gl.getUniformLocation(p, name);
      return { p, u, aPos: gl.getAttribLocation(p, "aPos") };
    }

    const blitVs = `#version 300 es
in vec4 aPos;
out vec2 vUv;
void main() {
  gl_Position = aPos;
  vUv = aPos.xy * 0.5 + 0.5;
}`;

    const simLib = `
const float SIM = ${SIM_SIZE.toFixed(1)};
const float TEX = 1.0 / SIM;

const float TREF = 0.60;
const float CORIOLIS = 0.030;
const float JET_AMP = 0.10;
const float JET_RELAX = 0.0030;
const float VEL_MAX = 3.0;

const float CONV_BUOY = 0.024;
const float CONV_RAIN = 0.028;
const float BUOY_T = 1.0;
const float BUOY_QC = 0.9;

const float DIV_SCALE = 0.0004;
const float ADIA = 0.0012;
const float MOIST_CONV = 0.004;
const float COND_RATE = 0.30;
const float REVAP_RATE = 0.10;
const float LATENT = 0.45;

const float RAIN_THRESH = 0.18;
const float RAIN_RATE = 0.09;
const float RAIN_DECAY = 0.92;
const float RAIN_COOL = 0.0016;

const float TRIGGER_T = 0.0026;
const float TRIGGER_Q = 0.0020;

const float SURF_EVAP = 0.010;
const float SURF_RH = 0.90;
const float RAD_RELAX = 0.008;

vec2 jetAt(vec2 uv) {
  return vec2(JET_AMP * sin(6.2831853 * uv.y), 0.0);
}

float qsat(float t) {
  return 0.42 * exp(3.2 * (t - TREF));
}

float tEquilibrium(vec2 uv) {
  return 0.615 + 0.035 * cos(6.2831853 * uv.y);
}

float divergenceAt(sampler2D vel, vec2 uv) {
  float l = texture(vel, uv - vec2(TEX, 0.0)).x;
  float r = texture(vel, uv + vec2(TEX, 0.0)).x;
  float b = texture(vel, uv - vec2(0.0, TEX)).y;
  float t = texture(vel, uv + vec2(0.0, TEX)).y;
  return 0.5 * ((r - l) + (t - b));
}
`;

    const noiseLib = `
float lattice(vec2 cell, float period, vec2 off, sampler2D tex) {
  vec2 c = mod(cell, period) + off;
  return textureLod(tex, (c + 0.5) / 256.0, 0.0).x;
}

float tileNoise(vec2 uv, float period, vec2 off, sampler2D tex) {
  vec2 p = uv * period;
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float s00 = lattice(i, period, off, tex);
  float s10 = lattice(i + vec2(1.0, 0.0), period, off, tex);
  float s01 = lattice(i + vec2(0.0, 1.0), period, off, tex);
  float s11 = lattice(i + vec2(1.0, 1.0), period, off, tex);
  return mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
}

float tileFbm(vec2 uv, float base, sampler2D tex) {
  float f = 0.0;
  float a = 0.5;
  float period = base;
  vec2 off = vec2(0.0);
  for (int i = 0; i < 4; i++) {
    f += a * tileNoise(uv, period, off, tex);
    period *= 2.0;
    a *= 0.5;
    off += vec2(37.0, 71.0);
  }
  return f;
}
`;

    const noiseGenFs = `#version 300 es
precision highp float;
uniform float uZ;
in vec2 vUv;
out vec4 fragColor;

float hash13(vec3 p) {
  p = fract(p * 0.1031);
  p += dot(p, p.yzx + 33.33);
  return fract((p.x + p.y) * p.z);
}

vec3 hash33(vec3 p) {
  return vec3(hash13(p), hash13(p + 19.19), hash13(p + 47.71));
}

float valueTile(vec3 p, float cells) {
  vec3 pc = p * cells;
  vec3 i = floor(pc);
  vec3 f = fract(pc);
  f = f * f * (3.0 - 2.0 * f);
  float sum = 0.0;
  for (int dz = 0; dz < 2; dz++) {
    for (int dy = 0; dy < 2; dy++) {
      for (int dx = 0; dx < 2; dx++) {
        vec3 o = vec3(float(dx), float(dy), float(dz));
        float v = hash13(mod(i + o, cells) + 0.5);
        vec3 w = mix(1.0 - f, f, o);
        sum += v * w.x * w.y * w.z;
      }
    }
  }
  return sum;
}

float worleyTile(vec3 p, float cells) {
  vec3 pc = p * cells;
  vec3 i = floor(pc);
  vec3 f = fract(pc);
  float md = 4.0;
  for (int dz = -1; dz <= 1; dz++) {
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        vec3 o = vec3(float(dx), float(dy), float(dz));
        vec3 d = o + hash33(mod(i + o, cells) + 0.5) - f;
        md = min(md, dot(d, d));
      }
    }
  }
  return 1.0 - clamp(sqrt(md), 0.0, 1.0);
}

void main() {
  vec3 p = vec3(vUv, uZ);
  float perlin = 0.5 * valueTile(p, 4.0)
    + 0.26 * valueTile(p, 8.0)
    + 0.14 * valueTile(p, 16.0)
    + 0.10 * valueTile(p, 32.0);
  float w3 = worleyTile(p, 3.0);
  float w6 = worleyTile(p, 6.0);
  float w12 = worleyTile(p, 12.0);
  float shape = clamp(perlin * 0.62 + w3 * 0.38, 0.0, 1.0);
  fragColor = vec4(shape, w3, w6, w12);
}`;

    const thermInitFs = `#version 300 es
precision highp float;
uniform sampler2D uNoise;
in vec2 vUv;
out vec4 fragColor;
${simLib}
${noiseLib}
void main() {
  float t = tEquilibrium(vUv) + 0.05 * (tileFbm(vUv, 5.0, uNoise) - 0.47);
  float humid = tileFbm(vUv + 3.1, 4.0, uNoise);
  float qv = qsat(t) * (0.55 + 0.45 * humid);
  fragColor = vec4(qv, 0.0, t, 0.0);
}`;

    const velInitFs = `#version 300 es
precision highp float;
uniform sampler2D uNoise;
in vec2 vUv;
out vec4 fragColor;
${simLib}
${noiseLib}
void main() {
  float jet = 0.18 * sin(6.2831853 * vUv.y);
  float wobble = 1.1 * (tileFbm(vUv + 7.3, 4.0, uNoise) - 0.47);
  float wobble2 = 1.1 * (tileFbm(vUv + 19.7, 4.0, uNoise) - 0.47);
  fragColor = vec4(jet + wobble, wobble2, 0.0, 1.0);
}`;

    const impulseThermFs = `#version 300 es
precision highp float;
uniform sampler2D uTherm;
uniform vec2 uPoint;
uniform float uRadius;
uniform float uHeat;
uniform float uMoist;
in vec2 vUv;
out vec4 fragColor;
${simLib}
void main() {
  vec4 s = texture(uTherm, vUv);
  vec2 rel = vUv - uPoint;
  rel -= round(rel);
  float w = exp(-dot(rel, rel) / max(uRadius * uRadius, 1e-6));
  s.z = clamp(s.z + uHeat * w, 0.0, 1.4);
  s.x = clamp(s.x + uMoist * w, 0.0, 2.0);
  fragColor = s;
}`;

    const impulseVelFs = `#version 300 es
precision highp float;
uniform sampler2D uVel;
uniform vec2 uPoint;
uniform vec2 uDir;
uniform float uRadius;
uniform float uForce;
uniform float uSpin;
uniform float uConverge;
in vec2 vUv;
out vec4 fragColor;
${simLib}
void main() {
  vec2 vel = texture(uVel, vUv).xy;
  vec2 rel = vUv - uPoint;
  rel -= round(rel);
  float w = exp(-dot(rel, rel) / max(uRadius * uRadius, 1e-6));
  vel += uDir * uForce * w;
  vel += vec2(-rel.y, rel.x) * (uSpin * w / max(uRadius, 1e-6));
  vel -= normalize(rel + vec2(1e-6)) * (uConverge * w);
  float speed = length(vel);
  if (speed > VEL_MAX) vel *= VEL_MAX / speed;
  fragColor = vec4(vel, 0.0, 1.0);
}`;

    const advectVelFs = `#version 300 es
precision highp float;
uniform sampler2D uVel;
in vec2 vUv;
out vec4 fragColor;
${simLib}
void main() {
  vec2 here = texture(uVel, vUv).xy;
  vec2 vel = texture(uVel, vUv - here * TEX).xy;

  float c = cos(CORIOLIS);
  float s = sin(CORIOLIS);
  vel = vec2(c * vel.x + s * vel.y, -s * vel.x + c * vel.y);
  vel += (jetAt(vUv) - vel) * JET_RELAX;
  vel *= 0.9995;

  float speed = length(vel);
  if (speed > VEL_MAX) vel *= VEL_MAX / speed;
  fragColor = vec4(vel, 0.0, 1.0);
}`;

    const advectThermFs = `#version 300 es
precision highp float;
uniform sampler2D uTherm;
uniform sampler2D uVel;
uniform sampler2D uNoise;
uniform float uTime;
in vec2 vUv;
out vec4 fragColor;
${simLib}
${noiseLib}
void main() {
  vec2 vel = texture(uVel, vUv).xy;
  vec2 src = vUv - vel * TEX;

  vec4 s = texture(uTherm, src);
  vec4 blur = 0.25 * (
    texture(uTherm, src + vec2(TEX, 0.0)) +
    texture(uTherm, src - vec2(TEX, 0.0)) +
    texture(uTherm, src + vec2(0.0, TEX)) +
    texture(uTherm, src - vec2(0.0, TEX))
  );
  s = mix(s, blur, 0.03);

  float qv = s.x;
  float qc = s.y;
  float t = s.z;
  float rain = s.w;

  float div = divergenceAt(uVel, vUv);
  float w = clamp(-div / DIV_SCALE, -3.0, 3.0);
  t -= ADIA * w;
  qv += MOIST_CONV * w * qv;

  float teq = tEquilibrium(vUv);
  t += RAD_RELAX * (teq - t);

  float trigger = tileFbm(vUv + vec2(uTime * 0.011, uTime * 0.006), 14.0, uNoise) - 0.47;
  t += TRIGGER_T * trigger;
  qv += TRIGGER_Q * trigger;

  float qs = qsat(t);
  float speed = length(vel);
  qv += SURF_EVAP * (1.0 + 0.4 * speed) * max(qs * SURF_RH - qv, 0.0);
  qs = qsat(t);
  float excess = qv - qs;
  float cond = excess > 0.0 ? excess * COND_RATE : -min(qc, -excess * REVAP_RATE);
  qv -= cond;
  qc += cond;
  t += LATENT * cond;

  float fall = max(qc - RAIN_THRESH, 0.0) * RAIN_RATE;
  qc -= fall;
  rain = rain * RAIN_DECAY + fall * 4.0;
  t -= RAIN_COOL * rain;
  qv += rain * 0.003;

  fragColor = vec4(
    clamp(qv, 0.0, 2.0),
    clamp(qc, 0.0, 1.6),
    clamp(t, 0.0, 1.4),
    clamp(rain, 0.0, 2.0)
  );
}`;

    const copyFs = `#version 300 es
precision highp float;
uniform sampler2D uSrc;
in vec2 vUv;
out vec4 fragColor;
void main() {
  fragColor = texture(uSrc, vUv);
}`;

    const divergenceFs = `#version 300 es
precision highp float;
uniform sampler2D uVel;
uniform sampler2D uTherm;
uniform sampler2D uMean;
in vec2 vUv;
out vec4 fragColor;
${simLib}
void main() {
  float div = divergenceAt(uVel, vUv);
  vec4 s = texture(uTherm, vUv);
  vec4 avg = textureLod(uMean, vec2(0.5), ${SIM_MIP.toFixed(1)});
  float buoy = (s.z - avg.z) * BUOY_T + (s.y - avg.y) * BUOY_QC;
  float target = -CONV_BUOY * buoy + CONV_RAIN * (s.w - avg.w);
  fragColor = vec4(div - target, 0.0, 0.0, 1.0);
}`;

    const jacobiFs = `#version 300 es
precision highp float;
uniform sampler2D uPressure;
uniform sampler2D uDiv;
in vec2 vUv;
out vec4 fragColor;
${simLib}
void main() {
  float l = texture(uPressure, vUv - vec2(TEX, 0.0)).x;
  float r = texture(uPressure, vUv + vec2(TEX, 0.0)).x;
  float b = texture(uPressure, vUv - vec2(0.0, TEX)).x;
  float t = texture(uPressure, vUv + vec2(0.0, TEX)).x;
  float rhs = texture(uDiv, vUv).x;
  fragColor = vec4(0.25 * (l + r + b + t - rhs), 0.0, 0.0, 1.0);
}`;

    const projectFs = `#version 300 es
precision highp float;
uniform sampler2D uVel;
uniform sampler2D uPressure;
in vec2 vUv;
out vec4 fragColor;
${simLib}
void main() {
  float l = texture(uPressure, vUv - vec2(TEX, 0.0)).x;
  float r = texture(uPressure, vUv + vec2(TEX, 0.0)).x;
  float b = texture(uPressure, vUv - vec2(0.0, TEX)).x;
  float t = texture(uPressure, vUv + vec2(0.0, TEX)).x;
  vec2 vel = texture(uVel, vUv).xy - 0.5 * vec2(r - l, t - b);
  float speed = length(vel);
  if (speed > VEL_MAX) vel *= VEL_MAX / speed;
  fragColor = vec4(vel, 0.0, 1.0);
}`;

    const marchFs = `#version 300 es
precision highp float;
uniform highp sampler3D uNoise3;
uniform sampler2D uBlueNoise;
uniform sampler2D uField;
uniform vec2 uResolution;
uniform int uFrame;
in vec2 vUv;
out vec4 fragColor;

#define STEPS 56
#define LIGHT_STEPS 6

const float CAM_Y = ${CAM_Y.toFixed(4)};
const float SPREAD = ${SPREAD.toFixed(4)};
const float BOT = ${CLOUD_BOT.toFixed(4)};
const float TOP = ${CLOUD_TOP.toFixed(4)};
const float DOMAIN = ${DOMAIN.toFixed(4)};
const float SIGMA = 22.0;
const float SIGMA_L = 11.0;
const float SHAPE_TILE = 1.7;
const float DETAIL_TILE = 0.52;
const vec3 SUN_DIR = normalize(vec3(0.66, 0.32, -0.46));
const vec3 SUN_COL = vec3(1.0);
const vec3 AMB_COL = vec3(0.56);
const float PI = 3.14159265;

float worleyFbm(vec4 n) {
  return n.g * 0.625 + n.b * 0.25 + n.a * 0.125;
}

float remap(float v, float lo, float hi) {
  return clamp((v - lo) / max(hi - lo, 1e-4), 0.0, 1.0);
}

vec4 fieldAt(vec3 p) {
  return texture(uField, p.xz / DOMAIN);
}

float topBumps(vec2 xz) {
  float b = 0.52 * texture(uNoise3, vec3(xz.x, 0.0, xz.y) / 2.5).r;
  b += 0.30 * texture(uNoise3, vec3(xz.x, 3.7, xz.y) / 0.95).r;
  b += 0.18 * texture(uNoise3, vec3(xz.x, 8.1, xz.y) / 0.38).r;
  return clamp((b - 0.5) * 3.0 + 0.5, 0.0, 1.0);
}

float cloudDensity(vec3 p, bool cheap) {
  if (p.y < BOT || p.y > TOP) return 0.0;

  vec4 f = fieldAt(p);
  float qc = f.y;
  float breakup = texture(uNoise3, vec3(p.x, 5.3, p.z) / 1.2).r;
  float cover = smoothstep(0.030, 0.175, qc * (0.72 + 0.58 * breakup));
  if (cover < 0.01) return 0.0;

  float bump = topBumps(p.xz);
  float tower = smoothstep(0.05, 0.30, qc);
  float topH = mix(0.12, 1.0, tower) * mix(0.22, 1.0, bump);
  float h = (p.y - BOT) / ((TOP - BOT) * topH);
  if (h > 1.0) return 0.0;

  float grad = smoothstep(0.0, 0.28, h) * smoothstep(1.0, 0.62, h);
  if (grad < 0.01) return 0.0;

  vec4 s = texture(uNoise3, p / SHAPE_TILE);
  float base = mix(s.r, worleyFbm(s), 0.4);
  float d = remap(base * grad, (1.0 - cover) * 0.8, 1.0);

  if (!cheap && d > 0.0) {
    vec4 t = texture(uNoise3, p / DETAIL_TILE + 0.37);
    float amp = 0.18 * (1.0 - h * 0.55) * smoothstep(0.02, 0.30, d);
    d = remap(d, worleyFbm(t) * amp, 1.0);
  }

  return clamp(d, 0.0, 1.0) * 2.6;
}

float lightTau(vec3 p, float jitter) {
  float tau = 0.0;
  float s = 0.032;
  p += SUN_DIR * s * jitter;
  for (int i = 0; i < LIGHT_STEPS; i++) {
    p += SUN_DIR * s;
    tau += cloudDensity(p, true) * s;
    s *= 1.55;
  }
  return tau;
}

float scatter(float tau) {
  float sum = 0.0;
  float att = 1.0;
  float ext = 1.0;
  for (int i = 0; i < 3; i++) {
    sum += att * exp(-tau * SIGMA_L * ext);
    att *= 0.35;
    ext *= 0.28;
  }
  return sum;
}

float hg(float g, float mu) {
  float gg = g * g;
  return (1.0 - gg) / (4.0 * PI * pow(1.0 + gg - 2.0 * g * mu, 1.5));
}

void main() {
  vec2 p = (gl_FragCoord.xy - 0.5 * uResolution) / uResolution.y;
  vec3 ro = vec3(0.0, CAM_Y, 0.0);
  vec3 rd = normalize(vec3(p.x * SPREAD, -1.0, p.y * SPREAD));

  float tTop = (TOP - ro.y) / rd.y;
  float tBot = (BOT - ro.y) / rd.y;
  float stepLen = (tBot - tTop) / float(STEPS);

  float blue = texture(uBlueNoise, gl_FragCoord.xy / 128.0).r;
  float jitter = fract(blue + float(uFrame % 32) * 0.618034);

  float mu = dot(rd, SUN_DIR);
  float phase = 0.62 * hg(0.42, mu) + 0.38 * hg(-0.18, mu);

  vec3 acc = vec3(0.0);
  float trans = 1.0;
  float t = tTop + stepLen * jitter;

  for (int i = 0; i < STEPS; i++) {
    if (trans < 0.02) break;
    vec3 pos = ro + rd * t;
    float den = cloudDensity(pos, false);

    if (den > 0.005) {
      float h = clamp((pos.y - BOT) / (TOP - BOT) * 1.4, 0.0, 1.0);
      float light = scatter(lightTau(pos, jitter));
      float powder = 1.0 - exp(-den * 6.0);
      float rain = fieldAt(pos).w;

      vec3 lit = SUN_COL * (light * phase * 21.0);
      lit += AMB_COL * mix(0.04, 0.18, h);
      lit *= mix(0.40, 1.0, powder);
      lit *= 1.0 - 0.55 * smoothstep(0.05, 0.9, rain) * (1.0 - h);

      float alpha = 1.0 - exp(-den * stepLen * SIGMA);
      acc += trans * alpha * lit;
      trans *= 1.0 - alpha;
    }

    t += stepLen;
  }

  vec3 col = acc;
  col = col / (1.0 + col * 0.55);
  col = max(col, 0.0);
  col = mix(col, col * col * (3.0 - 2.0 * col), 0.15);
  fragColor = vec4(col, 1.0);
}`;

    const fieldViewFs = `#version 300 es
precision highp float;
uniform sampler2D uField;
uniform sampler2D uVel;
in vec2 vUv;
out vec4 fragColor;
void main() {
  vec4 f = texture(uField, vUv);
  vec2 vel = texture(uVel, vUv).xy;
  vec3 col = vec3(0.0);
  col += vec3(1.0) * smoothstep(0.0, 0.5, f.y);
  col += vec3(0.2, 0.4, 1.0) * smoothstep(0.02, 0.8, f.w);
  col += vec3(0.9, 0.3, 0.1) * smoothstep(0.55, 0.85, f.z) * 0.5;
  col += vec3(0.1, 0.8, 0.2) * smoothstep(0.0, 0.9, f.x) * 0.25;
  col += vec3(0.9, 0.9, 0.2) * smoothstep(1.0, 5.0, length(vel)) * 0.3;
  fragColor = vec4(col, 1.0);
}`;

    const accumFs = `#version 300 es
precision highp float;
uniform sampler2D uRaw;
uniform sampler2D uHistory;
uniform float uBlend;
in vec2 vUv;
out vec4 fragColor;
void main() {
  vec3 raw = texture(uRaw, vUv).rgb;
  vec3 hist = texture(uHistory, vUv).rgb;
  fragColor = vec4(mix(hist, raw, uBlend), 1.0);
}`;

    const bicubicFs = `#version 300 es
precision highp float;
uniform sampler2D uCloud;
uniform sampler2D uBlueNoise;
in vec2 vUv;
out vec4 fragColor;

const float LEVELS = ${DITHER_LEVELS.toFixed(1)};
const float BLACK_POINT = 0.03;
const float WHITE_POINT = 0.95;
const float CONTRAST = 1.10;
vec4 cubic(float v) {
  vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
  vec4 s = n * n * n;
  float x = s.x;
  float y = s.y - 4.0 * s.x;
  float z = s.z - 4.0 * s.y + 6.0 * s.x;
  float w = 6.0 - x - y - z;
  return vec4(x, y, z, w) * (1.0 / 6.0);
}
vec4 textureBicubic(sampler2D tex, vec2 uv) {
  vec2 texSize = vec2(textureSize(tex, 0));
  vec2 invTexSize = 1.0 / texSize;
  uv = uv * texSize - 0.5;
  vec2 fxy = fract(uv);
  uv -= fxy;
  vec4 xcubic = cubic(fxy.x);
  vec4 ycubic = cubic(fxy.y);
  vec4 c = uv.xxyy + vec2(-0.5, 1.5).xyxy;
  vec4 s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
  vec4 offset = c + vec4(xcubic.yw, ycubic.yw) / s;
  offset *= invTexSize.xxyy;
  vec4 s0 = texture(tex, offset.xz);
  vec4 s1 = texture(tex, offset.yz);
  vec4 s2 = texture(tex, offset.xw);
  vec4 s3 = texture(tex, offset.yw);
  float sx = s.x / (s.x + s.y);
  float sy = s.z / (s.z + s.w);
  return mix(mix(s3, s2, sx), mix(s1, s0, sx), sy);
}
void main() {
  vec3 col = textureBicubic(uCloud, vUv).rgb;
  float lum = dot(col, vec3(0.299, 0.587, 0.114));
  lum = clamp((lum - BLACK_POINT) / (WHITE_POINT - BLACK_POINT), 0.0, 1.0);
  lum = pow(lum, CONTRAST);

  float threshold = texture(uBlueNoise, gl_FragCoord.xy / 128.0).r;
  float steps = LEVELS - 1.0;
  float v = floor(lum * steps + threshold) / steps;
  fragColor = vec4(vec3(clamp(v, 0.0, 1.0)), 1.0);
}`;

    const noiseGen = makeProgram(blitVs, noiseGenFs, ["uZ"]);
    const thermInit = makeProgram(blitVs, thermInitFs, ["uNoise"]);
    const velInit = makeProgram(blitVs, velInitFs, ["uNoise"]);
    const impulseTherm = makeProgram(blitVs, impulseThermFs, [
      "uTherm",
      "uPoint",
      "uRadius",
      "uHeat",
      "uMoist",
    ]);
    const impulseVel = makeProgram(blitVs, impulseVelFs, [
      "uVel",
      "uPoint",
      "uDir",
      "uRadius",
      "uForce",
      "uSpin",
      "uConverge",
    ]);
    const advectVel = makeProgram(blitVs, advectVelFs, ["uVel"]);
    const advectTherm = makeProgram(blitVs, advectThermFs, [
      "uTherm",
      "uVel",
      "uNoise",
      "uTime",
    ]);
    const copy = makeProgram(blitVs, copyFs, ["uSrc"]);
    const divergence = makeProgram(blitVs, divergenceFs, ["uVel", "uTherm", "uMean"]);
    const jacobi = makeProgram(blitVs, jacobiFs, ["uPressure", "uDiv"]);
    const project = makeProgram(blitVs, projectFs, ["uVel", "uPressure"]);
    const march = makeProgram(blitVs, marchFs, [
      "uNoise3",
      "uBlueNoise",
      "uField",
      "uResolution",
      "uFrame",
    ]);
    const fieldView = makeProgram(blitVs, fieldViewFs, ["uField", "uVel"]);
    const accum = makeProgram(blitVs, accumFs, ["uRaw", "uHistory", "uBlend"]);
    const display = makeProgram(blitVs, bicubicFs, ["uCloud", "uBlueNoise"]);

    const programs = [
      noiseGen,
      thermInit,
      velInit,
      impulseTherm,
      impulseVel,
      advectVel,
      advectTherm,
      copy,
      divergence,
      jacobi,
      project,
      march,
      fieldView,
      accum,
      display,
    ];
    if (programs.some((prog) => !prog)) return null;

    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);

    function makeTex(w, h, repeat, data, wantFloat) {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, repeat ? gl.REPEAT : gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, repeat ? gl.REPEAT : gl.CLAMP_TO_EDGE);
      if (wantFloat) {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.FLOAT, data);
      } else {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
      }
      return tex;
    }

    function makeTex3(size, data) {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_3D, tex);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.REPEAT);
      gl.texImage3D(
        gl.TEXTURE_3D,
        0,
        gl.RGBA8,
        size,
        size,
        size,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        data
      );
      return tex;
    }

    const noiseTex = makeTex(NOISE_SIZE, NOISE_SIZE, true, makeNoiseData(), false);
    const noise3Tex = makeTex3(NOISE3_SIZE, null);
    const blueTex = makeTex(BLUE_SIZE, BLUE_SIZE, true, makeBlueNoiseData(), false);
    gl.bindTexture(gl.TEXTURE_2D, blueTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    const simTex = () => makeTex(SIM_SIZE, SIM_SIZE, true, null, true);
    const thermTex = [simTex(), simTex()];
    const velTex = [simTex(), simTex()];
    const prsTex = [simTex(), simTex()];
    const divTex = simTex();
    const meanTex = simTex();
    gl.bindTexture(gl.TEXTURE_2D, meanTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);

    const simFb = gl.createFramebuffer();

    function bindQuad(prog) {
      gl.useProgram(prog.p);
      gl.bindBuffer(gl.ARRAY_BUFFER, quad);
      gl.enableVertexAttribArray(prog.aPos);
      gl.vertexAttribPointer(prog.aPos, 2, gl.FLOAT, false, 0, 0);
    }

    function targetTex(tex, w, h) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, simFb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      gl.viewport(0, 0, w, h);
    }

    function bindTex(unit, tex, loc) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(loc, unit);
    }

    function bindTex3(unit, tex, loc) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_3D, tex);
      gl.uniform1i(loc, unit);
    }

    function drawSim(tex) {
      targetTex(tex, SIM_SIZE, SIM_SIZE);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    bindQuad(noiseGen);
    gl.bindFramebuffer(gl.FRAMEBUFFER, simFb);
    gl.viewport(0, 0, NOISE3_SIZE, NOISE3_SIZE);
    for (let z = 0; z < NOISE3_SIZE; z++) {
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, noise3Tex, 0, z);
      gl.uniform1f(noiseGen.u.uZ, (z + 0.5) / NOISE3_SIZE);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    bindQuad(thermInit);
    bindTex(0, noiseTex, thermInit.u.uNoise);
    drawSim(thermTex[0]);

    bindQuad(velInit);
    bindTex(0, noiseTex, velInit.u.uNoise);
    drawSim(velTex[0]);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    let thermPing = 0;
    let velPing = 0;
    let prsPing = 0;
    let rawTex = null;
    let accumTex = [null, null];
    let accumPing = 0;
    let accumWarm = false;
    let marchWidth = 0;
    let marchHeight = 0;
    let displayWidth = 0;
    let displayHeight = 0;
    let running = false;
    let hidden = false;
    let raf = 0;
    let frame = 0;
    let viewMode = "clouds";
    const impulses = [];

    function ensureTarget(w, h) {
      let mw = Math.max(1, (w * MARCH_SCALE) | 0);
      let mh = Math.max(1, (h * MARCH_SCALE) | 0);
      const longEdge = Math.max(mw, mh);
      if (longEdge > MAX_MARCH) {
        const s = MAX_MARCH / longEdge;
        mw = Math.max(1, (mw * s) | 0);
        mh = Math.max(1, (mh * s) | 0);
      }
      if (mw === marchWidth && mh === marchHeight && rawTex) return;
      marchWidth = mw;
      marchHeight = mh;
      if (rawTex) gl.deleteTexture(rawTex);
      for (const tex of accumTex) if (tex) gl.deleteTexture(tex);
      rawTex = makeTex(mw, mh, false, null, false);
      accumTex = [makeTex(mw, mh, false, null, false), makeTex(mw, mh, false, null, false)];
      accumWarm = false;
    }

    function screenToSim(sx, sy) {
      const aspect = displayHeight > 0 ? displayWidth / displayHeight : 1;
      const px = (sx - 0.5) * aspect;
      const py = sy - 0.5;
      const reach = (CAM_Y - (CLOUD_BOT + CLOUD_TOP) * 0.5) * SPREAD;
      return {
        x: (px * reach) / DOMAIN + 0.5,
        y: (py * reach) / DOMAIN + 0.5,
        scale: reach / DOMAIN,
      };
    }

    function applyImpulses() {
      while (impulses.length) {
        const imp = impulses.shift();

        if (imp.heat || imp.moist) {
          const src = thermPing;
          const dst = 1 - thermPing;
          bindQuad(impulseTherm);
          bindTex(0, thermTex[src], impulseTherm.u.uTherm);
          gl.uniform2f(impulseTherm.u.uPoint, imp.x, imp.y);
          gl.uniform1f(impulseTherm.u.uRadius, imp.radius);
          gl.uniform1f(impulseTherm.u.uHeat, imp.heat);
          gl.uniform1f(impulseTherm.u.uMoist, imp.moist);
          drawSim(thermTex[dst]);
          thermPing = dst;
        }

        if (imp.force || imp.spin || imp.converge) {
          const src = velPing;
          const dst = 1 - velPing;
          bindQuad(impulseVel);
          bindTex(0, velTex[src], impulseVel.u.uVel);
          gl.uniform2f(impulseVel.u.uPoint, imp.x, imp.y);
          gl.uniform2f(impulseVel.u.uDir, imp.dx, imp.dy);
          gl.uniform1f(impulseVel.u.uRadius, imp.radius);
          gl.uniform1f(impulseVel.u.uForce, imp.force);
          gl.uniform1f(impulseVel.u.uSpin, imp.spin);
          gl.uniform1f(impulseVel.u.uConverge, imp.converge);
          drawSim(velTex[dst]);
          velPing = dst;
        }
      }
    }

    function stepSim() {
      applyImpulses();

      bindQuad(advectVel);
      bindTex(0, velTex[velPing], advectVel.u.uVel);
      drawSim(velTex[1 - velPing]);
      velPing = 1 - velPing;

      bindQuad(advectTherm);
      bindTex(0, thermTex[thermPing], advectTherm.u.uTherm);
      bindTex(1, velTex[velPing], advectTherm.u.uVel);
      bindTex(2, noiseTex, advectTherm.u.uNoise);
      gl.uniform1f(advectTherm.u.uTime, frame * 0.016);
      drawSim(thermTex[1 - thermPing]);
      thermPing = 1 - thermPing;

      bindQuad(copy);
      bindTex(0, thermTex[thermPing], copy.u.uSrc);
      drawSim(meanTex);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, meanTex);
      gl.generateMipmap(gl.TEXTURE_2D);

      bindQuad(divergence);
      bindTex(0, velTex[velPing], divergence.u.uVel);
      bindTex(1, thermTex[thermPing], divergence.u.uTherm);
      bindTex(2, meanTex, divergence.u.uMean);
      drawSim(divTex);

      bindQuad(jacobi);
      bindTex(1, divTex, jacobi.u.uDiv);
      for (let i = 0; i < JACOBI_STEPS; i++) {
        bindTex(0, prsTex[prsPing], jacobi.u.uPressure);
        drawSim(prsTex[1 - prsPing]);
        prsPing = 1 - prsPing;
      }

      bindQuad(project);
      bindTex(0, velTex[velPing], project.u.uVel);
      bindTex(1, prsTex[prsPing], project.u.uPressure);
      drawSim(velTex[1 - velPing]);
      velPing = 1 - velPing;
    }

    function render() {
      raf = 0;
      if (!running || hidden) return;
      raf = requestAnimationFrame(render);
      if (displayWidth <= 0 || displayHeight <= 0) return;

      ensureTarget(displayWidth, displayHeight);
      if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
        canvas.width = displayWidth;
        canvas.height = displayHeight;
      }

      stepSim();

      if (viewMode === "field") {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, displayWidth, displayHeight);
        bindQuad(fieldView);
        bindTex(0, thermTex[thermPing], fieldView.u.uField);
        bindTex(1, velTex[velPing], fieldView.u.uVel);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        return;
      }

      bindQuad(march);
      bindTex3(3, noise3Tex, march.u.uNoise3);
      bindTex(1, blueTex, march.u.uBlueNoise);
      bindTex(2, thermTex[thermPing], march.u.uField);
      gl.uniform2f(march.u.uResolution, marchWidth, marchHeight);
      gl.uniform1i(march.u.uFrame, frame++);
      targetTex(rawTex, marchWidth, marchHeight);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      const histSrc = accumPing;
      const histDst = 1 - accumPing;
      bindQuad(accum);
      bindTex(0, rawTex, accum.u.uRaw);
      bindTex(1, accumWarm ? accumTex[histSrc] : rawTex, accum.u.uHistory);
      gl.uniform1f(accum.u.uBlend, accumWarm ? 0.45 : 1.0);
      targetTex(accumTex[histDst], marchWidth, marchHeight);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      accumPing = histDst;
      accumWarm = true;

      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, displayWidth, displayHeight);
      bindQuad(display);
      bindTex(0, accumTex[accumPing], display.u.uCloud);
      bindTex(1, blueTex, display.u.uBlueNoise);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    function start() {
      if (!raf && running && !hidden) raf = requestAnimationFrame(render);
    }

    return {
      setSize(width, height) {
        displayWidth = width | 0;
        displayHeight = height | 0;
      },
      setHidden(value) {
        hidden = value;
        start();
      },
      setRunning(value) {
        running = value;
        start();
      },
      setView(mode) {
        viewMode = mode === "field" ? "field" : "clouds";
        accumWarm = false;
      },
      stats() {
        const size = 64;
        const out = {};
        const read = (tex, label, names) => {
          gl.bindFramebuffer(gl.FRAMEBUFFER, simFb);
          gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
          const buf = new Float32Array(size * size * 4);
          gl.readPixels(0, 0, size, size, gl.RGBA, gl.FLOAT, buf);
          gl.bindFramebuffer(gl.FRAMEBUFFER, null);
          names.forEach((name, ch) => {
            if (!name) return;
            let min = Infinity;
            let max = -Infinity;
            let sum = 0;
            for (let i = 0; i < size * size; i++) {
              const v = buf[i * 4 + ch];
              if (v < min) min = v;
              if (v > max) max = v;
              sum += v;
            }
            out[`${label}.${name}`] = [min, sum / (size * size), max].map((v) => +v.toFixed(4));
          });
        };
        read(thermTex[thermPing], "therm", ["qv", "qc", "T", "rain"]);
        read(velTex[velPing], "vel", ["u", "v"]);
        read(divTex, "rhs", ["div"]);
        return out;
      },
      addImpulse(imp) {
        const at = screenToSim(imp.sx, imp.sy);
        const dx = imp.dx || 0;
        const dy = imp.dy || 0;
        const len = Math.hypot(dx, dy);
        impulses.push({
          x: at.x,
          y: at.y,
          dx: len > 1e-6 ? dx / len : 0,
          dy: len > 1e-6 ? dy / len : 0,
          radius: imp.radius || 0.06,
          force: len > 1e-6 ? Math.min(len * 90, 3.0) : 0,
          spin: imp.spin || 0,
          converge: imp.converge || 0,
          heat: imp.heat || 0,
          moist: imp.moist || 0,
        });
      },
    };
  }

  scope.createRenderer = createRenderer;
})(typeof self !== "undefined" ? self : window);

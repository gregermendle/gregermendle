(function (scope) {
  const NOISE_SIZE = 256;
  const NOISE3_SIZE = 96;
  const BLUE_SIZE = 128;
  const SIM_SIZE = 256;
  const SIM_MIP = 8;
  const JACOBI_STEPS = 40;
  const MARCH_SCALE = 0.72;
  const MAX_MARCH = 720;

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
      const v = (r / (n - 1)) * 255;
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
const float CORIOLIS = 0.003;
const float JET_AMP = 0.0;
const float JET_RELAX = 0.0006;
const float VEL_MAX = 2.6;
const float CONV_BUOY = 0.004;
const float CONV_RAIN = 0.006;
const float BUOY_T = 0.4;
const float BUOY_QC = 0.28;
const float DIV_SCALE = 0.00045;
const float ADIA = 0.00028;
const float MOIST_CONV = 0.0009;
const float COND_RATE = 0.055;
const float REVAP_RATE = 0.01;
const float LATENT = 0.18;
const float RAIN_THRESH = 0.16;
const float RAIN_RATE = 0.045;
const float RAIN_DECAY = 0.94;
const float RAIN_COOL = 0.0006;
const float TRIGGER_T = 0.0;
const float TRIGGER_Q = 0.0;
const float SURF_EVAP = 0.0024;
const float SURF_RH = 0.82;
const float RAD_RELAX = 0.0035;
const float CLOUD_ADV = 0.35;

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

vec2 wrapTo(vec2 p, vec2 a) {
  return p - a;
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

    const thermInitFs = `#version 300 es
precision highp float;
uniform sampler2D uNoise;
in vec2 vUv;
out vec4 fragColor;
${simLib}
${noiseLib}
void main() {
  float t = tEquilibrium(vUv) + 0.04 * (tileFbm(vUv, 4.0, uNoise) - 0.47);
  float humid = tileFbm(vUv + 3.1, 3.5, uNoise);
  float qv = qsat(t) * (0.5 + 0.22 * humid);
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
  fragColor = vec4(0.0, 0.0, 0.0, 1.0);
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
  vec2 rel = wrapTo(vUv, uPoint);
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
  vec2 rel = wrapTo(vUv, uPoint);
  float w = exp(-dot(rel, rel) / max(uRadius * uRadius, 1e-6));
  vel += uDir * uForce * w;
  vel += vec2(-rel.y, rel.x) * (uSpin * w / max(uRadius, 1e-6));
  vel -= normalize(rel + vec2(1e-6)) * (uConverge * w);
  float speed = length(vel);
  if (speed > VEL_MAX) vel *= VEL_MAX / speed;
  fragColor = vec4(vel, 0.0, 1.0);
}`;

    const frontThermFs = `#version 300 es
precision highp float;
uniform sampler2D uTherm;
uniform vec2 uA;
uniform vec2 uB;
uniform float uWidth;
uniform float uCold;
uniform float uMoist;
in vec2 vUv;
out vec4 fragColor;
${simLib}
void main() {
  vec4 s = texture(uTherm, vUv);
  vec2 ab = wrapTo(uB, uA);
  float len = max(length(ab), 1e-5);
  vec2 t = ab / len;
  vec2 n = vec2(-t.y, t.x);
  vec2 rel = wrapTo(vUv, uA);
  float along = clamp(dot(rel, t), 0.0, len);
  vec2 closest = t * along;
  vec2 d = rel - closest;
  float dist = length(d);
  float side = dot(rel, n);
  float w = exp(-dist * dist / max(uWidth * uWidth, 1e-6));
  float taper = smoothstep(0.0, uWidth * 1.4, along) * smoothstep(len, len - uWidth * 1.4, along);
  w *= mix(0.45, 1.0, taper);
  s.z = clamp(s.z - uCold * sign(side) * w, 0.0, 1.4);
  s.x = clamp(s.x + uMoist * w * mix(0.25, 1.15, 0.5 + 0.5 * sign(side)), 0.0, 2.0);
  fragColor = s;
}`;

    const frontVelFs = `#version 300 es
precision highp float;
uniform sampler2D uVel;
uniform vec2 uA;
uniform vec2 uB;
uniform float uWidth;
uniform float uAlong;
uniform float uConverge;
uniform float uSpin;
in vec2 vUv;
out vec4 fragColor;
${simLib}
void main() {
  vec2 vel = texture(uVel, vUv).xy;
  vec2 ab = wrapTo(uB, uA);
  float len = max(length(ab), 1e-5);
  vec2 t = ab / len;
  vec2 n = vec2(-t.y, t.x);
  vec2 rel = wrapTo(vUv, uA);
  float along = clamp(dot(rel, t), 0.0, len);
  vec2 closest = t * along;
  vec2 d = rel - closest;
  float dist = length(d);
  float w = exp(-dist * dist / max(uWidth * uWidth, 1e-6));
  float taper = smoothstep(0.0, uWidth, along) * smoothstep(len, len - uWidth, along);
  w *= mix(0.4, 1.0, taper);
  vel += t * uAlong * w;
  vel -= n * sign(dot(rel, n) + 1e-6) * uConverge * w;
  vel += vec2(-d.y, d.x) * (uSpin * w / max(uWidth, 1e-6));
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
  vel *= 0.988;
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
  vec2 vel = texture(uVel, vUv).xy * CLOUD_ADV;
  vec2 src = vUv - vel * TEX;
  vec4 s = texture(uTherm, src);
  vec4 blur = 0.25 * (
    texture(uTherm, src + vec2(TEX, 0.0)) +
    texture(uTherm, src - vec2(TEX, 0.0)) +
    texture(uTherm, src + vec2(0.0, TEX)) +
    texture(uTherm, src - vec2(0.0, TEX))
  );
  s = mix(s, blur, 0.08);

  float qv = s.x;
  float qc = s.y;
  float t = s.z;
  float rain = s.w;

  float div = 0.4 * divergenceAt(uVel, vUv)
    + 0.15 * (
      divergenceAt(uVel, vUv + vec2(TEX, 0.0)) +
      divergenceAt(uVel, vUv - vec2(TEX, 0.0)) +
      divergenceAt(uVel, vUv + vec2(0.0, TEX)) +
      divergenceAt(uVel, vUv - vec2(0.0, TEX))
    );
  float lift = clamp(-div / DIV_SCALE, -1.6, 1.6);
  t -= ADIA * lift;
  qv += MOIST_CONV * lift * qv;

  float teq = tEquilibrium(vUv);
  t += RAD_RELAX * (teq - t);

  float trigger = tileFbm(vUv + vec2(uTime * 0.008, uTime * 0.005), 10.0, uNoise) - 0.47;
  t += TRIGGER_T * trigger;
  qv += TRIGGER_Q * trigger;

  float qs = qsat(t);
  float speed = length(vel);
  qv += SURF_EVAP * (1.0 + 0.35 * speed) * max(qs * SURF_RH - qv, 0.0);
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
  qv += rain * 0.0025;

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
  float perlin = 0.5 * valueTile(p, 4.0) + 0.26 * valueTile(p, 8.0) + 0.14 * valueTile(p, 16.0) + 0.10 * valueTile(p, 32.0);
  float w3 = worleyTile(p, 3.0);
  float w6 = worleyTile(p, 6.0);
  float w12 = worleyTile(p, 12.0);
  fragColor = vec4(clamp(perlin * 0.62 + w3 * 0.38, 0.0, 1.0), w3, w6, w12);
}`;

    const weatherFs = `#version 300 es
precision highp float;
uniform highp sampler3D uNoise3;
uniform sampler2D uField;
uniform sampler2D uVel;
uniform sampler2D uBlueNoise;
uniform sampler2D uStorm;
uniform vec2 uResolution;
uniform int uFrame;
uniform int uSteps;
in vec2 vUv;
out vec4 fragColor;
${simLib}

#define MAX_STEPS 96
#define LIGHT_STEPS 4

const vec3 SUN_DIR = normalize(vec3(0.52, 0.64, 0.44));
const float SIGMA = 14.0;
const float SIGMA_L = 8.8;

float worleyFbm(vec4 n) {
  return n.g * 0.6 + n.b * 0.28 + n.a * 0.12;
}
float remap(float v, float lo, float hi) {
  return clamp((v - lo) / max(hi - lo, 1e-4), 0.0, 1.0);
}

float lobe(vec3 p, vec2 uv, float yMid, float thick, float cov, float nOff, float nScale, float settle, bool cheap) {
  if (cov < 0.02) return 0.0;
  float vh = (p.y - yMid) / max(thick, 1e-4);
  if (abs(vh) > 1.05) return 0.0;
  float vert = smoothstep(0.0, 0.28, 1.0 - vh * vh);
  vec4 n = texture(uNoise3, vec3(uv.x, yMid * 0.4 + nOff, uv.y) * nScale);
  vec2 shift = (n.gb - 0.5) * (0.028 + nOff * 0.05);
  vec4 n2 = texture(uNoise3, vec3(uv.x + shift.x, yMid * 0.4 + nOff, uv.y + shift.y) * nScale);
  float cell = mix(0.06, 0.14, settle) * mix(1.0, 0.4, cov);
  float shape = mix(n2.r, worleyFbm(n2), cell);
  float rim = mix(0.04, 0.14, nOff) + (1.0 - cov) * 0.32;
  float d = remap(shape * vert, rim, 1.0);
  if (!cheap && d > 0.0) {
    vec4 t = texture(uNoise3, vec3(uv.x, p.y, uv.y) * nScale * 1.45 + 0.28);
    d = remap(d, worleyFbm(t) * mix(0.01, 0.04, settle) * (0.2 + 0.25 * abs(vh)), 1.0);
  }
  return d * mix(1.25, 1.7, cov);
}

float cloudDensity(vec3 p, bool cheap) {
  if (p.y < 0.0 || p.y > 1.42) return 0.0;
  float h = min(p.y, 1.0);
  vec2 vel = texture(uVel, p.xz).xy;
  vec4 n0 = texture(uNoise3, vec3(p.x, p.y * 0.55, p.z) * 2.4);
  vec4 n1 = texture(uNoise3, vec3(p.x, p.y * 0.8, p.z) * 4.0 + 0.19);
  vec4 f0 = texture(uField, p.xz);
  float qcN0 = 0.25 * (
    texture(uField, p.xz + vec2(TEX, 0.0)).y +
    texture(uField, p.xz - vec2(TEX, 0.0)).y +
    texture(uField, p.xz + vec2(0.0, TEX)).y +
    texture(uField, p.xz - vec2(0.0, TEX)).y
  );
  float fresh0 = smoothstep(0.03, 0.14, abs(f0.y - qcN0) * 7.0);
  vec2 warp = (n0.gb - 0.5) * mix(0.03, 0.11, h) + (n1.ba - 0.5) * mix(0.016, 0.06, h);
  warp += vel * CLOUD_ADV * h * 0.045;
  warp += (n0.ra - 0.5) * fresh0 * mix(0.02, 0.09, h);
  vec2 uv = p.xz + warp;
  vec4 f = texture(uField, uv);
  float qc = f.y;
  float temp = f.z;
  float rain = f.w;
  float cover = smoothstep(0.012, 0.2, qc);
  if (cover < 0.008) return 0.0;

  float qcN = 0.25 * (
    texture(uField, uv + vec2(TEX, 0.0)).y +
    texture(uField, uv - vec2(TEX, 0.0)).y +
    texture(uField, uv + vec2(0.0, TEX)).y +
    texture(uField, uv - vec2(0.0, TEX)).y
  );
  float fresh = max(fresh0, smoothstep(0.03, 0.14, abs(qc - qcN) * 7.0));
  float qcSoft = mix(qc, qcN, 0.48);
  float settle = 1.0 - 0.86 * fresh;

  float unstable = clamp((temp - tEquilibrium(uv)) * 5.5 + cover * 0.35, 0.0, 1.0);
  float wet = smoothstep(0.05, 0.45, rain);
  float mass = smoothstep(0.05, 0.2, qcSoft);
  float tower = smoothstep(0.14, 0.42, qcSoft);
  float overshoot = smoothstep(0.28, 0.65, qcSoft);
  float top = mix(0.26, 0.5, mass);
  top = mix(top, 0.96, tower * 0.8 + unstable * 0.26);
  top = mix(top, 1.34, overshoot);
  vec4 bump0 = texture(uNoise3, vec3(uv.x, 0.12, uv.y) * 1.8);
  top *= mix(1.0, mix(0.93, 1.04, bump0.r), settle * (1.0 - cover * 0.35));
  top *= 1.0 - 0.14 * wet * (1.0 - overshoot);
  if (p.y > top) return 0.0;

  float d0 = lobe(p, uv, top * 0.18, top * 0.26, cover, 0.0, 1.8, settle, cheap);
  float d1 = lobe(p, uv, top * 0.42, top * 0.22, cover * mix(0.45, 0.92, mass), 0.16, 2.2, settle, cheap);
  float d2 = lobe(p, uv, top * 0.68, top * 0.18, cover * mix(0.0, 0.88, tower), 0.32, 2.6, settle, cheap);
  float d3 = lobe(p, uv, top * 0.9, top * 0.14, cover * mix(0.0, 0.82, overshoot), 0.5, 3.0, settle, cheap);
  float d = max(max(d0, d1), max(d2, d3));
  float wisp = remap(n0.r, mix(0.18, 0.42, cover), 0.78);
  d *= mix(wisp, 1.0, settle * cover);
  return clamp(d, 0.0, 1.0);
}

float peakShade(vec2 uv, float tower, float overshoot) {
  vec2 sd = normalize(SUN_DIR.xz) * 0.02;
  float h0 = texture(uNoise3, vec3(uv.x, 0.3, uv.y) * 2.4).r;
  float h1 = texture(uNoise3, vec3(uv.x + sd.x, 0.3, uv.y + sd.y) * 2.4).r;
  float t0 = texture(uNoise3, vec3(uv.x, 0.48, uv.y) * 3.6).r;
  float t1 = texture(uNoise3, vec3(uv.x + sd.x, 0.48, uv.y + sd.y) * 3.6).r;
  float block = max(h1 - h0, 0.0) * tower + max(t1 - t0, 0.0) * overshoot;
  return exp(-block * 1.15);
}

float lightTau(vec3 p, float jitter) {
  float tau = 0.0;
  float s = 0.04;
  p += SUN_DIR * s * jitter;
  for (int i = 0; i < LIGHT_STEPS; i++) {
    p += SUN_DIR * s;
    tau += cloudDensity(p, true) * s;
    s *= 1.45;
  }
  return tau;
}

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
vec2 hash22(vec2 p) {
  return vec2(hash12(p), hash12(p + 17.13));
}

const vec3 BOLT_CORE = vec3(1.0, 0.93, 1.0);
const vec3 BOLT_SCAT = vec3(0.5, 0.34, 1.0);

float distSeg(vec3 p, vec3 a, vec3 b, out float u) {
  vec3 ba = b - a;
  u = clamp(dot(p - a, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
  return length(p - a - ba * u);
}

vec3 pathAt(vec3 a, vec3 b, vec3 c, vec3 d, float u) {
  if (u < 0.33) return mix(a, b, u / 0.33);
  if (u < 0.66) return mix(b, c, (u - 0.33) / 0.33);
  return mix(c, d, (u - 0.66) / 0.34);
}

float moistDen(vec3 p) {
  float d = cloudDensity(p, true);
  float qc = texture(uField, p.xz).y;
  return d * (0.55 + 0.6 * smoothstep(0.03, 0.3, qc));
}

float boltFlicker(float frame, float seed) {
  float a = step(0.28, hash12(vec2(floor(frame * 0.55), seed)));
  float b = step(0.42, hash12(vec2(floor(frame), seed + 2.7)));
  float c = 0.45 + 0.55 * hash12(vec2(floor(frame * 0.28), seed + 8.1));
  float dart = exp(-abs(fract(frame * 0.19 + seed) - 0.5) * 14.0);
  return mix(0.35, 1.0, a) * mix(0.5, 1.0, b) * c + dart * 0.85;
}

vec3 boltLight(vec3 pos, vec3 a, vec3 b, vec3 c, vec3 d, float amp, float den, float travel) {
  if (amp < 0.008) return vec3(0.0);
  float u0, u1, u2;
  float d0 = distSeg(pos, a, b, u0);
  float d1 = distSeg(pos, b, c, u1);
  float d2 = distSeg(pos, c, d, u2);
  float md = d0;
  float u = u0 * 0.33;
  vec3 closest = mix(a, b, u0);
  if (d1 < md) {
    md = d1;
    u = 0.33 + u1 * 0.33;
    closest = mix(b, c, u1);
  }
  if (d2 < md) {
    md = d2;
    u = 0.66 + u2 * 0.34;
    closest = mix(c, d, u2);
  }
  if (md > 0.2) return vec3(0.0);
  float live = 1.0 - smoothstep(travel, travel + 0.1, u);
  float head = exp(-abs(u - travel) * 16.0) * step(u, travel + 0.06);
  if (live < 0.02 && head < 0.02) return vec3(0.0);
  float alongTau = 0.0;
  for (int i = 1; i <= 4; i++) {
    alongTau += moistDen(pathAt(a, b, c, d, min(u, travel) * float(i) / 4.0));
  }
  alongTau *= min(u, travel) * 0.12;
  vec3 toC = closest - pos;
  float clen = length(toC);
  vec3 dir = toC / max(clen, 1e-5);
  float s = min(clen, 0.12) / 4.0;
  vec3 p = pos;
  float offTau = 0.0;
  for (int i = 0; i < 4; i++) {
    p += dir * s;
    offTau += moistDen(p) * s;
  }
  float transAlong = exp(-alongTau * 1.8);
  float transOff = exp(-offTau * 4.2);
  float moist = texture(uField, pos.xz).y;
  float scatter = den * (0.35 + 0.65 * smoothstep(0.03, 0.28, moist));
  float core = (head * 2.4 + live * 1.1) * exp(-md * md * 700.0) * transAlong;
  float fill = (head * 1.1 + live) * exp(-md / 0.065) / (1.0 + md * 5.0);
  fill *= transOff * transAlong * scatter;
  return amp * (core * BOLT_CORE * 4.2 + fill * BOLT_SCAT * 3.1);
}

void main() {
  vec3 rd = normalize(vec3(0.1, -1.0, 0.07));
  vec3 ro = vec3(vUv.x, 1.74, vUv.y);
  float tAim = (0.42 - ro.y) / rd.y;
  ro.xz -= rd.xz * tAim;
  float t0 = (1.42 - ro.y) / rd.y;
  float t1 = (0.0 - ro.y) / rd.y;
  float dt = (t1 - t0) / float(max(uSteps, 1));
  float jitter = fract(texture(uBlueNoise, gl_FragCoord.xy / 128.0).r + float(uFrame % 24) * 0.618034);
  float t = t0 + dt * jitter;
  float trans = 1.0;
  float acc = 0.0;
  vec3 boltAcc = vec3(0.0);
  vec3 ba[4];
  vec3 bb[4];
  vec3 bc[4];
  vec3 bd[4];
  float boltF[4];
  float boltT[4];
  float tm = float(uFrame);
  for (int s = 0; s < 4; s++) {
    ba[s] = bb[s] = bc[s] = bd[s] = vec3(0.5, 0.12, 0.5);
    boltF[s] = 0.0;
    boltT[s] = 0.0;
    float seed = 8.3 + float(s) * 21.9;
    float period = 130.0 + hash12(vec2(seed, 1.4)) * 150.0;
    float shifted = tm + seed * 37.0;
    float cycle = floor(shifted / period);
    float phase = fract(shifted / period);
    float env = exp(-phase * 4.8) * step(phase, 0.4);
    if (env < 0.008) continue;
    vec2 best = hash22(vec2(cycle, seed));
    float bestCh = 0.0;
    for (int k = 0; k < 5; k++) {
      vec2 p = hash22(vec2(cycle + 0.13, seed + float(k) * 9.4));
      float ch = texture(uStorm, p).x;
      if (ch > bestCh) {
        bestCh = ch;
        best = p;
      }
    }
    if (bestCh < 0.14) continue;
    vec2 dir = normalize(hash22(vec2(cycle, seed + 3.2)) - 0.5);
    float reach = 0.12 + hash12(vec2(cycle, seed + 5.0)) * 0.16;
    vec2 endp = best + dir * reach;
    if (texture(uStorm, endp).x < 0.1) {
      vec2 alt = hash22(vec2(cycle, seed + 14.0));
      if (texture(uStorm, alt).x > 0.12) endp = mix(best, alt, 0.8);
    }
    vec2 n = vec2(-dir.y, dir.x);
    vec2 m1 = mix(best, endp, 0.34) + n * (hash12(vec2(cycle, seed + 6.1)) * 2.0 - 1.0) * 0.055;
    vec2 m2 = mix(best, endp, 0.67) + n * (hash12(vec2(cycle, seed + 7.4)) * 2.0 - 1.0) * 0.05;
    ba[s] = vec3(best.x, 0.045 + hash12(vec2(cycle, seed + 11.0)) * 0.07, best.y);
    bb[s] = vec3(m1.x, 0.07 + hash12(vec2(cycle, seed + 12.0)) * 0.16, m1.y);
    bc[s] = vec3(m2.x, 0.06 + hash12(vec2(cycle, seed + 13.0)) * 0.18, m2.y);
    bd[s] = vec3(endp.x, 0.08 + hash12(vec2(cycle, seed + 14.2)) * 0.2, endp.y);
    boltF[s] = env * smoothstep(0.14, 0.55, bestCh) * boltFlicker(tm, seed + cycle);
    boltT[s] = clamp(phase / 0.11, 0.0, 1.0);
  }

  for (int i = 0; i < MAX_STEPS; i++) {
    if (i >= uSteps || trans < 0.03) break;
    vec3 pos = ro + rd * t;
    float den = cloudDensity(pos, false);
    if (den > 0.004) {
      float light = exp(-lightTau(pos, jitter) * SIGMA_L);
      float powder = 1.0 - exp(-den * 4.2);
      float h = clamp(pos.y, 0.0, 1.0);
      vec4 fld = texture(uField, pos.xz);
      float rain = fld.w;
      float tower = smoothstep(0.14, 0.42, fld.y);
      float overshoot = smoothstep(0.28, 0.65, fld.y);
      light *= peakShade(pos.xz, tower, overshoot);
      float crest = smoothstep(0.6, 1.1, pos.y) * mix(0.1, 0.7, tower);
      float lit = mix(0.18, 1.08, light) * mix(0.7, 1.08, powder);
      lit *= mix(1.0, 1.08, crest * light);
      lit *= mix(1.0, 0.62, smoothstep(0.06, 0.5, rain) * (1.0 - h));
      vec3 glow = vec3(0.0);
      for (int s = 0; s < 4; s++) glow += boltLight(pos, ba[s], bb[s], bc[s], bd[s], boltF[s], den, boltT[s]);
      glow *= 0.55 + 0.45 * powder;
      float alpha = 1.0 - exp(-den * dt * SIGMA);
      acc += trans * alpha * lit;
      boltAcc += trans * alpha * glow;
      trans *= 1.0 - alpha;
    }
    t += dt;
  }

  float lum = acc / (1.0 + acc * 0.35);
  vec3 flash = boltAcc / (1.0 + boltAcc * 0.05);
  float w = clamp(max(flash.b, flash.r) * 1.15, 0.0, 0.92);
  vec3 col = vec3(lum) * (1.0 - w * 0.82) + flash;
  fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}`;

    const stormFs = `#version 300 es
precision highp float;
uniform sampler2D uStorm;
uniform sampler2D uTherm;
in vec2 vUv;
out vec4 fragColor;
${simLib}
void main() {
  vec4 f = texture(uTherm, vUv);
  float moist = f.y + f.w * 0.28;
  float prev = texture(uStorm, vUv).x;
  float blur = 0.25 * (
    texture(uStorm, vUv + vec2(TEX, 0.0)).x +
    texture(uStorm, vUv - vec2(TEX, 0.0)).x +
    texture(uStorm, vUv + vec2(0.0, TEX)).x +
    texture(uStorm, vUv - vec2(0.0, TEX)).x
  );
  prev = mix(prev, blur, 0.18);
  float charge = prev;
  if (moist > 0.11) charge = min(1.0, charge + 0.018 + (moist - 0.11) * 0.05);
  else if (moist < 0.055) charge = max(0.0, charge - 0.005);
  else charge = max(0.0, charge - 0.0012);
  fragColor = vec4(charge, 0.0, 0.0, 1.0);
}`;

    const inkFs = `#version 300 es
precision highp float;
uniform sampler2D uSrc;
uniform vec2 uA;
uniform vec2 uB;
uniform float uWidth;
uniform float uStrength;
in vec2 vUv;
out vec4 fragColor;
void main() {
  vec3 prev = texture(uSrc, vUv).rgb;
  vec2 ab = uB - uA;
  float len = max(length(ab), 1e-5);
  vec2 t = ab / len;
  vec2 rel = vUv - uA;
  float along = clamp(dot(rel, t), 0.0, len);
  float dist = length(rel - t * along);
  float w = exp(-dist * dist / max(uWidth * uWidth, 1e-6));
  fragColor = vec4(max(prev, vec3(w * uStrength)), 1.0);
}`;

    const fadeFs = `#version 300 es
precision highp float;
uniform sampler2D uSrc;
uniform float uFade;
in vec2 vUv;
out vec4 fragColor;
void main() {
  fragColor = vec4(texture(uSrc, vUv).rgb * uFade, 1.0);
}`;

    const compositeFs = `#version 300 es
precision highp float;
uniform sampler2D uWeather;
uniform sampler2D uInk;
in vec2 vUv;
out vec4 fragColor;
void main() {
  vec3 cloud = texture(uWeather, vUv).rgb;
  float ink = texture(uInk, vUv).r;
  fragColor = vec4(max(cloud, vec3(ink * 0.55)), 1.0);
}`;

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
    const frontTherm = makeProgram(blitVs, frontThermFs, [
      "uTherm",
      "uA",
      "uB",
      "uWidth",
      "uCold",
      "uMoist",
    ]);
    const frontVel = makeProgram(blitVs, frontVelFs, [
      "uVel",
      "uA",
      "uB",
      "uWidth",
      "uAlong",
      "uConverge",
      "uSpin",
    ]);
    const advectVel = makeProgram(blitVs, advectVelFs, ["uVel"]);
    const advectTherm = makeProgram(blitVs, advectThermFs, [
      "uTherm",
      "uVel",
      "uNoise",
      "uTime",
    ]);
    const noiseGen = makeProgram(blitVs, noiseGenFs, ["uZ"]);
    const copy = makeProgram(blitVs, copyFs, ["uSrc"]);
    const divergence = makeProgram(blitVs, divergenceFs, ["uVel", "uTherm", "uMean"]);
    const jacobi = makeProgram(blitVs, jacobiFs, ["uPressure", "uDiv"]);
    const project = makeProgram(blitVs, projectFs, ["uVel", "uPressure"]);
    const weather = makeProgram(blitVs, weatherFs, [
      "uNoise3",
      "uField",
      "uVel",
      "uBlueNoise",
      "uStorm",
      "uResolution",
      "uFrame",
      "uSteps",
    ]);
    const stormCharge = makeProgram(blitVs, stormFs, ["uStorm", "uTherm"]);
    const accum = makeProgram(blitVs, `#version 300 es
precision highp float;
uniform sampler2D uRaw;
uniform sampler2D uHistory;
uniform float uBlend;
in vec2 vUv;
out vec4 fragColor;
void main() {
  vec3 raw = texture(uRaw, vUv).rgb;
  vec3 hist = texture(uHistory, vUv).rgb;
  float rawFlash = max(raw.b - raw.r, 0.0);
  float k = mix(uBlend, max(uBlend, 0.78), smoothstep(0.012, 0.07, rawFlash));
  fragColor = vec4(mix(hist, raw, k), 1.0);
}`, ["uRaw", "uHistory", "uBlend"]);
    const ink = makeProgram(blitVs, inkFs, ["uSrc", "uA", "uB", "uWidth", "uStrength"]);
    const fade = makeProgram(blitVs, fadeFs, ["uSrc", "uFade"]);
    const composite = makeProgram(blitVs, compositeFs, ["uWeather", "uInk"]);

    const programs = [
      noiseGen,
      thermInit,
      velInit,
      impulseTherm,
      impulseVel,
      frontTherm,
      frontVel,
      advectVel,
      advectTherm,
      copy,
      divergence,
      jacobi,
      project,
      weather,
      stormCharge,
      accum,
      ink,
      fade,
      composite,
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

    function makeTex3(size) {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_3D, tex);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.REPEAT);
      gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA8, size, size, size, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      return tex;
    }

    function bindTex3(unit, tex, loc) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_3D, tex);
      gl.uniform1i(loc, unit);
    }

    const noiseTex = makeTex(NOISE_SIZE, NOISE_SIZE, true, makeNoiseData(), false);
    const noise3Tex = makeTex3(NOISE3_SIZE);
    const blueTex = makeTex(BLUE_SIZE, BLUE_SIZE, true, makeBlueNoiseData(), false);
    gl.bindTexture(gl.TEXTURE_2D, blueTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    const simTex = () => makeTex(SIM_SIZE, SIM_SIZE, false, null, true);
    const thermTex = [simTex(), simTex()];
    const velTex = [simTex(), simTex()];
    const stormTex = [simTex(), simTex()];
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
    gl.clearColor(0, 0, 0, 1);
    targetTex(stormTex[0], SIM_SIZE, SIM_SIZE);
    gl.clear(gl.COLOR_BUFFER_BIT);
    targetTex(stormTex[1], SIM_SIZE, SIM_SIZE);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    let thermPing = 0;
    let velPing = 0;
    let stormPing = 0;
    let prsPing = 0;
    let weatherTex = null;
    let rawTex = null;
    let accumTex = [null, null];
    let accumPing = 0;
    let accumWarm = false;
    let inkTex = [null, null];
    let inkPing = 0;
    let viewWidth = 0;
    let viewHeight = 0;
    let marchWidth = 0;
    let marchHeight = 0;
    let displayWidth = 0;
    let displayHeight = 0;
    let marchSteps = 40;
    let running = false;
    let hidden = false;
    let raf = 0;
    let frame = 0;
    let fpsFrames = 0;
    let fpsLast = 0;
    let onFps = null;
    const impulses = [];
    const fronts = [];

    function ensureView(w, h) {
      let mw = Math.max(1, (w * MARCH_SCALE) | 0);
      let mh = Math.max(1, (h * MARCH_SCALE) | 0);
      const longEdge = Math.max(mw, mh);
      if (longEdge > MAX_MARCH) {
        const s = MAX_MARCH / longEdge;
        mw = Math.max(1, (mw * s) | 0);
        mh = Math.max(1, (mh * s) | 0);
      }
      if (w === viewWidth && h === viewHeight && mw === marchWidth && weatherTex) return;
      viewWidth = w;
      viewHeight = h;
      marchWidth = mw;
      marchHeight = mh;
      if (weatherTex) gl.deleteTexture(weatherTex);
      if (rawTex) gl.deleteTexture(rawTex);
      for (const tex of accumTex) if (tex) gl.deleteTexture(tex);
      for (const tex of inkTex) if (tex) gl.deleteTexture(tex);
      weatherTex = makeTex(w, h, false, null, false);
      rawTex = makeTex(mw, mh, false, null, false);
      accumTex = [makeTex(mw, mh, false, null, false), makeTex(mw, mh, false, null, false)];
      inkTex = [makeTex(w, h, false, null, false), makeTex(w, h, false, null, false)];
      accumWarm = false;
    }

    function screenToSim(sx, sy) {
      return { x: sx, y: sy };
    }

    function applyFront(seg) {
      const srcT = thermPing;
      const dstT = 1 - thermPing;
      bindQuad(frontTherm);
      bindTex(0, thermTex[srcT], frontTherm.u.uTherm);
      gl.uniform2f(frontTherm.u.uA, seg.ax, seg.ay);
      gl.uniform2f(frontTherm.u.uB, seg.bx, seg.by);
      gl.uniform1f(frontTherm.u.uWidth, seg.width);
      gl.uniform1f(frontTherm.u.uCold, seg.cold);
      gl.uniform1f(frontTherm.u.uMoist, seg.moist);
      drawSim(thermTex[dstT]);
      thermPing = dstT;

      const srcV = velPing;
      const dstV = 1 - velPing;
      bindQuad(frontVel);
      bindTex(0, velTex[srcV], frontVel.u.uVel);
      gl.uniform2f(frontVel.u.uA, seg.ax, seg.ay);
      gl.uniform2f(frontVel.u.uB, seg.bx, seg.by);
      gl.uniform1f(frontVel.u.uWidth, seg.width);
      gl.uniform1f(frontVel.u.uAlong, seg.along);
      gl.uniform1f(frontVel.u.uConverge, seg.converge);
      gl.uniform1f(frontVel.u.uSpin, seg.spin);
      drawSim(velTex[dstV]);
      velPing = dstV;
    }

    function applyImpulses() {
      while (fronts.length) applyFront(fronts.shift());

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

      bindQuad(stormCharge);
      bindTex(0, stormTex[stormPing], stormCharge.u.uStorm);
      bindTex(1, thermTex[thermPing], stormCharge.u.uTherm);
      drawSim(stormTex[1 - stormPing]);
      stormPing = 1 - stormPing;
    }

    function fadeInk() {
      if (!inkTex[0]) return;
      const src = inkPing;
      const dst = 1 - inkPing;
      bindQuad(fade);
      bindTex(0, inkTex[src], fade.u.uSrc);
      gl.uniform1f(fade.u.uFade, 0.94);
      targetTex(inkTex[dst], viewWidth, viewHeight);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      inkPing = dst;
    }

    function drawWeather() {
      ensureView(displayWidth, displayHeight);
      fadeInk();

      bindQuad(weather);
      bindTex3(3, noise3Tex, weather.u.uNoise3);
      bindTex(0, thermTex[thermPing], weather.u.uField);
      bindTex(1, velTex[velPing], weather.u.uVel);
      bindTex(2, blueTex, weather.u.uBlueNoise);
      bindTex(4, stormTex[stormPing], weather.u.uStorm);
      gl.uniform2f(weather.u.uResolution, marchWidth, marchHeight);
      gl.uniform1i(weather.u.uFrame, frame);
      gl.uniform1i(weather.u.uSteps, marchSteps);
      targetTex(rawTex, marchWidth, marchHeight);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      const histSrc = accumPing;
      const histDst = 1 - accumPing;
      bindQuad(accum);
      bindTex(0, rawTex, accum.u.uRaw);
      bindTex(1, accumWarm ? accumTex[histSrc] : rawTex, accum.u.uHistory);
      gl.uniform1f(accum.u.uBlend, accumWarm ? 0.26 : 1.0);
      targetTex(accumTex[histDst], marchWidth, marchHeight);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      accumPing = histDst;
      accumWarm = true;

      bindQuad(copy);
      bindTex(0, accumTex[accumPing], copy.u.uSrc);
      targetTex(weatherTex, viewWidth, viewHeight);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, displayWidth, displayHeight);
      bindQuad(composite);
      bindTex(0, weatherTex, composite.u.uWeather);
      bindTex(1, inkTex[inkPing], composite.u.uInk);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    function render() {
      raf = 0;
      if (!running || hidden) return;
      raf = requestAnimationFrame(render);
      if (displayWidth <= 0 || displayHeight <= 0) return;
      if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
        canvas.width = displayWidth;
        canvas.height = displayHeight;
      }
      stepSim();
      frame++;
      drawWeather();
      fpsFrames++;
      const now = performance.now();
      if (!fpsLast) fpsLast = now;
      if (now - fpsLast >= 500) {
        if (onFps) onFps((fpsFrames * 1000) / (now - fpsLast));
        fpsFrames = 0;
        fpsLast = now;
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
      setHidden(value) {
        hidden = value;
        start();
      },
      setRunning(value) {
        running = value;
        start();
      },
      setSteps(value) {
        marchSteps = Math.max(16, Math.min(96, value | 0));
      },
      setOnFps(fn) {
        onFps = fn;
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
          force: len > 1e-6 ? Math.min(len * 70, 2.2) : 0,
          spin: imp.spin || 0,
          converge: imp.converge || 0,
          heat: imp.heat || 0,
          moist: imp.moist || 0,
        });
      },
      addFront(seg) {
        const a = screenToSim(seg.ax, seg.ay);
        const b = screenToSim(seg.bx, seg.by);
        fronts.push({
          ax: a.x,
          ay: a.y,
          bx: b.x,
          by: b.y,
          width: seg.width || 0.035,
          cold: seg.cold || 0.045,
          moist: seg.moist || 0.08,
          along: seg.along || 0.22,
          converge: seg.converge || 0.16,
          spin: seg.spin || 0.12,
        });
        if (inkTex[0]) {
          const src = inkPing;
          const dst = 1 - inkPing;
          bindQuad(ink);
          bindTex(0, inkTex[src], ink.u.uSrc);
          gl.uniform2f(ink.u.uA, a.x, a.y);
          gl.uniform2f(ink.u.uB, b.x, b.y);
          gl.uniform1f(ink.u.uWidth, Math.max(0.004, (seg.width || 0.035) * 0.32));
          gl.uniform1f(ink.u.uStrength, 0.7);
          targetTex(inkTex[dst], viewWidth, viewHeight);
          gl.drawArrays(gl.TRIANGLES, 0, 3);
          inkPing = dst;
        }
      },
    };
  }

  scope.createRenderer = createRenderer;
})(typeof self !== "undefined" ? self : window);
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   for (const s of systems) {
        const segs = s.segs;
        if (!segs.length) continue;
        if (segs.length <= 16) {
          for (const seg of segs) stampInk(seg.ax, seg.ay, seg.bx, seg.by, 0.0038, 0.28);
          continue;
        }
        const pts = [{ x: segs[0].ax, y: segs[0].ay }];
        for (const seg of segs) pts.push({ x: seg.bx, y: seg.by });
        const step = (pts.length - 1) / 16;
        for (let i = 0; i < 16; i++) {
          const a = pts[Math.round(i * step)];
          const b = pts[Math.round((i + 1) * step)];
          stampInk(a.x, a.y, b.x, b.y, 0.0038, 0.28);
        }
      }
    }

    function stepSim() {
      applyWarms();
      applyImpulses();

      bindQuad(advectVel);
      bindTex(0, velTex[velPing], advectVel.u.uVel);
      bindTex(1, thermTex[thermPing], advectVel.u.uTherm);
      drawSim(velTex[1 - velPing]);
      velPing = 1 - velPing;

      bindQuad(advectTherm);
      bindTex(0, thermTex[thermPing], advectTherm.u.uTherm);
      bindTex(1, velTex[velPing], advectTherm.u.uVel);
      bindTex(2, noiseTex, advectTherm.u.uNoise);
      bindTex(3, sstTex[sstPing], advectTherm.u.uSst);
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
      bindSeeds();
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

      applySystems();

      bindQuad(stormCharge);
      bindTex(0, stormTex[stormPing], stormCharge.u.uStorm);
      bindTex(1, thermTex[thermPing], stormCharge.u.uTherm);
      drawSim(stormTex[1 - stormPing]);
      stormPing = 1 - stormPing;

      relaxSst();
    }

    function fadeInk() {
      if (!inkTex[0]) return;
      const src = inkPing;
      const dst = 1 - inkPing;
      bindQuad(fade);
      bindTex(0, inkTex[src], fade.u.uSrc);
      gl.uniform1f(fade.u.uFade, 0.94);
      targetTex(inkTex[dst], viewWidth, viewHeight);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      inkPing = dst;
    }

    function drawWeather() {
      ensureView(displayWidth, displayHeight);
      fadeInk();
      drawSystemInk();

      bindQuad(weather);
      bindTex3(3, noise3Tex, weather.u.uNoise3);
      bindTex(0, thermTex[thermPing], weather.u.uField);
      bindTex(1, velTex[velPing], weather.u.uVel);
      bindTex(2, blueTex, weather.u.uBlueNoise);
      bindTex(4, stormTex[stormPing], weather.u.uStorm);
      gl.uniform2f(weather.u.uResolution, marchWidth, marchHeight);
      gl.uniform1i(weather.u.uFrame, frame);
      gl.uniform1i(weather.u.uSteps, marchSteps);
      targetTex(rawTex, marchWidth, marchHeight);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      const histSrc = accumPing;
      const histDst = 1 - accumPing;
      bindQuad(accum);
      bindTex(0, rawTex, accum.u.uRaw);
      bindTex(1, accumWarm ? accumTex[histSrc] : rawTex, accum.u.uHistory);
      gl.uniform1f(accum.u.uBlend, accumWarm ? 0.28 : 1.0);
      targetTex(accumTex[histDst], marchWidth, marchHeight);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      accumPing = histDst;
      accumWarm = true;

      bindQuad(copy);
      bindTex(0, accumTex[accumPing], copy.u.uSrc);
      targetTex(weatherTex, viewWidth, viewHeight);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, displayWidth, displayHeight);
      bindQuad(composite);
      bindTex(0, weatherTex, composite.u.uWeather);
      bindTex(1, inkTex[inkPing], composite.u.uInk);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    function render() {
      raf = 0;
      if (!running || hidden) return;
      raf = requestAnimationFrame(render);
      if (displayWidth <= 0 || displayHeight <= 0) return;
      if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
        canvas.width = displayWidth;
        canvas.height = displayHeight;
      }
      stepSim();
      frame++;
      drawWeather();
      fpsFrames++;
      const now = performance.now();
      if (!fpsLast) fpsLast = now;
      if (now - fpsLast >= 500) {
        if (onFps) onFps((fpsFrames * 1000) / (now - fpsLast));
        fpsFrames = 0;
        fpsLast = now;
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
      setHidden(value) {
        hidden = value;
        start();
      },
      setRunning(value) {
        running = value;
        start();
      },
      setSteps(value) {
        marchSteps = Math.max(16, Math.min(96, value | 0));
      },
      setOnFps(fn) {
        onFps = fn;
      },
      setSystems(list) {
        systems = (list || []).map((s) => {
          const at = screenToSim(s.sx, s.sy);
          const kind = s.kind === "H" || s.kind < 0 ? -1 : 1;
          return {
            kind,
            x: at.x,
            y: at.y,
            radius: s.radius || 0.22,
            spin: (s.spin || 1.55) * kind,
            born: s.born || 0,
            segs: (s.segs || []).map((seg) => {
              const a = screenToSim(seg.ax, seg.ay);
              const b = screenToSim(seg.bx, seg.by);
              return { ax: a.x, ay: a.y, bx: b.x, by: b.y };
            }),
          };
        });
      },
      addWarm(w) {
        const at = screenToSim(w.sx, w.sy);
        warms.push({
          x: at.x,
          y: at.y,
          radius: w.radius || 0.22,
          amount: w.amount || 0.45,
        });
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
          force: len > 1e-6 ? Math.min(len * 70, 2.2) : 0,
          spin: imp.spin || 0,
          converge: imp.converge || 0,
          heat: imp.heat || 0,
          moist: imp.moist || 0,
        });
      },
      addFront(seg) {
        const a = screenToSim(seg.ax, seg.ay);
        const b = screenToSim(seg.bx, seg.by);
        fronts.push({
          ax: a.x,
          ay: a.y,
          bx: b.x,
          by: b.y,
          width: seg.width || 0.035,
          cold: seg.cold || 0.045,
          moist: seg.moist || 0.08,
          along: seg.along || 0.22,
          converge: seg.converge || 0.16,
          spin: seg.spin || 0.12,
        });
        stampInk(
          a.x,
          a.y,
          b.x,
          b.y,
          Math.max(0.004, (seg.width || 0.035) * 0.32),
          0.7
        );
      },
    };
  }

  scope.createRenderer = createRenderer;
})(typeof self !== "undefined" ? self : window);

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
  vec2 d = p - a;
  d -= round(d);
  return d;
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
uniform vec2 uResolution;
uniform int uFrame;
in vec2 vUv;
out vec4 fragColor;
${simLib}

#define STEPS 40
#define LIGHT_STEPS 4

const vec3 SUN_DIR = normalize(vec3(0.42, 0.78, 0.46));
const float SIGMA = 14.0;
const float SIGMA_L = 7.5;

float worleyFbm(vec4 n) {
  return n.g * 0.6 + n.b * 0.28 + n.a * 0.12;
}
float remap(float v, float lo, float hi) {
  return clamp((v - lo) / max(hi - lo, 1e-4), 0.0, 1.0);
}

float cloudDensity(vec3 p, bool cheap) {
  if (p.y < 0.0 || p.y > 1.05) return 0.0;
  vec4 n0 = texture(uNoise3, vec3(p.x, p.y * 0.7, p.z) * 3.2);
  vec4 n1 = texture(uNoise3, vec3(p.x, p.y * 0.95, p.z) * 5.6 + 0.19);
  vec2 warp = (n0.gb - 0.5) * mix(0.018, 0.068, p.y) + (n1.ba - 0.5) * mix(0.01, 0.038, p.y);
  vec2 uv = p.xz + warp;
  vec4 f = texture(uField, uv);
  float qc = f.y;
  float temp = f.z;
  float rain = f.w;
  float cover = smoothstep(0.02, 0.16, qc);
  if (cover < 0.01) return 0.0;

  float unstable = clamp((temp - tEquilibrium(uv)) * 5.5 + cover * 0.35, 0.0, 1.0);
  float wet = smoothstep(0.05, 0.45, rain);
  vec4 bump = texture(uNoise3, vec3(uv.x, 0.15, uv.y) * 3.6);
  float top = mix(0.28, 1.0, unstable * 0.75 + cover * 0.25) * mix(0.5, 1.08, worleyFbm(bump));
  top *= 1.0 - 0.35 * wet;
  float h = p.y / max(top, 1e-4);
  if (h > 1.0) return 0.0;

  float grad = smoothstep(0.0, 0.16, h) * smoothstep(1.0, 0.7, h);
  float edge = 1.0 - cover;
  float mid = smoothstep(0.06, 0.26, h) * smoothstep(0.94, 0.52, h);
  float puff = mix(n0.r, worleyFbm(n0), 0.55);
  cover *= mix(1.0, puff, edge * (0.4 + 0.35 * mid));
  if (cover < 0.008) return 0.0;

  vec3 q = vec3(uv.x, p.y, uv.y) * vec3(3.4, 3.8, 3.4);
  vec4 s = texture(uNoise3, q);
  float base = mix(s.r, worleyFbm(s), 0.42);
  float d = remap(base * grad, edge * 0.16, 1.0);
  if (!cheap && d > 0.0) {
    vec4 t = texture(uNoise3, q * 2.8 + 0.31);
    d = remap(d, worleyFbm(t) * mix(0.08, 0.18, h) * (0.35 + 0.65 * edge), 1.0);
  }
  return clamp(d, 0.0, 1.0) * mix(2.0, 2.7, cover);
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

void main() {
  vec2 nd = (gl_FragCoord.xy - 0.5 * uResolution) / uResolution.y;
  vec3 ro = vec3(vUv.x, 1.55, vUv.y);
  vec3 rd = normalize(vec3(nd.x * 0.2, -1.0, nd.y * 0.2));
  float t0 = (1.05 - ro.y) / rd.y;
  float t1 = (0.0 - ro.y) / rd.y;
  float dt = (t1 - t0) / float(STEPS);
  float jitter = fract(texture(uBlueNoise, gl_FragCoord.xy / 128.0).r + float(uFrame % 24) * 0.618034);
  float t = t0 + dt * jitter;
  float trans = 1.0;
  float acc = 0.0;

  for (int i = 0; i < STEPS; i++) {
    if (trans < 0.03) break;
    vec3 pos = ro + rd * t;
    float den = cloudDensity(pos, false);
    if (den > 0.004) {
      float light = exp(-lightTau(pos, jitter) * SIGMA_L);
      float powder = 1.0 - exp(-den * 4.2);
      float h = clamp(pos.y, 0.0, 1.0);
      float rain = texture(uField, pos.xz).w;
      float lit = mix(0.22, 1.05, light) * mix(0.7, 1.08, powder);
      lit *= mix(1.0, 0.62, smoothstep(0.06, 0.5, rain) * (1.0 - h));
      float alpha = 1.0 - exp(-den * dt * SIGMA);
      acc += trans * alpha * lit;
      trans *= 1.0 - alpha;
    }
    t += dt;
  }

  float lum = acc / (1.0 + acc * 0.35);
  fragColor = vec4(vec3(clamp(lum, 0.0, 1.0)), 1.0);
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
  float cloud = texture(uWeather, vUv).r;
  float ink = texture(uInk, vUv).r;
  float lum = max(cloud, ink * 0.55);
  fragColor = vec4(vec3(lum), 1.0);
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
      "uResolution",
      "uFrame",
    ]);
    const accum = makeProgram(blitVs, `#version 300 es
precision highp float;
uniform sampler2D uRaw;
uniform sampler2D uHistory;
uniform float uBlend;
in vec2 vUv;
out vec4 fragColor;
void main() {
  fragColor = vec4(mix(texture(uHistory, vUv).rgb, texture(uRaw, vUv).rgb, uBlend), 1.0);
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
    let running = false;
    let hidden = false;
    let raf = 0;
    let frame = 0;
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
      gl.uniform2f(weather.u.uResolution, marchWidth, marchHeight);
      gl.uniform1i(weather.u.uFrame, frame);
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

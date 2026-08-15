import { writeFileSync } from "node:fs";

const SEED = 0x6d656e646c65;
const RS = 0.25;
const OUT = "js/clusters.json";
const ARMS = 3;
const PITCH = 0.26;
const R_CORE = 18;
const R_DISK = 230;
const R_SCALE = 72;
const DUST_MUL = 0.82;

function stellarRadiusRs(rng) {
  const t = rng();
  if (t < 0.72) return 0.12 + rng() * 0.23;
  if (t < 0.94) return 0.35 + rng() * 0.3;
  return 0.65 + rng() * 0.3;
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

function round1(n) {
  return Math.round(n * 10) / 10;
}

function round3(n) {
  return Math.round(n * 1000) / 1000;
}

function gauss(rng) {
  const u = Math.max(1e-6, rng());
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(Math.PI * 2 * v);
}

function expRadius(rng, rMin, rMax, scale) {
  const a = Math.exp(-rMin / scale);
  const b = Math.exp(-rMax / scale);
  return -scale * Math.log(Math.max(1e-8, a - rng() * (a - b)));
}

function starColor(rng) {
  const t = rng();
  if (t < 0.32) {
    const u = t / 0.32;
    return [round3(0.82 + u * 0.18), round3(0.04 + u * 0.22), round3(0.02 + u * 0.08)];
  }
  if (t < 0.68) {
    const u = (t - 0.32) / 0.36;
    return [round3(1), round3(0.26 + u * 0.52), round3(0.1 + u * 0.28)];
  }
  const u = (t - 0.68) / 0.32;
  return [round3(1), round3(0.78 + u * 0.22), round3(0.38 + u * 0.62)];
}

function starPoint(rng, x, y, z) {
  const color = starColor(rng);
  return [round1(x), round1(y), round1(z), round3(stellarRadiusRs(rng) * RS), color[0], color[1], color[2]];
}

function armTheta(arm, r, scatter) {
  return arm * ((Math.PI * 2) / ARMS) + PITCH * Math.log(Math.max(r, R_CORE) / R_CORE) + scatter;
}

function diskPos(rng, r, arm, scatter, yScale) {
  const theta = armTheta(arm, r, scatter);
  const y = gauss(rng) * yScale;
  return [r * Math.cos(theta), y, r * Math.sin(theta)];
}

function bulgePos(rng) {
  const r = 10 + Math.pow(rng(), 0.62) * 36;
  const u = rng();
  const v = rng();
  const theta = Math.PI * 2 * u;
  const z = 2 * v - 1;
  const s = Math.sqrt(Math.max(0, 1 - z * z));
  return [s * Math.cos(theta) * r, z * r * 0.52, s * Math.sin(theta) * r];
}

function buildPoints(rng, origin, n, spread) {
  const points = [starPoint(rng, 0, 0, 0)];
  if (n <= 1) return points;
  const ox = origin[0];
  const oz = origin[2];
  const rl = Math.hypot(ox, oz) || 1;
  const rx = ox / rl;
  const rz = oz / rl;
  const tx = -rz;
  const tz = rx;
  for (let i = 1; i < n; i++) {
    const dr = (rng() < 0.5 ? -1 : 1) * (7 + rng() * spread);
    const dt = (rng() - 0.5) * spread * 0.7;
    const dy = (rng() - 0.5) * spread * 0.1;
    points.push(starPoint(rng, rx * dr + tx * dt, dy, rz * dr + tz * dt));
  }
  return points;
}

function buildDust(rng, count, radius, size, opacity, height, stretch) {
  return {
    count: Math.max(1, Math.round(count * DUST_MUL)),
    radius: round1(radius),
    size: round3(size),
    opacity: round3(opacity * DUST_MUL),
    height: round3(height),
    stretch: round3(stretch),
    color: [1, 1, 1],
  };
}

function clusterOf(rng, origin, points, dust) {
  const cluster = {
    origin: origin.map(round1),
    radius: round3((0.16 + rng() * 0.12) * RS),
    points,
  };
  if (dust) cluster.dust = dust;
  return cluster;
}

const rng = mulberry32(SEED);
const clusters = [];

for (let i = 0; i < 860; i++) {
  const arm = i % ARMS;
  const r = expRadius(rng, R_CORE, R_DISK, R_SCALE);
  const origin = diskPos(rng, r, arm, (rng() - 0.5) * 0.2, 1.6 + r * 0.01);
  const n = rng() < 0.42 ? 1 : 2;
  const dust = buildDust(
    rng,
    24 + ((rng() * 16) | 0),
    16 + rng() * 16,
    3.4 + rng() * 3.0,
    0.024 + rng() * 0.016,
    0.09 + rng() * 0.05,
    2.2 + rng() * 0.9
  );
  clusters.push(clusterOf(rng, origin, buildPoints(rng, origin, n, 10 + rng() * 9), dust));
}

for (let i = 0; i < 380; i++) {
  const arm = (rng() * ARMS) | 0;
  const r = expRadius(rng, R_CORE + 6, R_DISK, R_SCALE * 1.15);
  const origin = diskPos(rng, r, arm, (rng() - 0.5) * 0.72, 2.1 + r * 0.012);
  const n = rng() < 0.62 ? 1 : 2;
  const dust =
    rng() < 0.7
      ? buildDust(
          rng,
          10 + ((rng() * 8) | 0),
          9 + rng() * 10,
          2.2 + rng() * 1.8,
          0.01 + rng() * 0.01,
          0.11 + rng() * 0.06,
          1.2 + rng() * 0.5
        )
      : null;
  clusters.push(clusterOf(rng, origin, buildPoints(rng, origin, n, 8 + rng() * 7), dust));
}

for (let i = 0; i < 110; i++) {
  const origin = bulgePos(rng);
  const n = 2 + ((rng() * 3) | 0);
  const dust = buildDust(
    rng,
    18 + ((rng() * 12) | 0),
    8 + rng() * 9,
    2.4 + rng() * 2,
    0.02 + rng() * 0.014,
    0.38 + rng() * 0.18,
    0.85 + rng() * 0.3
  );
  clusters.push(clusterOf(rng, origin, buildPoints(rng, origin, n, 5 + rng() * 5), dust));
}

for (let i = 0; i < 70; i++) {
  const arm = (rng() * ARMS) | 0;
  const r = 260 + rng() * 900;
  const origin = diskPos(rng, r, arm, (rng() - 0.5) * 0.9, 4 + r * 0.004);
  const dust =
    rng() < 0.55
      ? buildDust(
          rng,
          10 + ((rng() * 8) | 0),
          10 + rng() * 12,
          2.2 + rng() * 2,
          0.008 + rng() * 0.008,
          0.12 + rng() * 0.08,
          1.3 + rng() * 0.5
        )
      : null;
  clusters.push(clusterOf(rng, origin, buildPoints(rng, origin, 1, 6), dust));
}

const payload = { clusters };
writeFileSync(OUT, `${JSON.stringify(payload)}\n`);
console.log(`Wrote ${clusters.length} clusters to ${OUT}`);

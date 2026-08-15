import { writeFileSync } from "node:fs";

const COUNT = 500;
const NEAR_COUNT = 450;
const SEED = 0x6d656e646c65;
const RS = 0.25;
const OUT = "js/clusters.json";
const NEAR_MIN = 55;
const NEAR_MAX = 240;
const FAR_MIN = 320;
const FAR_MAX = 3600;

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

function pick(rng, arr) {
  return arr[(rng() * arr.length) | 0];
}

function round1(n) {
  return Math.round(n * 10) / 10;
}

function round3(n) {
  return Math.round(n * 1000) / 1000;
}

function randomDir(rng) {
  const u = rng();
  const v = rng();
  const theta = 2 * Math.PI * u;
  const z = 2 * v - 1;
  const r = Math.sqrt(1 - z * z);
  return [r * Math.cos(theta), z * 0.35 + (rng() - 0.5) * 0.12, r * Math.sin(theta)];
}

function randomShell(rng, minR, maxR) {
  const u = rng();
  const r = minR + (maxR - minR) * Math.cbrt(u);
  const dir = randomDir(rng);
  return [dir[0] * r, dir[1] * r * 0.18, dir[2] * r];
}

function starPoint(rng, x, y, z) {
  const color = starColor(rng);
  return [round1(x), round1(y), round1(z), round3(stellarRadiusRs(rng) * RS), color[0], color[1], color[2]];
}

function buildPoints(rng) {
  const n = 3 + ((rng() * 6) | 0);
  const points = [starPoint(rng, 0, 0, 0)];
  let x = 0;
  let y = 0;
  let z = 0;
  let yaw = rng() * Math.PI * 2;
  let pitch = (rng() - 0.5) * 0.35;

  for (let i = 1; i < n; i++) {
    yaw += (rng() - 0.5) * 1.4;
    pitch = pitch * 0.65 + (rng() - 0.5) * 0.25;
    const step = 1.8 + rng() * 3.8;
    x += Math.cos(yaw) * Math.cos(pitch) * step;
    y += Math.sin(pitch) * step * 0.35 + (rng() - 0.5) * 0.4;
    z += Math.sin(yaw) * Math.cos(pitch) * step;

    if (rng() < 0.14 && i > 1) {
      const bx = x + (rng() - 0.5) * 2.4;
      const by = y + (rng() - 0.5) * 0.8;
      const bz = z + (rng() - 0.5) * 2.4;
      points.push(starPoint(rng, bx, by, bz));
    }

    points.push(starPoint(rng, x, y, z));
  }

  return points;
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

function dustColor() {
  return [1, 1, 1];
}

function buildDust(rng) {
  if (rng() < 0.38) return null;
  const radius = 3.8 + rng() * 6.5;
  return {
    count: 12 + ((rng() * 10) | 0),
    radius: round1(radius),
    size: round3(radius * (0.32 + rng() * 0.16)),
    opacity: round3(0.034 + rng() * 0.018),
    color: dustColor(),
  };
}

function buildCluster(rng, origin) {
  const radius = round3((0.18 + rng() * 0.14) * RS);
  const ampScale = 4 + rng() * 14;
  const dir = randomDir(rng);
  const glide = {
    amp: [
      round1(dir[0] * ampScale),
      round1((rng() - 0.5) * 2.4),
      round1(dir[2] * ampScale),
    ],
    speed: round3(0.006 + rng() * 0.008),
    phase: round1(rng() * Math.PI * 2),
  };
  const dust = buildDust(rng);
  const cluster = {
    origin: origin.map(round1),
    radius,
    glide,
    points: buildPoints(rng),
  };
  if (dust) cluster.dust = dust;
  return cluster;
}

const rng = mulberry32(SEED);
const clusters = [];

for (let i = 0; i < NEAR_COUNT; i++) {
  clusters.push(buildCluster(rng, randomShell(rng, NEAR_MIN, NEAR_MAX)));
}
for (let i = NEAR_COUNT; i < COUNT; i++) {
  clusters.push(buildCluster(rng, randomShell(rng, FAR_MIN, FAR_MAX)));
}

const payload = { clusters };
writeFileSync(OUT, `${JSON.stringify(payload, null, 2)}\n`);
console.log(`Wrote ${clusters.length} clusters to ${OUT}`);

'use strict';

// ── Guard: only run on homepage ────────────────────────────
(function () {

const histEl = document.getElementById('hist');
if (!histEl) return;

// ── Constants ──────────────────────────────────────────────
const COL_COUNT = 70;
const MAX_ROWS = 14;
const SPRING_K = 0.003;    // stiffness — low = more lag
const SPRING_DAMP = 0.92;  // damping — high = less overshoot
const SPRING_VEL_CAP = 0.004;

// ── DOM references ─────────────────────────────────────────
const section = document.getElementById('distSection');
const navEl = document.getElementById('nav');

// ── Build histogram columns ────────────────────────────────
const cols = [];

for (let i = 0; i < COL_COUNT; i++) {
  const col = document.createElement('div');
  col.className = 'hist-col';
  histEl.appendChild(col);
  cols.push({ el: col, count: 0 });
}

// ── State ──────────────────────────────────────────────────
const display = new Array(COL_COUNT).fill(0.05);
const idleValues = new Array(COL_COUNT).fill(0).map(() => 0.04 + Math.random() * 0.05);
const activeNoise = new Array(COL_COUNT).fill(0);

let isHovering = false;
let targetCenter = 0.5;
let springCenter = 0.5;
let springVel = 0;

// ── Math helpers ───────────────────────────────────────────
function gaussian(x, mean, sigma) {
  return Math.exp(-0.5 * ((x - mean) / sigma) ** 2);
}

function clamp(val, min, max) {
  return Math.max(min, Math.min(max, val));
}

// ── Idle simulation ────────────────────────────────────────
function tickIdle() {
  for (let i = 0; i < COL_COUNT; i++) {
    // Random walk: very small nudge each frame
    idleValues[i] += (Math.random() - 0.5) * 0.006;
    // Gently pull back toward a baseline
    idleValues[i] += (0.065 - idleValues[i]) * 0.008;
    // Occasional spike — simulate a new sample landing in this bin
    if (Math.random() < 0.003) {
      idleValues[i] += 0.04 + Math.random() * 0.06;
    }
    idleValues[i] = clamp(idleValues[i], 0.01, 0.25);
  }
}

// ── Spring physics ─────────────────────────────────────────
function updateSpring() {
  if (isHovering) {
    const force = (targetCenter - springCenter) * SPRING_K;
    springVel += force;
    springVel *= SPRING_DAMP;
    springVel = clamp(springVel, -SPRING_VEL_CAP, SPRING_VEL_CAP);
  } else {
    const force = (0.5 - springCenter) * 0.004;
    springVel += force;
    springVel *= SPRING_DAMP;
    tickIdle();
  }
  springCenter += springVel;
}

// ── Per-bar distribution calculation ───────────────────────
function computeActiveBar(i, speed) {
  const x = i / (COL_COUNT - 1);
  const dist = x - springCenter;
  const movingRight = springVel > 0;

  // Asymmetric sigma: trailing side stretches, leading side tightens
  const stretchFactor = Math.min(speed * 60, 0.4);
  let sigmaLeft, sigmaRight;

  if (movingRight) {
    sigmaRight = 0.09 - stretchFactor * 0.3;
    sigmaLeft = 0.09 + stretchFactor * 0.4;
  } else {
    sigmaLeft = 0.09 - stretchFactor * 0.3;
    sigmaRight = 0.09 + stretchFactor * 0.4;
  }

  sigmaLeft = Math.max(sigmaLeft, 0.04);
  sigmaRight = Math.max(sigmaRight, 0.04);

  const sigma = dist < 0 ? sigmaLeft : sigmaRight;
  let targetVal = gaussian(x, springCenter, sigma);

  // Random walk noise on top of the distribution
  activeNoise[i] += (Math.random() - 0.5) * 0.008;
  activeNoise[i] *= 0.95;
  if (Math.random() < 0.004) {
    activeNoise[i] += (Math.random() - 0.5) * 0.06;
  }
  activeNoise[i] = clamp(activeNoise[i], -0.15, 0.15);
  targetVal = Math.max(0, targetVal + activeNoise[i] * targetVal);

  // Per-bar lerp: bars closer to center converge faster
  const proximity = 1 - Math.min(Math.abs(dist) * 3, 1);
  const barLerp = clamp(0.02 + proximity * 0.05 + speed * 4, 0, 0.2);
  display[i] += (targetVal - display[i]) * barLerp;
}

function computeIdleBar(i) {
  display[i] += (idleValues[i] - display[i]) * 0.05;
  activeNoise[i] *= 0.95;
}

// ── DOM update ─────────────────────────────────────────────
function updateColumn(i) {
  const rows = Math.round(display[i] * MAX_ROWS);
  if (cols[i].count === rows) return;

  const col = cols[i].el;
  const diff = rows - cols[i].count;

  if (diff > 0) {
    for (let r = 0; r < diff; r++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      col.appendChild(cell);
    }
  } else {
    for (let r = 0; r < -diff; r++) {
      if (col.lastChild) col.removeChild(col.lastChild);
    }
  }
  cols[i].count = rows;
}

// ── Main render loop ───────────────────────────────────────
function render() {
  updateSpring();

  const speed = Math.abs(springVel);

  for (let i = 0; i < COL_COUNT; i++) {
    if (isHovering || speed > 0.0005) {
      computeActiveBar(i, speed);
    } else {
      computeIdleBar(i);
    }
    updateColumn(i);
  }

  requestAnimationFrame(render);
}

// ── Event listeners ────────────────────────────────────────
section.addEventListener('mousemove', (e) => {
  const navRect = navEl.getBoundingClientRect();
  const x = e.clientX - navRect.left;
  const normalized = x / navRect.width;
  targetCenter = 0.14 + clamp(normalized, 0, 1) * 0.72;
  isHovering = true;
});

section.addEventListener('mouseleave', () => {
  isHovering = false;
});

// ── Start ──────────────────────────────────────────────────
render();

})();

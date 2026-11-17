// Paper figures:
//  - next to the Matcha paper: a ribosome density (PDB 7K00) as a point cloud; rotate a particle
//    to match a reference (illustration only)
//  - next to the ICLR paper: microscopy results from the paper, with an uncertainty toggle

// ---------- quaternions [w, x, y, z]
const Q = {
  mul: (a, b) => [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ],
  axis: (x, y, z, ang) => { const s = Math.sin(ang / 2), n = Math.hypot(x, y, z) || 1; return [Math.cos(ang / 2), x / n * s, y / n * s, z / n * s]; },
  norm: (q) => { const n = Math.hypot(...q); return q.map((v) => v / n); },
  angle: (a, b) => 2 * Math.acos(Math.min(1, Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]))),
  slerp(a, b, t) {
    let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    if (d < 0) { b = b.map((v) => -v); d = -d; }
    if (d > 0.9995) return Q.norm(a.map((v, i) => v + t * (b[i] - v)));
    const th = Math.acos(d), s = Math.sin(th);
    return a.map((v, i) => (Math.sin((1 - t) * th) * v + Math.sin(t * th) * b[i]) / s);
  },
  mat: ([w, x, y, z]) => [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ],
};

let seed = 11;
const rand = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function readColors() {
  const s = getComputedStyle(document.documentElement);
  return { ink: s.getPropertyValue("--text").trim(), accent: s.getPropertyValue("--accent").trim() };
}
let col = readColors();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { col = readColors(); });

// ---------- point-cloud renderer (draggable, optional per-point noise)
class Cloud {
  constructor(canvas, pts, { drag = true, onDrag = null } = {}) {
    this.c = canvas; this.ctx = canvas.getContext("2d"); this.pts = pts; this.N = pts.length;
    this.q = Q.axis(1, 0, 0, -0.5);
    this.sigma = 0;
    this.eps = Float32Array.from({ length: this.N * 3 }, gauss);
    this.proj = new Float32Array(this.N * 3); this.order = [...Array(this.N).keys()];
    new ResizeObserver(() => this.resize()).observe(canvas); this.resize();
    if (!drag) return;
    let down = false, px = 0, py = 0;
    canvas.addEventListener("pointerdown", (e) => { down = true; px = e.clientX; py = e.clientY; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener("pointerup", () => { down = false; });
    canvas.addEventListener("pointercancel", () => { down = false; });
    canvas.addEventListener("pointermove", (e) => {
      if (!down) return;
      const dx = e.clientX - px, dy = e.clientY - py, ang = Math.hypot(dx, dy) * 0.012;
      px = e.clientX; py = e.clientY;
      if (ang > 0) this.q = Q.norm(Q.mul(Q.axis(dy, dx, 0, ang), this.q));
      onDrag && onDrag();
    });
  }
  resize() {
    this.dpr = Math.min(devicePixelRatio || 1, 2);
    this.W = this.c.clientWidth; this.H = this.c.clientHeight;
    this.c.width = this.W * this.dpr; this.c.height = this.H * this.dpr;
  }
  draw(dt) {
    const { N, pts, eps, proj, ctx, W, H } = this;
    const R = Q.mat(this.q), scale = Math.min(W, H) * 0.42, cam = 4, cx = W / 2, cy = H / 2;
    const s = this.sigma * 0.3, k = 0.8, ou = Math.sqrt(2 * k * dt);
    for (let i = 0; i < N; i++) {
      const j = i * 3, p = pts[i];
      if (s > 0 && !reduceMotion) for (let d = 0; d < 3; d++) eps[j + d] += -k * eps[j + d] * dt + ou * gauss();
      const x = p[0] + s * eps[j], y = p[1] + s * eps[j + 1], z = p[2] + s * eps[j + 2];
      const X = R[0] * x + R[1] * y + R[2] * z, Y = R[3] * x + R[4] * y + R[5] * z, Z = R[6] * x + R[7] * y + R[8] * z;
      const f = cam / (cam - Z);
      proj[j] = cx + X * scale * f; proj[j + 1] = cy - Y * scale * f; proj[j + 2] = Z;
    }
    this.order.sort((a, b) => proj[a * 3 + 2] - proj[b * 3 + 2]);
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const dot = Math.max(0.7, W / 300);
    const stride = W < 220 ? 2 : 1; // fewer dots in small figures
    for (const i of this.order) {
      if (i % stride) continue;
      const depth = Math.max(0, Math.min(1, (proj[i * 3 + 2] + 1) / 2));
      ctx.globalAlpha = 0.12 + 0.8 * depth;
      ctx.fillStyle = pts[i][3] ? col.accent : col.ink;
      ctx.beginPath();
      ctx.arc(proj[i * 3], proj[i * 3 + 1], dot * (0.5 + 0.9 * depth), 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }
}

// ---------- animation loop, only while a figure is on screen
const demos = [];
let last = performance.now(), scheduled = false;
function schedule() {
  if (scheduled || document.hidden || !demos.some((d) => d.visible)) return;
  scheduled = true;
  requestAnimationFrame(loop);
}
function loop(now) {
  scheduled = false;
  const dt = Math.min((now - last) / 1000, 0.05); last = now;
  for (const d of demos) if (d.visible) d.tick(now, dt);
  schedule();
}
function addDemo(el, tick) {
  const d = { visible: false, tick };
  demos.push(d);
  new IntersectionObserver(([e]) => { d.visible = e.isIntersecting; last = performance.now(); schedule(); }).observe(el);
  return d;
}
document.addEventListener("visibilitychange", () => { last = performance.now(); schedule(); });

// ---------- ICLR paper: toggle the sample with its pixelwise spread over 500 continuations
const spread = document.getElementById("spread");
if (spread) {
  const img = document.getElementById("ours"), out = document.getElementById("spread-readout");
  spread.addEventListener("click", () => {
    const on = spread.getAttribute("aria-pressed") !== "true";
    spread.setAttribute("aria-pressed", on);
    spread.textContent = on ? "Show sample" : "Show uncertainty";
    img.src = `assets/iclr/${on ? "uncertainty" : "consistency"}.png`;
    img.alt = on ? "Pixelwise standard deviation over 500 continuations" : "Continuation with consistency: sharp cell walls";
    out.textContent = on ? "std / mean, 500 samples" : "";
  });
}

// ---------- Matcha paper: rotational alignment (illustration, eases to the known pose)
if (window.RIBOSOME && document.getElementById("ref")) {
  let anim = null;
  const ref = new Cloud(document.getElementById("ref"), window.RIBOSOME, { drag: false });
  const part = new Cloud(document.getElementById("part"), window.RIBOSOME, { onDrag: () => { anim = null; } });
  const out = document.getElementById("align-readout");
  ref.q = Q.norm(Q.mul(Q.axis(0, 1, 0, 0.6), Q.axis(1, 0, 0, -0.4)));
  const randomize = () => {
    part.q = Q.norm(Q.mul(Q.axis(gauss(), gauss(), gauss(), 1.2 + rand() * 1.6), ref.q));
    anim = null;
  };
  randomize();
  document.getElementById("shuffle").addEventListener("click", randomize);
  document.getElementById("align").addEventListener("click", () => { anim = { from: part.q, t0: performance.now() }; });
  addDemo(document.getElementById("part"), (now, dt) => {
    if (anim) {
      const u = Math.min((now - anim.t0) / 1100, 1);
      const e = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
      part.q = Q.slerp(anim.from, ref.q, e);
      if (u >= 1) anim = null;
    }
    ref.draw(dt); part.draw(dt);
    const err = Q.angle(part.q, ref.q) * 180 / Math.PI;
    out.textContent = `error ${err.toFixed(1)}°`;
    out.classList.toggle("ok", err < 1);
  });
}

// ---------- "Cite" buttons copy BibTeX to the clipboard
document.querySelectorAll(".cite").forEach((btn) => {
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(btn.dataset.bibtex);
      btn.textContent = "Copied";
    } catch {
      btn.textContent = "Copy failed";
    }
    setTimeout(() => { btn.textContent = "Cite"; }, 1500);
  });
});

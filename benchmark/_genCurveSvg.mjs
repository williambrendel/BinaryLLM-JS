import fs from "node:fs";
const rows = fs.readFileSync("/tmp/curve.csv", "utf8").trim().split("\n").slice(1).map((l) => l.split(","));
const col = (k) => rows.map((r) => (r[k] === "" || r[k] === undefined ? null : Number(r[k])));
const series = [
  { name: "r1 (native — peels)", data: col(1), color: "#2563eb", dash: "" },
  { name: "r5 (native — bisects)", data: col(2), color: "#dc2626", dash: "" },
  { name: "r5 · r1-parts, retune k", data: col(3), color: "#16a34a", dash: "6 4" },
  { name: "r5 · r1-parts, fixed k", data: col(4), color: "#f59e0b", dash: "2 3" },
];
const W = 760, Hh = 460, mL = 62, mR = 210, mT = 34, mB = 52;
const pw = W - mL - mR, ph = Hh - mT - mB;
const nX = rows.length - 1, maxY = 3.0;
const X = (i) => mL + (i / nX) * pw;
const Y = (v) => mT + ph - (v / maxY) * ph;
const poly = (s) => { let pts = [], run = []; s.data.forEach((v, i) => { if (v == null) { if (run.length) { pts.push(run); run = []; } } else run.push(`${X(i).toFixed(1)},${Y(v).toFixed(1)}`); }); if (run.length) pts.push(run); return pts.map((p) => `<polyline fill="none" stroke="${s.color}" stroke-width="2.2" stroke-dasharray="${s.dash}" points="${p.join(" ")}"/>`).join(""); };
let g = "";
for (let t = 0; t <= 3; t += 0.5) { const y = Y(t); g += `<line x1="${mL}" y1="${y}" x2="${mL + pw}" y2="${y}" stroke="#e5e7eb" stroke-width="1"/><text x="${mL - 8}" y="${y + 4}" text-anchor="end" font-size="12" fill="#6b7280">${t.toFixed(1)}</text>`; }
for (let s = 0; s <= 200; s += 40) { const x = X(s); g += `<line x1="${x}" y1="${mT}" x2="${x}" y2="${mT + ph}" stroke="#f3f4f6" stroke-width="1"/><text x="${x}" y="${mT + ph + 18}" text-anchor="middle" font-size="12" fill="#6b7280">${s}</text>`; }
const xc = X(60);
const cross = `<line x1="${xc}" y1="${mT}" x2="${xc}" y2="${mT + ph}" stroke="#9ca3af" stroke-width="1" stroke-dasharray="3 3"/><text x="${xc + 5}" y="${mT + 14}" font-size="11" fill="#6b7280">r5 overtakes r1 (~60)</text>`;
let leg = "";
series.forEach((s, i) => { const y = mT + 10 + i * 22; leg += `<line x1="${mL + pw + 16}" y1="${y}" x2="${mL + pw + 40}" y2="${y}" stroke="${s.color}" stroke-width="2.6" stroke-dasharray="${s.dash}"/><text x="${mL + pw + 46}" y="${y + 4}" font-size="12" fill="#374151">${s.name}</text>`; });
const svg = `<svg viewBox="0 0 ${W} ${Hh}" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif"><rect width="${W}" height="${Hh}" fill="#ffffff"/><text x="${mL}" y="20" font-size="15" font-weight="600" fill="#111827">Global IG vs #splits (best-first) — higher IG, worse tree</text>${g}${cross}${series.map(poly).join("")}${leg}<text x="${mL + pw / 2}" y="${Hh - 12}" text-anchor="middle" font-size="12" fill="#6b7280"># splits</text><text x="16" y="${mT + ph / 2}" text-anchor="middle" font-size="12" fill="#6b7280" transform="rotate(-90 16 ${mT + ph / 2})">global IG (nats)</text></svg>`;
fs.writeFileSync("/tmp/curve.svg", svg);
console.log("wrote /tmp/curve.svg");

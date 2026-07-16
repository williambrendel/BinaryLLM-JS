import fs from "node:fs";
const rows = fs.readFileSync("/tmp/hc.csv", "utf8").trim().split("\n").slice(1).map((l) => l.split(","));
const num = (v) => (v === "" || v === undefined ? null : Number(v));
const xs = rows.map((r) => num(r[0]));
const S = { r1IG: rows.map((r) => num(r[1])), r1CE: rows.map((r) => num(r[2])), r5IG: rows.map((r) => num(r[3])), r5CE: rows.map((r) => num(r[4])) };
const W = 760, Hh = 540, mL = 66, mR = 24, mT = 40, gap = 46;
const pw = W - mL - mR, ph = (Hh - mT - gap - 40) / 2;
const nX = 180;
const X = (i) => mL + (xs[i] / nX) * pw;
const panel = (y0, data, lo, hi, title, fmt) => {
  const Y = (v) => y0 + ph - ((v - lo) / (hi - lo)) * ph;
  let g = `<text x="${mL}" y="${y0 - 8}" font-size="13" font-weight="600" fill="#111827">${title}</text>`;
  const ntick = 4;
  for (let t = 0; t <= ntick; t++) { const v = lo + (t / ntick) * (hi - lo); const y = Y(v); g += `<line x1="${mL}" y1="${y}" x2="${mL + pw}" y2="${y}" stroke="#eceff3" stroke-width="1"/><text x="${mL - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="#6b7280">${fmt(v)}</text>`; }
  for (let s = 0; s <= 180; s += 30) { const x = mL + (s / nX) * pw; g += `<line x1="${x}" y1="${y0}" x2="${x}" y2="${y0 + ph}" stroke="#f6f7f9" stroke-width="1"/><text x="${x}" y="${y0 + ph + 16}" text-anchor="middle" font-size="11" fill="#6b7280">${s}</text>`; }
  const line = (arr, color) => { const pts = []; arr.forEach((v, i) => { if (v != null) pts.push(`${X(i).toFixed(1)},${Y(v).toFixed(1)}`); }); return `<polyline fill="none" stroke="${color}" stroke-width="2.4" points="${pts.join(" ")}"/>`; };
  g += line(data.r1, "#2563eb") + line(data.r5, "#dc2626");
  // mark the held-out minimum region
  return g;
};
let svg = `<svg viewBox="0 0 ${W} ${Hh}" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif"><rect width="${W}" height="${Hh}" fill="#ffffff"/>`;
svg += panel(mT, { r1: S.r1IG, r5: S.r5IG }, 0, 2.9, "in-sample global IG — r5 (red) is HIGHER (IG says r5 wins)", (v) => v.toFixed(1));
svg += panel(mT + ph + gap, { r1: S.r1CE, r5: S.r5CE }, 6.83, 7.13, "held-out cross-entropy — r5 (red) is WORSE at every split (held-out says r1 wins)", (v) => v.toFixed(2));
const xmin = mL + (44 / nX) * pw;
svg += `<line x1="${xmin}" y1="${mT + ph + gap}" x2="${xmin}" y2="${mT + ph + gap + ph}" stroke="#9ca3af" stroke-dasharray="3 3"/><text x="${xmin + 4}" y="${mT + ph + gap + 14}" font-size="10" fill="#6b7280">both best ~44, then overfit ↑</text>`;
svg += `<line x1="${mL + pw - 150}" y1="${Hh - 16}" x2="${mL + pw - 130}" y2="${Hh - 16}" stroke="#2563eb" stroke-width="2.6"/><text x="${mL + pw - 124}" y="${Hh - 12}" font-size="12" fill="#374151">r1</text><line x1="${mL + pw - 90}" y1="${Hh - 16}" x2="${mL + pw - 70}" y2="${Hh - 16}" stroke="#dc2626" stroke-width="2.6"/><text x="${mL + pw - 64}" y="${Hh - 12}" font-size="12" fill="#374151">r5</text><text x="${mL}" y="${Hh - 12}" font-size="12" fill="#6b7280"># splits →</text>`;
svg += "</svg>";
fs.writeFileSync("/tmp/hc.svg", svg);
console.log("wrote /tmp/hc.svg");

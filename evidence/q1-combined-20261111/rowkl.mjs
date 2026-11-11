// Per-position KL(teacher || model) and top-1 agreement, summarised by position range (aggregates only).
import fs from 'node:fs';
const [mf, tf, rowsS] = process.argv.slice(2); const V = 128000, rows = Number(rowsS);
const fm = fs.openSync(mf, 'r'), ft = fs.openSync(tf, 'r'), bm = Buffer.alloc(V * 4), bt = Buffer.alloc(V * 4);
const kl = [], same = [];
for (let r = 0; r < rows; r++) {
  fs.readSync(fm, bm, 0, V * 4, r * V * 4); fs.readSync(ft, bt, 0, V * 4, r * V * 4);
  const m = new Float32Array(bm.buffer, bm.byteOffset, V), t = new Float32Array(bt.buffer, bt.byteOffset, V);
  let mm = -Infinity, tm = -Infinity, am = 0, at = 0;
  for (let j = 0; j < V; j++) { if (m[j] > mm) { mm = m[j]; am = j; } if (t[j] > tm) { tm = t[j]; at = j; } }
  let zm = 0, zt = 0; for (let j = 0; j < V; j++) { zm += Math.exp(m[j] - mm); zt += Math.exp(t[j] - tm); }
  const lm = Math.log(zm), lt = Math.log(zt); let k = 0;
  for (let j = 0; j < V; j++) { const pt = Math.exp(t[j] - tm) / zt; if (pt > 1e-12) k += pt * ((t[j] - tm - lt) - (m[j] - mm - lm)); }
  kl.push(k); same.push(am === at ? 1 : 0);
}
const seg = (a, b) => ({ positions: `${a}-${b - 1}`, klSum: +kl.slice(a, b).reduce((x, y) => x + y, 0).toFixed(2), klMean: +(kl.slice(a, b).reduce((x, y) => x + y, 0) / (b - a)).toFixed(4), top1: `${same.slice(a, b).reduce((x, y) => x + y, 0)}/${b - a}` });
console.log(JSON.stringify([seg(0, Math.min(29, rows)), ...(rows > 29 ? [seg(29, Math.min(64, rows))] : []), ...(rows > 64 ? [seg(64, rows)] : []), seg(0, rows)]));

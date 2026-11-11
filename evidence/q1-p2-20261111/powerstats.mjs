import fs from 'node:fs';
const [log, mac] = process.argv.slice(2);
const text = fs.readFileSync(log, 'utf8');
const mark = (k) => Number((text.match(new RegExp(k + '=(\\d+)')) || [])[1]);
const ev = (text.match(/BENCH_EVIDENCE=(\S+)/) || [])[1];
const S = fs.readFileSync(mac, 'utf8').trim().split('\n').map((l) => JSON.parse(l)).map((s) => ({ t: Date.parse(s.timestamp) / 1000, ane: s.ane_power, cpu: s.cpu_power, gpu: s.gpu_power, all: s.all_power, sys: s.sys_power }));
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const win = (a, b) => S.filter((s) => s.t >= a && s.t <= b);
const stat = (w) => ({ n: w.length, ane: +mean(w.map((s) => s.ane)).toFixed(3), cpu: +mean(w.map((s) => s.cpu)).toFixed(2), all: +mean(w.map((s) => s.all)).toFixed(2), sys: +mean(w.map((s) => s.sys)).toFixed(2) });
const idle = [...win(mark('idle_start') + 2, mark('bench_start') - 1), ...win(mark('bench_end') + 3, mark('idle_end'))];
const I = stat(idle); console.log('idle', JSON.stringify(I));
for (const i of [1, 2, 3]) {
  const f = `${ev}/ane_${i}.json`; const r = JSON.parse(fs.readFileSync(f, 'utf8')); const end = fs.statSync(f).mtimeMs / 1000;
  const w = win(end - r.sustainedSeconds + 2, end - 2); const A = stat(w);
  const tps = r.sustainedSteps / r.sustainedSeconds; const aneNet = A.ane - I.ane;
  console.log(`arm${i}`, JSON.stringify({ steps: r.sustainedSteps, seconds: +r.sustainedSeconds.toFixed(1), passes: r.passes, meanTokPerS: +tps.toFixed(1), medianMs: +r.decodeMedianMS.toFixed(2), medianTokPerS: +(1000 / r.decodeMedianMS).toFixed(1), power: A, aneNetW: +aneNet.toFixed(3), mJPerToken: +(1000 * aneNet / tps).toFixed(1), cpuNetW: +(A.cpu - I.cpu).toFixed(2) }));
  // 10 s bins of step rate inside the arm, to show drift
  const ms = r.stepMS ?? []; 
}

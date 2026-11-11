// Tables for the prefill run: tokens per second and ANE power per benchmark arm.
//   node prefillstats.mjs job.log macmon.jsonl <bench-dir>
import fs from 'node:fs';
const [log, mac, ev] = process.argv.slice(2);
const text = fs.readFileSync(log, 'utf8');
const mark = (k) => Number((text.match(new RegExp(k + '=(\\d+)')) || [])[1]);
const S = fs.readFileSync(mac, 'utf8').trim().split('\n').map((l) => JSON.parse(l))
  .map((s) => ({ t: Date.parse(s.timestamp) / 1000, ane: s.ane_power, cpu: s.cpu_power }));
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const win = (a, b) => S.filter((s) => s.t >= a && s.t <= b);
const stat = (w) => ({ n: w.length, ane: +mean(w.map((s) => s.ane)).toFixed(3), cpu: +mean(w.map((s) => s.cpu)).toFixed(2) });
const idle = stat([...win(mark('idle_start') + 2, mark('bench_start') - 1), ...win(mark('bench_end') + 3, mark('idle_end'))]);
console.log('idle', JSON.stringify(idle));
for (const i of [1, 2, 3]) {
  const f = `${ev}/ane_${i}.json`; const r = JSON.parse(fs.readFileSync(f, 'utf8')); const end = fs.statSync(f).mtimeMs / 1000;
  const per = r.tokensPerCall ?? 1, calls = r.sustainedSteps, tokens = calls * per;
  const A = stat(win(end - r.sustainedSeconds + 2, end - 2)); const tps = tokens / r.sustainedSeconds; const net = A.ane - idle.ane;
  console.log(`arm${i}`, JSON.stringify({ tokensPerCall: per, calls, tokens, seconds: +r.sustainedSeconds.toFixed(1), passes: r.passes,
    meanTokPerS: +tps.toFixed(1), medianCallMs: +r.decodeMedianMS.toFixed(1), medianTokPerS: +(per * 1000 / r.decodeMedianMS).toFixed(1),
    aneW: A.ane, aneNetW: +net.toFixed(3), mJPerToken: +(1000 * net / tps).toFixed(1), cpuW: A.cpu, cpuIdleW: idle.cpu, samples: A.n }));
}

import fs from 'node:fs';
const EOS = 124900;
for (const k of ['sky', 'rope', 'france']) {
  const r = JSON.parse(fs.readFileSync(`${process.argv[2]}/chat_${k}.json`, 'utf8'));
  const P = r.prompt.length, ms = r.stepMS, gen = r.sequence.slice(P);
  const e = gen.indexOf(EOS), G = e < 0 ? gen.length : e + 1;       // tokens produced up to and including end-of-turn
  const sum = (a) => a.reduce((x, y) => x + y, 0);
  const ttft = sum(ms.slice(0, P));                                  // prompt ingestion: step P-1 emits the first new token
  const genMs = sum(ms.slice(P, P + G - 1));                         // steps that emit new tokens 2..G
  const total = ttft + genMs;
  console.log(JSON.stringify({ k, promptTokens: P, steps: ms.length, newTokensToEOS: G, hitEOS: e >= 0,
    loadMS: Object.fromEntries(Object.entries(r.loadMS).map(([a, b]) => [a, Math.round(b)])),
    ttftMs: +ttft.toFixed(1), promptTokPerS: +(P / (ttft / 1000)).toFixed(1),
    genTokPerS: +((G - 1) / (genMs / 1000)).toFixed(1), totalMs: +total.toFixed(0), endToEndTokPerS: +(G / (total / 1000)).toFixed(1),
    firstStepMs: +ms[0].toFixed(1), medStepMs: +r.decodeMedianMS.toFixed(2) }));
  console.log('GEN_' + k + '=' + gen.slice(0, G).join(','));
}

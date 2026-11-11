// Tables for the --prefill hand-off runs.   node chainstats.mjs <dir>
import fs from 'node:fs';
const d = process.argv[2]; const J = (f) => JSON.parse(fs.readFileSync(`${d}/${f}`, 'utf8'));
const eq = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
const old = J('tf_old.json'), plain = J('tf_plain.json'), chain = J('tf_chain.json');
console.log('regression new-vs-published runner, 270 positions identical argmax:', eq(old.argmaxPerStep, plain.argmaxPerStep));
const diff = plain.argmaxPerStep.map((x, i) => (x === chain.argmaxPerStep[i] ? -1 : i)).filter((i) => i >= 0);
console.log('with-vs-without prefill, positions with a different argmax:', JSON.stringify(diff), '| positions 256..269 after the hand-off identical:', eq(plain.argmaxPerStep.slice(256), chain.argmaxPerStep.slice(256)));
console.log('tf chain prefill:', JSON.stringify({ calls: chain.prefill.calls, tokens: chain.prefill.tokens, callMS: chain.prefill.callMS.map((x) => +x.toFixed(1)) }));
const EOS = 124900;
for (const i of [1, 2, 3]) {
  const p = J(`chat_plain_${i}.json`), c = J(`chat_chain_${i}.json`);
  const P = p.prompt.length, gp = p.sequence.slice(P), gc = c.sequence.slice(P);
  const firstDiff = gp.findIndex((x, k) => x !== gc[k]);
  const e = gp.indexOf(EOS);
  const genRate = (r, promptSteps) => { const g = r.stepMS.slice(promptSteps); return +(g.length / (g.reduce((x, y) => x + y, 0) / 1000)).toFixed(1); };
  console.log(`chat run ${i}`, JSON.stringify({ promptTokens: P, newTokens: gp.length, stopAt: e,
    plainFirstNewTokenMS: +p.firstNewTokenMS.toFixed(0), chainFirstNewTokenMS: +c.firstNewTokenMS.toFixed(0),
    prefillMS: +c.prefill.callMS.reduce((x, y) => x + y, 0).toFixed(0), prefillCallMS: c.prefill.callMS.map((x) => +x.toFixed(0)), prefillTokens: c.prefill.tokens,
    oneTokenPromptSteps: c.stepMS.length - (gc.length - 1), plainGenTokPerS: genRate(p, P), chainGenTokPerS: genRate(c, P - c.prefill.tokens),
    sameReply: firstDiff < 0, firstDifferentNewToken: firstDiff }));
  if (i === 1) { console.log('GEN_plain=' + gp.join(',')); if (firstDiff >= 0) console.log('GEN_chain=' + gc.join(',')); }
}

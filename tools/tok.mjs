#!/usr/bin/env node
// tok.mjs: chat message -> token ids for anec-run, and ids -> text.
//
//   npm install @huggingface/tokenizers @huggingface/jinja
//   node tools/tok.mjs encode "Explain in one sentence why the sky is blue."   comma-separated ids
//   node tools/tok.mjs encode --think "..."     leave the reasoning block open
//   node tools/tok.mjs encode --raw "..."       plain text, no chat template
//   node tools/tok.mjs decode 597,9669,4713     ids -> text, special tokens shown
//   node tools/tok.mjs reply out.json           the generated part of an `anec-run --out` record,
//                                               cut at the end-of-turn token
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Tokenizer } from '@huggingface/tokenizers';
import { Template } from '@huggingface/jinja';

const dir = process.env.LFM3_TOKENIZER_DIR
  ?? path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'tokenizer');
const read = (f) => fs.readFileSync(path.join(dir, f), 'utf8');
const config = JSON.parse(read('tokenizer_config.json'));
const tokenizer = new Tokenizer(JSON.parse(read('tokenizer.json')), config);
const encode = (s) => Array.from(tokenizer.encode(s, { add_special_tokens: false }).ids);
const special = (t, fallback) => (typeof t === 'string' ? t : t?.content ?? fallback);
const END_OF_TURN = 124900; // <|im_end|>

const [mode, ...rest] = process.argv.slice(2);
if (mode === 'encode') {
  const flags = new Set(rest.filter((a) => a.startsWith('--')));
  const text = rest.filter((a) => !a.startsWith('--')).join(' ');
  let prompt = text;
  if (!flags.has('--raw')) {
    prompt = new Template(read('chat_template.jinja')).render({
      messages: [{ role: 'user', content: text }],
      add_generation_prompt: true,
      bos_token: special(config.bos_token, '<|startoftext|>'),
      enable_thinking: flags.has('--think'),
    });
  }
  console.log(encode(prompt).join(','));
} else if (mode === 'decode') {
  console.log(tokenizer.decode(rest[0].split(',').map(Number), { skip_special_tokens: false }));
} else if (mode === 'reply') {
  const record = JSON.parse(fs.readFileSync(rest[0], 'utf8'));
  const generated = record.sequence.slice(record.prompt.length);
  const end = generated.indexOf(END_OF_TURN);
  console.log(tokenizer.decode(end < 0 ? generated : generated.slice(0, end), { skip_special_tokens: true }));
} else {
  console.error('usage: tok.mjs encode [--think|--raw] TEXT | decode IDS | reply RECORD.json');
  process.exit(2);
}

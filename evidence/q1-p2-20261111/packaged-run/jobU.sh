#!/bin/bash
# Seat job U (program revision 2): the commands of "Running it", run from the packaged files
# (the published q1 bundle's packages, the p2 programs archive, runner r3, the downloaded
# embedding table), in the folder a user would have.
set -uo pipefail
R=/Users/nayte/ane-hot/scratch/lfm3-release-20261111; U=$R/p2/user; TK=$R/repo/tokenizer/tokenizer.json
cd $U
run() { ./lfm3-ane-runner/anec-run lfm3_q1p2_C288 --model lfm3-ane-runner/model --prefill lfm3_q1p2_C288_prefill32 --tokenizer $TK "$@"; }
echo "=== one prompt $(date -u +%T)"
run --prompt "What is the capital of France?" --out france.json
echo "=== the 105-token note, as text $(date -u +%T)"
NOTE=$(node -e 'const t=JSON.parse(require("fs").readFileSync(process.argv[1],"utf8")).promptText;const a=t.indexOf("user\n")+5,b=t.indexOf("<|im_end|>");process.stdout.write(t.slice(a,b))' $R/p2/chat_long.json)
run --prompt "$NOTE" --max-tokens 64 --out long.json
node -e 'const fs=require("fs");const r=JSON.parse(fs.readFileSync("long.json","utf8"));const want=fs.readFileSync(process.argv[1],"utf8").trim();console.log("tokenizer check long:",r.prompt.join(",")===want?"same ids as the Node tokenizer":"DIFFERENT",r.prompt.length,"tokens; prefill tokens",r.prefill?r.prefill.tokens:0)' $R/repo/evidence/prefill-chain-20261111/ids_long
echo "=== sampled reply, seed 7 $(date -u +%T)"
run --prompt "Write one sentence about the sea." --temperature 0.8 --top-k 40 --seed 7 --max-tokens 48 --out sampled.json
echo "=== three-turn chat $(date -u +%T)"
printf 'And what river runs through it?\nName one museum there.\n' | run --prompt "What is the capital of France?" --chat 1 --max-tokens 96 --out chat.json
echo JOB_DONE

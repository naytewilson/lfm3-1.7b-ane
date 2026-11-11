#!/bin/bash
# Seat job V (longer context, from the packaged files): the commands of "Running it" with --grow,
# in the folder a user would have (published p2 archive, the larger-context archive, runner r4,
# the downloaded embedding table).
set -uo pipefail
R=/Users/nayte/ane-hot/scratch/lfm3-release-20261111; U=$R/p2/user2; TK=$R/repo/tokenizer/tokenizer.json
cd $U
run() { ./lfm3-ane-runner/anec-run lfm3_q1p2_C288 --model lfm3-ane-runner/model --prefill lfm3_q1p2_C288_prefill32 \
        --grow lfm3_q1p2_C1024,lfm3_q1p2_C2048 --tokenizer $TK "$@"; }
echo "=== one long reply (crosses 288) $(date -u +%T)"
run --prompt "Explain, step by step and in detail, how a bicycle stays upright while moving, and what a rider does to steer it." --max-tokens 420 --out long_reply.json 2>&1 | fold -w 300 | cut -c1-300
echo "=== five-turn chat (crosses 288 between turns) $(date -u +%T)"
printf 'What are the three largest cities there?\nWhich of them is on the Mediterranean?\nWhat is that city known for?\nSum up this conversation in two sentences.\n' | run --prompt "What is the capital of France?" --chat 1 --max-tokens 110 --out long_chat.json 2>&1 | fold -w 300 | cut -c1-300
echo JOB_DONE

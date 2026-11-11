#!/bin/bash
# eval_seat.sh <name> — the part of eval.sh that loads Core ML models (runs under the seat).
set -uo pipefail
R=/Users/nayte/ane-hot/scratch/lfm3-release-20261111; N=$R/num; Q=/Users/nayte/ane-hot/models/lfm3-q1-carry-20261111
ROOT=/Users/nayte/Projects/ANE-Lab/worktrees/claude-anec-prefill-chain-20261111/anec
RUN=$ROOT/.build/release/anec-run; ORA=$ROOT/.build/release/anec-oracle
V=$N/$1
{
  for p in $V/s1/*.mlpackage; do
    n=$(basename "$p" .mlpackage)
    "$ORA" plan "$p" --out "$V/$n.plan.json" > /dev/null 2>"$V/$n.plan.err"
    echo "placement $n: $(tr -d '\n ' < "$V/$n.plan.json" | grep -o '"summary":{[^}]*}[^}]*}' | head -1 | cut -c1-200)"
    rm -rf "$HOME/Library/Caches/anec-oracle/com.apple.e5rt.e5bundlecache"
  done
  "$RUN" $V/s1 --model $Q --tokens "$(cat $R/q1/fixture270.ids)" --generate 0 --units ane --dump $V/d1 --out $V/tf_s1.json | grep -E "stateful decode|rror" | cut -c1-200
} > $V/seat.log 2>&1

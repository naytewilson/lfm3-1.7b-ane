#!/bin/bash
# probe_diag.sh <root-with-A-B-C> — run the hardware oracle on each tap package under the seat and
# print, per tapped value, ANE vs fp32 reference and CPU-only vs fp32 reference.
set -uo pipefail
ROOT=/Users/nayte/Projects/ANE-Lab/worktrees/claude-anec-prefill-chain-20261111/anec
A=$ROOT/.build/release/anec
R="${1:?root}"; shift
for d in "${@:-A B C}"; do
  [ -d "$R/$d" ] || continue
  bash $ROOT/scripts/neo_seat_probe.sh "lfm3num-$(basename "$R")-$d" "$R/$d" 2>&1 | grep -E "ANVIL_ANE|CRASH|BLOCKED|SEAT_RC" | cut -c1-160
  base=$(basename "$(ls -d "$R/$d"/*.mlpackage | head -1)" .mlpackage)
  echo "## $d  $base"
  tr -d "\n " < "$R/$d/$base.oracle.json" | grep -o '"summary":{[^}]*}[^}]*}' | head -1 | cut -c1-300
  for r in $(ls -tr "$R/$d/$base".*.ref.f32); do
    t=$(basename "$r" .ref.f32); t=${t#$base.}
    a="$R/$d/$base.$t.ane.f32"; c="$R/$d/$base.$t.cpu.f32"
    [ -f "$a" ] || { echo "| $t | missing |"; continue; }
    echo "| $t | $($A compare "$a" "$r" --rows 8 --rowstats 1 | sed -E 's/compare: n=[0-9]+ //; s/rms=[^ ]+ //; s/ top1=[0-9/]+//' | cut -c1-170) | $($A compare "$c" "$r" --rows 8 --rowstats 1 | sed -E 's/compare: n=[0-9]+ //; s/rms=[^ ]+ //; s/ top1=[0-9/]+//' | cut -c1-60) |"
  done
done

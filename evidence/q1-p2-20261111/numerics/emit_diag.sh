#!/bin/bash
# emit_diag.sh <anec-binary> <out-dir> [extra decoder flags...]
# Emits three stateless tap packages of the q1 LFM3 checkpoint (S=8, seeded token ids):
#   A  layer 0 (conv), every intermediate value
#   B  layers 0-2, every intermediate value of layer 2 (full attention)
#   C  layers 0-5, the stream after each operator and each MLP
# No Core ML load here; the device comparison runs later under the seat.
set -euo pipefail
A="${1:?anec binary}"; R="${2:?out dir}"; shift 2
M=/Users/nayte/ane-hot/models/lfm3-q1-carry-20261111
mkdir -p "$R/A" "$R/B" "$R/C"
conv() { local L=L$1; echo "${L}_norm1,${L}_inproj,${L}_bx,${L}_conv,${L}_cy,${L}_outproj,${L}_postnorm1,${L}_res1,${L}_norm2,${L}_gate,${L}_gate_act,${L}_up,${L}_gated,${L}_down,${L}_postnorm2,${L}_res2"; }
attn() { local L=L$1; echo "${L}_norm1,${L}_q,${L}_k,${L}_v,${L}_qn,${L}_kn,${L}_q_rope,${L}_k_rope,${L}_scores,${L}_scaled,${L}_probs,${L}_attn,${L}_hgate_s,${L}_attn_gated,${L}_o,${L}_postnorm1,${L}_res1,${L}_norm2,${L}_gate,${L}_gate_act,${L}_up,${L}_gated,${L}_down,${L}_postnorm2,${L}_res2"; }
stream() { local s=""; for l in $(seq 0 $1); do s="$s,L${l}_res1,L${l}_res2"; done; echo "${s#,}"; }
"$A" decoder --model $M --seq 8 --layers 0:1 --embed 1 --head 0 --taps "$(conv 0)" --ref 1 --tag A --out "$R/A" "$@" | cut -c1-160
"$A" decoder --model $M --seq 8 --layers 0:3 --embed 1 --head 0 --taps "$(attn 2)" --ref 1 --tag B --out "$R/B" "$@" | cut -c1-160
"$A" decoder --model $M --seq 8 --layers 0:6 --embed 1 --head 0 --taps "$(stream 5)" --ref 1 --tag C --out "$R/C" "$@" | cut -c1-160
du -sh "$R"/A "$R"/B "$R"/C

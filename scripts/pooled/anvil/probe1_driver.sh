#!/bin/bash
# probe1: ensemble-mode sweep over the 12 C7_E runs (G1 decorrelation probe).
# Modes: island_champions (paired baseline, identical code path),
#        diverse (N=8, gate 3, target-free greedy min-max-similarity),
#        all_elites (64-member scaling), topn (N=16 round-robin).
BASE=/anvil/scratch/x-jchang5
SCRIPTS=$HOME/probe_scripts
OUT=$BASE/probe1
mkdir -p "$OUT"
: > "$OUT/commands.txt"
for run in "$BASE"/results_v2/C7_E/set*_seed*; do
  name=$(basename "$run"); set=${name%%_seed*}
  case $set in
    set1) NTW=712 ;; set2) NTW=695 ;; set3) NTW=702 ;; set4) NTW=738 ;;
    *) echo "unknown set $set"; exit 1 ;;
  esac
  for mode in island_champions diverse all_elites topn; do
    case $mode in
      diverse) extra="--top-n 8 --diverse-gate 3" ;;
      topn)    extra="--top-n 16" ;;
      *)       extra="" ;;
    esac
    d="$OUT/$mode/$name"
    mkdir -p "$d"
    echo "nice -n 19 python3 $SCRIPTS/score_ensemble.py --run-dir $run \
--sidecar-dir $BASE/panels_core7/${set}_core7 --step 5 --length 40 \
--num-training-windows $NTW --validation-sets 5 --param RET_CS \
--score-from 2020-01-01 --top-k 10 --book sleeves --hold-days 10 \
--ensemble $mode $extra --combine rank_mean --out-dir $d --emit-json \
> $d/stdout.log 2>&1" >> "$OUT/commands.txt"
  done
done
wc -l < "$OUT/commands.txt"
xargs -P 4 -I CMD sh -c 'CMD' < "$OUT/commands.txt"
echo PROBE1_DONE
grep -L "wrote" "$OUT"/*/*/stdout.log 2>/dev/null | head

#!/bin/bash
# HP screen submit lines (see HP_SCREEN.md for the pre-registration).
# Every cell: SPAN=tune16 (2016-2019 clock), panels set1-set2 via --array=1-2,
# seeds 42-44. Control C0 pins NTS=2000 (the registered primary; the script
# default is 200). ARM names the results_v2 output directory.
#
# Usage on Anvil:   bash hp_screen_submit.sh            # print all sbatch lines
#                   bash hp_screen_submit.sh | bash     # submit everything
set -u
SB="sbatch --array=1-2 factorial_v2.sbatch"
COMMON="SPAN=tune16,PANELS=core7,EXTRA_FLAGS=--write_elite_predictions"

emit() {  # emit <arm> <extra export assignments...>
  local arm="$1"; shift
  local extra="" nts_set=0
  for kv in "$@"; do
    extra+=",$kv"
    [[ "$kv" == NTS=* ]] && nts_set=1
  done
  # registered primary uses NTS=2000 (the launcher default is 200)
  [ "$nts_set" -eq 0 ] && extra+=",NTS=2000"
  for seed in 42 43 44; do
    echo "$SB --export=ALL,$COMMON,ARM=hp_${arm},SEED=${seed}${extra}"
  done
}

# ---- control ----------------------------------------------------------------
emit C0

# ---- Wave 1: fitness signal and target --------------------------------------
emit W1_S1_icgated  SELECT=ic_gated
emit W1_S2_ic       SELECT=ic
emit W1_T1_cs5      TARGET=RET_CS5
emit W1_TS_cs5_icg  TARGET=RET_CS5 SELECT=ic_gated

# ---- Wave 2: published ONE-NAS configuration values --------------------------
emit W2_R1_e5g10    ELITE=5 GENPOP=10
emit W2_R2_e5g15    ELITE=5 GENPOP=15
emit W2_F1_repop25  REPOP=25
emit W2_F2_repop100 REPOP=100
emit W2_N1_nts600   NTS=600
emit W2_G1_rounds2  ROUNDS=2

# ---- Wave 3: replay and training budget --------------------------------------
emit W3_P1_pa04     PER_ALPHA=0.4
emit W3_P2_pa08     PER_ALPHA=0.8
emit W3_P3_pl021    PER_LAMBDA=0.021
emit W3_B1_bp20     BP=20

#!/bin/bash -l
# build_seed.sh -- compile make_seed_genome OUT OF the ONENAS source tree.
#
# Deliberately does NOT touch ~/ONENAS/CMakeLists.txt or add anything under
# ~/ONENAS/: other agents rebuild that tree and a stray target would collide.
# We just link against the static libraries CMake already produced there.
set -euo pipefail
module load gcc/11.2.0 openmpi/4.0.6 libtiff/4.1.0

SRC=$HOME/ONENAS
BUILD=$SRC/build
HERE=$HOME/wwtp_scripts/v3/control
OUTBIN=$HERE/make_seed_genome

for lib in onenas/libonenas_strategy.a onenas/libexamm_strategy.a rnn/libexamm_nn.a \
           time_series/libexact_time_series.a time_series/libonline_series.a \
           weights/libexact_weights.a common/libexact_common.a; do
  [ -f "$BUILD/$lib" ] || { echo "missing $BUILD/$lib -- build ONENAS first"; exit 1; }
done

mpicxx -std=c++20 -O2 -o "$OUTBIN" "$HERE/make_seed_genome.cxx" \
  -I"$SRC" \
  -Wl,--start-group \
    "$BUILD/onenas/libonenas_strategy.a" \
    "$BUILD/onenas/libexamm_strategy.a" \
    "$BUILD/rnn/libexamm_nn.a" \
    "$BUILD/time_series/libexact_time_series.a" \
    "$BUILD/time_series/libonline_series.a" \
    "$BUILD/weights/libexact_weights.a" \
    "$BUILD/common/libexact_common.a" \
  -Wl,--end-group \
  -ltiff -lpthread

echo "built $OUTBIN"

#!/usr/bin/env bash
# build.sh [extra hipcc flags...] -> build/libfmt.so, build/libfmt.s (gfx950 ISA), build/resource.txt
# Compiles csrc/kernels/fp8_mqa_topk/fp8_mqa_topk_capi.cu (extern "C" launchers for the ctypes dev harness) with the
# v35wes module flags. CPU only; runs inside the minh-pham-dev-kernel container (no GPU lock needed).
set -euo pipefail
REPO=/workspace/workspace/rocm-aiter
SRC=$REPO/csrc/kernels/fp8_mqa_topk/fp8_mqa_topk_capi.cu
OUT=$REPO/op_tests/fp8_mqa_topk/build
FLAGS="-O3 --offload-arch=gfx950 -std=c++17 -fno-honor-nans -fno-slp-vectorize -mllvm -disable-machine-sink -fPIC -shared"
docker exec minh-pham-dev-kernel bash -c "
  set -e; mkdir -p $OUT/tmp && cd $OUT/tmp
  hipcc $FLAGS -save-temps -Rpass-analysis=kernel-resource-usage -Wno-unused-result $* $SRC -o $OUT/libfmt.so 2> $OUT/resource_raw.txt || { cat $OUT/resource_raw.txt | grep -v 'remark' | head -60; exit 1; }
  cp \$(ls *gfx950*.s | head -1) $OUT/libfmt.s
  python3 $REPO/op_tests/fp8_mqa_topk/resources.py $OUT/resource_raw.txt > $OUT/resource.txt
  chmod -R a+rwX $OUT
"
HOSTOUT=/home/user01/minhpham/workspace/mv-5042/workspace/rocm-aiter/op_tests/fp8_mqa_topk/build
echo "built $(date +%T): $(wc -l < $HOSTOUT/resource.txt) kernel resource lines"
cat $HOSTOUT/resource.txt

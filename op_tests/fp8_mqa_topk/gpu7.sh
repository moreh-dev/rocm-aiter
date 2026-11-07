#!/usr/bin/env bash
# gpu7.sh [-t SECS] [-C DIR] -- CMD...   run CMD on MI355X amd-smi GPU 7 (HIP_VISIBLE_DEVICES=4) inside the
# minh-pham-dev-kernel container.
#  * GPU jobs are serialized with flock (lock file inside the container), so concurrent callers queue up;
#  * after taking the lock, refuses to run if another process holds VRAM on GPU 7 (e.g. the user's benchmark);
#  * the job is killed INSIDE the container after SECS (default 900): a host-side kill of `docker exec`
#    would leave it running on the GPU.
# DIR is a container path (host /home/user01/minhpham/workspace/mv-5042 == container /workspace).
set -uo pipefail
T=900; C=/workspace/workspace/rocm-aiter/op_tests/fp8_mqa_topk
while [[ $# -gt 0 ]]; do
  case "$1" in
    -t) T=$2; shift 2 ;;
    -C) C=$2; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
[[ $# -gt 0 ]] || { echo "usage: gpu7.sh [-t SECS] [-C DIR] -- CMD..." >&2; exit 2; }
CMD=$(printf '%q ' "$@")
exec docker exec -w "$C" minh-pham-dev-kernel bash -c "
  exec 9>/tmp/fp8_mqa_topk_gpu7.lock
  flock -w 7200 9 || { echo 'gpu7.sh: lock wait timeout' >&2; exit 99; }
  used=\$(amd-smi metric -g 7 --mem-usage --json 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); d=d[\"gpu_data\"][0] if isinstance(d,dict) else d[0]; print(d[\"mem_usage\"][\"used_vram\"][\"value\"])' 2>/dev/null || echo 0)
  if [ \"\${used:-0}\" -gt 2048 ]; then echo \"gpu7.sh: GPU 7 busy (\${used} MB VRAM held by another process) - refusing\" >&2; exit 98; fi
  export HIP_VISIBLE_DEVICES=4
  timeout -s KILL $T $CMD
"

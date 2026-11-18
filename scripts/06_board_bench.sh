#!/usr/bin/env bash
# ⑥ 板端性能：单实例（单核 / 三核）+ 2 路 + 3 路并行 + 8000 帧长稳
#    SSHPASS='****' bash scripts/06_board_bench.sh
set -e
source "$(dirname "$0")/ssh_lib.sh"

RK="build/$(basename "$RKNN_FP16")"
VID="videos/man.mp4"
BBOX="80 30 145 190"

run1() {  # tag core
  bssh "cd $BOARD_DIR && $BOARD_PY scripts/bench.py --rknn $RK --video $VID \
    --bbox $BBOX --frames 300 --core $2 --out results/bench_$1.json --tag $1" \
    | grep -E '"fps"|"p50"' | head -8
}

echo "===== 1 instance / core0 ====="
run1 1x_core0 0
echo "===== 1 instance / 3 cores ====="
run1 1x_core012 012

for n in 2 3; do
  echo "===== ${n} instances（各绑一核）====="
  CMD="cd $BOARD_DIR && "
  for c in $(seq 0 $((n - 1))); do
    CMD="$CMD ( $BOARD_PY scripts/bench.py --rknn $RK --video $VID --bbox $BBOX \
      --frames 300 --core $c --out results/bench_${n}x_core$c.json --tag ${n}x_core$c \
      > results/bench_${n}x_core$c.log 2>&1 & ); "
  done
  bssh "$CMD sleep $((n * 25))"
  bssh "cd $BOARD_DIR && $BOARD_PY -c \"
import glob, json
t = 0.0
for p in sorted(glob.glob('results/bench_${n}x_core*.json')):
    d = json.load(open(p)); t += d['fps']
    print('%-10s fps=%6.2f npu=%6.2f ms' % (d['tag'], d['fps'], d['npu_ms']['mean']))
print('AGGREGATE fps = %.2f' % t)\""
done

echo "===== soak: 8000 帧单核 ====="
bssh "cd $BOARD_DIR && echo temp_before=\$(cat /sys/class/thermal/thermal_zone0/temp) \
  && free -m | head -2 && $BOARD_PY scripts/bench.py --rknn $RK --video $VID --bbox $BBOX \
  --frames 8000 --core 0 --out results/bench_soak.json --tag soak 2>&1 | tail -14 \
  && echo temp_after=\$(cat /sys/class/thermal/thermal_zone0/temp) && free -m | head -2"

for f in bench_1x_core0 bench_1x_core012 bench_2x_core0 bench_2x_core1 \
         bench_3x_core0 bench_3x_core1 bench_3x_core2 bench_soak; do
  sshpass -e scp $SSHOPT "$BOARD:$BOARD_DIR/results/$f.json" "$WORK/results/" 2>/dev/null || true
done
echo "结果已拉回 $WORK/results/"

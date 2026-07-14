#!/bin/bash
# ids_arm_watch.sh -- Periodic non-intrusive diagnostics while ARM orchestrator is running.
#
# Run this in a second nf6 terminal while `ids_arm_start` is up in the first.
# Prints every 3 seconds:
#   - FIFO status (pkt_ready, empty, head, tail, pkt_word_cnt, mode)
#   - CPU status (running, halted, thread_id)
#   - Last 8 LA entries (recent orchestrator activity)
#   - GPU DMEM[126..127] (result buffer -- updates when a kernel fires)
#
# IMPORTANT: Does NOT read FIFO BRAM. `fifo_read` does `fifo_set_mode(1)`
# internally, and mode != 0 blocks NF2 ingress (convertible_fifo.v:72:
# `in_rdy = (mode == 2'd0) && ...`). Calling it in a diagnostic loop
# would continuously drop incoming packets. GPU DMEM and the LA are on
# separate BRAMs and their PCI reads do NOT mode-flip.
#
# To inspect FIFO BRAM contents safely, FIRST stop ARM (Ctrl-C the
# ids_arm_start terminal), THEN run `python lab9reg.py fifo_read 0 9`.
#
# Usage:
#   bash ids_arm_watch.sh              # run until Ctrl-C
#   INTERVAL=5 bash ids_arm_watch.sh   # custom interval
#   LOOPS=1 bash ids_arm_watch.sh      # single snapshot then exit

INTERVAL="${INTERVAL:-3}"
LOOPS="${LOOPS:-}"

iter=0
while true; do
    iter=$((iter + 1))
    echo ""
    echo "============================================================"
    echo "[$(date +%T)] iteration $iter"
    echo "============================================================"

    echo "-- FIFO status (safe, no mode flip) --"
    python lab9reg.py status

    echo "-- CPU status (safe) --"
    python lab9reg.py cpu_status

    echo "-- LA (last 8 events; separate BRAM, safe) --"
    python lab9reg.py cpu_la_read 0 8

    echo "-- GPU DMEM[126..127] (result buffer; separate BRAM, safe) --"
    python lab9reg.py gpu_read_dmem 126 2

    if [ -n "$LOOPS" ] && [ "$iter" -ge "$LOOPS" ]; then
        exit 0
    fi

    sleep "$INTERVAL"
done

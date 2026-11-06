#!/usr/bin/env bash

set -u

cd "$HOME/HI-Diff"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BATCH_SIZES=(8 4 2)

mkdir -p benchmark_logs

SUCCESS_BATCH=""

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    CONFIG="options/train/GoPro_Student_Lite_S1_p384_b${BATCH_SIZE}_bench.yml"
    LOG="benchmark_logs/student_lite_s1_p384_b${BATCH_SIZE}.log"

    echo
    echo "=================================================="
    echo "Testing patch 384 x 384, batch size ${BATCH_SIZE}"
    echo "Config: ${CONFIG}"
    echo "Log: ${LOG}"
    echo "=================================================="

    python train.py \
      -opt "${CONFIG}" \
      --launcher none \
      2>&1 | tee "${LOG}"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ "${EXIT_CODE}" -eq 0 ] && \
       grep -q "End of training" "${LOG}" && \
       ! grep -Eqi \
       "out of memory|cuda error|traceback|nan|inf" \
       "${LOG}"; then

        echo
        echo "[SUCCESS] batch size ${BATCH_SIZE}"
        SUCCESS_BATCH="${BATCH_SIZE}"
        break
    else
        echo
        echo "[FAILED] batch size ${BATCH_SIZE}"

        python - <<'PY'
import gc
import torch

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("CUDA cache cleared.")
PY

        sleep 5
    fi
done

echo
echo "=================================================="

if [ -n "${SUCCESS_BATCH}" ]; then
    echo "Maximum tested stable batch size: ${SUCCESS_BATCH}"
    echo "${SUCCESS_BATCH}" \
      > benchmark_logs/max_stable_batch.txt
else
    echo "No tested batch size completed successfully."
    echo "Try batch size 1."
    echo "1" \
      > benchmark_logs/max_stable_batch.txt
fi

echo "=================================================="

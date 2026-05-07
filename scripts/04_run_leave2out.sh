#!/bin/bash
# Leave-2-out cross-validation across all C(9,2)=36 KO pairs.
#
# Usage:
#     bash 04_run_leave2out.sh          # CVAE (ATAC), default
#     bash 04_run_leave2out.sh atac     # CVAE (ATAC), explicit
#     bash 04_run_leave2out.sh rna      # CVAE (RNA)
#
# Each training run is ~10 sec on CPU; total ~6 minutes for 36 pairs.

set -e

INPUT=${1:-atac}

PY=/insomnia001/depts/houlab/users/lc3716/envs/epifoundation/bin/python
TRAIN=/insomnia001/depts/houlab/users/lc3716/HIGH_DIM_final/scripts/03_train.py

KOS=(ACTL6A DMAP1 EP400 EZH2 SMARCA4 SMARCB1 SMARCE1 SUZ12 YY1)
NK=${#KOS[@]}

echo "====== leave-2-out CV: input=$INPUT  $((NK*(NK-1)/2)) pairs ======"
COUNT=0
for ((i=0; i<NK-1; i++)); do
    for ((j=i+1; j<NK; j++)); do
        COUNT=$((COUNT+1))
        A=${KOS[i]}; B=${KOS[j]}
        echo ""
        echo "------ [$COUNT/$((NK*(NK-1)/2))] heldout=$A+$B  input=$INPUT ------"
        "$PY" "$TRAIN" --input "$INPUT" --holdout "$A" "$B"
    done
done

echo ""
echo "====== done ======"

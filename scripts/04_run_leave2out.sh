#!/bin/bash
# Leave-2-out cross-validation across all C(9,2)=36 KO pairs.
# Final model = only_fix1_CD: residual targeting + Fix C (z-score target)
#                              + Fix D (variance-weighted MSE).
#
# Usage:
#     bash 04_run_leave2out.sh                                # default: only_fix1_CD + genept + atac
#     bash 04_run_leave2out.sh only_fix1_CD genept atac       # explicit ATAC input
#     bash 04_run_leave2out.sh only_fix1_CD genept rna        # RNA-input variant (cell's own RNA_PCA)
#
# Each training is ~10 sec on CPU, total ~6 minutes for the 36 pairs.

set -e

VARIANT=${1:-only_fix1_CD}
GENE_EMB=${2:-genept}
INPUT=${3:-atac}

PY=/insomnia001/depts/houlab/users/lc3716/envs/epifoundation/bin/python
TRAIN=/insomnia001/depts/houlab/users/lc3716/HIGH_DIM_final/scripts/03_train.py

KOS=(ACTL6A DMAP1 EP400 EZH2 SMARCA4 SMARCB1 SMARCE1 SUZ12 YY1)
NK=${#KOS[@]}

echo "====== leave-2-out CV: variant=$VARIANT  gene_emb=$GENE_EMB  input=$INPUT  $((NK*(NK-1)/2)) pairs ======"
COUNT=0
for ((i=0; i<NK-1; i++)); do
    for ((j=i+1; j<NK; j++)); do
        COUNT=$((COUNT+1))
        A=${KOS[i]}; B=${KOS[j]}
        echo ""
        echo "------ [$COUNT/$((NK*(NK-1)/2))] heldout=$A+$B  variant=$VARIANT  gene_emb=$GENE_EMB  input=$INPUT ------"
        "$PY" "$TRAIN" --variant "$VARIANT" --gene_emb "$GENE_EMB" --input "$INPUT" --holdout "$A" "$B"
    done
done

echo ""
echo "====== done ======"
GENE_SUFFIX=$([ "$GENE_EMB" = "genept" ] && echo "" || echo "_${GENE_EMB}")
INPUT_SUFFIX=$([ "$INPUT" = "atac" ] && echo "" || echo "_${INPUT}")
SUFFIX="${GENE_SUFFIX}${INPUT_SUFFIX}"
ls /insomnia001/depts/houlab/users/lc3716/HIGH_DIM_final/models/cvae_${VARIANT}_loko_*${SUFFIX}.pt 2>/dev/null | wc -l
echo "checkpoints written for variant=$VARIANT, gene_emb=$GENE_EMB, input=$INPUT"

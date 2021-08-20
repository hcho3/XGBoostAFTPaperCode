#!/bin/bash

set -e
set -u
shopt -s failglob
set -o pipefail

for dataset in ATAC_JV_adipose CTCF_TDH_ENCODE H3K27ac-H3K4me3_TDHAM_BP H3K27ac_TDH_some \
               H3K36me3_AM_immune H3K27me3_RL_cancer H3K27me3_TDH_some H3K36me3_TDH_ENCODE \
               H3K36me3_TDH_immune H3K36me3_TDH_other
do
    mkdir -p survreg-log/${dataset}
    echo "Rscript run_survreg.R chipseq ${dataset}"
    Rscript run_survreg.R chipseq ${dataset}
done

for dataset in simulated.sin simulated.abs simulated.linear simulated.model.1 simulated.model.2 simulated.model.3
do
    mkdir -p survreg-log/${dataset}
    echo "Rscript run_survreg.R simulated ${dataset}"
    Rscript run_survreg.R simulated ${dataset}
done

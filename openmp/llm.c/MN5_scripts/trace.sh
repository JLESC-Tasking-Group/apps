#!/bin/bash
source /apps/GPP/BSCTOOLS/extrae/latest/impi_2021_10_0/etc/extrae.sh
export EXTRAE_CONFIG_FILE=${GPT_HOME}/outputs/tmp/${SLURM_JOBID}/extrae.xml
export LD_PRELOAD=${EXTRAE_HOME}/lib/libomptrace.so
echo $LD_PRELOAD

$*

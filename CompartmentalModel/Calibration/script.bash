#!/bin/bash
##---------------SLURM Parameters - NLHPC ----------------
#SBATCH -J sir_test
##SBATCH -p general
#SBATCH -n 10
#SBATCH --ntasks-per-node=10
#SBATCH -c 1
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-user=pauribe@dim.uchile.cl
#SBATCH --mail-type=ALL
#SBATCH -o sir_test_%j.out
#SBATCH -e sir_test_%j.err

#-----------------Toolchain---------------------------
# ----------------Modules----------------------------
ml  Python/3.7.2
# ----------------Command--------------------------

## PRIMERA CAPA
#***************
#python oneLayer_CI.py $SLURM_JOBID oneLayer_diciembre_3.stan 16747472.4108264 diciembre 5646.77078914297 15867.0784805259 1159165.9187173 

## PRIMERA Y SEGUNDA CAPA
#************************

## ULTIMO

# MARZO
## CASO G
#python twoLayers_3.py $SLURM_JOBID twoLayers_marzo_ultimo.stan marzo 0.23532938 0.120576811 0.002293053 /home/puribe/covid/ultimos/output/oneLayer/23926145/tasas_means.csv

## CASO H
#python twoLayers_3.py $SLURM_JOBID twoLayers_marzo_ultimo.stan marzo 0.1666046264 0.08957179406 0.002056422339 /home/puribe/covid/ultimos/output/oneLayer/23922175/tasas_means.csv

# ABRIL EN ADELANTE
## CASO G
python twoLayers_abril_ultimo.py $SLURM_JOBID twoLayers_abril_ultimo_v2.stan julio 0.23532938 0.120576811 0.002293053 /home/puribe/covid/ultimos/datos_2.csv

## CASO H
#python twoLayers_abril_ultimo.py $SLURM_JOBID twoLayers_abril_ultimo_v2.stan abril 0.166604626 0.089571794 0.002056422 /home/puribe/covid/ultimos/output/twoLayers/23926977/tasas_means.csv

#python twoLayers_3_abril_ultimo.py $SLURM_JOBID twoLayers_abril_ultimo.stan abril /home/puribe/covid/ultimos/output/twoLayers/23925391/tasas_means.csv 0.178124433 0.088450329 0.002041426

#python twoLayers_abril_ultimo.py $SLURM_JOBID twoLayers_abril_ultimo_v2.stan abril 0.166604626 0.089571794 0.002056422 /home/puribe/covid/ultimos/output/twoLayers/23923480/tasas_means.csv


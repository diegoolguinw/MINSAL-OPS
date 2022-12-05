#!/bin/bash
##---------------SLURM Parameters - NLHPC ----------------
#SBATCH -J sir_test
##SBATCH -p debug
#SBATCH -n 10
#SBATCH --ntasks-per-node=10
#SBATCH -c 1
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-user=pauribe@dim.uchile.cl
#SBATCH --mail-type=ALL
#SBATCH -o sir_test_%j.out
#SBATCH -e sir_test_%j.err

#-----------------Toolchain---------------------------
# ----------------Modules----------------------------
ml  Python/3.7.2
# ----------------Command--------------------------


# modelo 3 capas
# parámetros de entrada
# 1. job id (desde el sistema)
# 2. modelo stan a utilizar
# 3. mes
# 4. gamma_aN
# 5. gamma_sN
# 6. deltaN
# 7. gamma_aV
# 8. gamma_sV
# 9. deltaV
# 10. input file

# tramo 1
#python threeLayers.py $SLURM_JOBID threeLayers.stan agosto-dic 0.23532938 0.120576811 0.002293053 /home/puribe/covid/ultimos/output/twoLayers/23928217/tasas_means.csv

#python threeLayers.py $SLURM_JOBID threeLayers.stan agosto-sept 0.23532938 0.120576811 0.002293053 /home/puribe/covid/ultimos/output/twoLayers/23928217/tasas_means.csv

#python threeLayers.py $SLURM_JOBID threeLayers.stan agosto-15oct 0.23532938 0.120576811 0.002293053 /home/puribe/covid/ultimos/output/twoLayers/23928217/tasas_means.csv

#python threeLayers_ultimo.py $SLURM_JOBID threeLayers_CI_0_v2.stan agosto-sept 0.23532938 0.120576811 0.002293053 /home/puribe/covid/ultimos/output/twoLayers/23928217/tasas_means.csv

## SEGUNDO BLOQUE
python threeLayers_ultimo_bloque2.py $SLURM_JOBID threeLayers_CI_0_v2_bloque2.stan octubre-dic 0.23532938 0.120576811 0.002293053 /home/puribe/covid/ultimos/output/threeLayers/24422610/tasas_means.csv

#python threeLayers_ultimo_bloque2.py $SLURM_JOBID threeLayers_CI_0_v2_bloque2.stan 15oct-dic 0.23532938 0.120576811 0.002293053 /home/puribe/covid/ultimos/output/threeLayers/24422607/tasas_means.csv


#python threeLayers_ultimo.py $SLURM_JOBID threeLayers_CI_0_v2.stan agosto-oct 0.23532938 0.120576811 0.002293053 /home/puribe/covid/ultimos/output/twoLayers/23928217/tasas_means.csv

#python threeLayers.py $SLURM_JOBID threeLayers_CI_0.stan agosto-15oct 0.23532938 0.120576811 0.002293053 /home/puribe/covid/ultimos/output/twoLayers/23928217/tasas_means.csv


# tramo 2
#python threeLayers.py $SLURM_JOBID threeLayers.stan octubre 0.23532938 0.120576811 0.002293053 /home/puribe/covid/ultimos/output/threeLayers/24248738/tasas_means.csv

#python threeLayers.py $SLURM_JOBID threeLayers.stan diciembre 0.23532938 0.120576811 0.002293053 /home/puribe/covid/ultimos/output/threeLayers/tasas_means.csv



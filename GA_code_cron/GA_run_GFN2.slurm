#!/bin/bash
#SBATCH --output=test.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-05:00:00

# allows cron to run bash
source /ihome/crc/install/lmod/lmod/init/bash

# Load modules
module purge
module load python/anaconda3.5-4.2.0

molname=${SLURM_JOB_NAME}

# perform forcefield geometry optimization with pybel
python /ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/GA_cron/forcefield.py ${molname}

export XTBHOME=/ihome/ghutchison/oda6/xtb/xtb-641/bin/xtb

# Copy files to SLURM_SCRATCH
cp $SLURM_JOB_NAME $SLURM_SCRATCH

# cd to the SCRATCH space
cd $SLURM_SCRATCH

# run GFN2-xTB geometry optimization
${XTBHOME} /ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/input_mol/${SLURM_JOB_NAME}.mol -opt > /ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/output_GFN2/${molname}.out

gzip /ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/output_GFN2/${molname}.out


# copy incremental progress
cp xtbopt.mol /ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/opt_mol/${molname}.mol
rm -f *restart 

touch /ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/generations_${run_name}_{run_label}/${gen_counter}/${molname}.txt 
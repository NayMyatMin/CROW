#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ## 
#################################################

#SBATCH --nodes=1                   # Use 1 node
#SBATCH --cpus-per-task=10          # Increase to 10 CPUs for faster processing
#SBATCH --mem=80GB                  # Increase memory to 24GB
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --constraint=a100           # Target A100 GPUs specifically
#SBATCH --time=02-00:00:00          # Maximum run time of 2 days
##SBATCH --mail-type=BEGIN,END,FAIL  # Email notifications for job start, end, and failure
#SBATCH --output=%u.eval         # Log file location

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=researchshort                 # Partition assigned
#SBATCH --account=sunjunresearch   # Account assigned (use myinfo command to check)
#SBATCH --qos=research-1-qos         # QOS assigned (use myinfo command to check)
#SBATCH --job-name=crow_llama2               # Job name (set for running LLaMA-2-7b)
#SBATCH --mail-user=dummy@phdcs.smu.edu.sg  # Email notifications

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Purge the environment, load the modules we require.
# Refer to https://violet.smu.edu.sg/origami/module/ for more information
module purge
module load Python/3.7.13
module load CUDA/11.7.0

# Create a virtual environment
# python3 -m venv ~/myenv

# This command assumes that you've already created the environment previously
# We're using an absolute path here. You may use a relative path, as long as SRUN is execute in the same working directory
source ~/myenv/bin/activate

# srun --gres=gpu:1 python3 backdoor_evaluate_negsenti_badnet.py
# srun --gres=gpu:1 python3 backdoor_evaluate_negsenti_ctba.py
# srun --gres=gpu:1 python3 backdoor_evaluate_negsenti_mtba.py
# srun --gres=gpu:1 python3 backdoor_evaluate_negsenti_sleeper.py
# srun --gres=gpu:1 python3 backdoor_evaluate_negsenti_vpi.py

# srun --gres=gpu:1 python3 backdoor_evaluate_refusal_badnet.py
# srun --gres=gpu:1 python3 backdoor_evaluate_refusal_ctba.py
# srun --gres=gpu:1 python3 backdoor_evaluate_refusal_mtba.py
# srun --gres=gpu:1 python3 backdoor_evaluate_refusal_sleeper.py
# srun --gres=gpu:1 python3 backdoor_evaluate_refusal_vpi.py

srun --gres=gpu:1 python3 backdoor_evaluate_codeinjection_vpi-ci.py

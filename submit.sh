#!/bin/bash
NAME="$1"

sbatch <<EOF
#!/bin/bash
#SBATCH -J ${NAME}
#SBATCH -p gpu
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --constraint=h100
#SBATCH --mem=100G
#SBATCH --gpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH -o output/slurm_logs/${NAME}.log
#SBATCH --mail-type=BEGIN  # Send an email when the job starts
#SBATCH --mail-user=rgower@flatironinstitute.org  # Your email address

export OMP_NUM_THREADS=1

module load modules/2.4-alpha2
source nano11/bin/activate
module list 


# Run the Python script with the config file
time torchrun --standalone --nproc_per_node=4 train_gpt.py  --name $NAME
EOF

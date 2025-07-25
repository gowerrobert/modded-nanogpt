#!/bin/bash
CONFIG_NAME=$(basename "$1" .yaml)
NUM_GPUS="$2"

sbatch <<EOF
#!/bin/bash
#SBATCH -J ${CONFIG_NAME}
#SBATCH -p gpu
#SBATCH --ntasks=${NUM_GPUS}
#SBATCH --nodes=1
#SBATCH --constraint=h100
#SBATCH --mem=200G
#SBATCH --gpus-per-task=1
#SBATCH --time=100:00:00
#SBATCH -o slurm_logs/${CONFIG_NAME}.log
#SBATCH --mail-type=BEGIN  # Send an email when the job starts
#SBATCH --mail-user=rgower@flatironinstitute.org  # Your email address

export OMP_NUM_THREADS=1

module load modules/2.4-alpha2
source nano11/bin/activate
module list 

# Run the Python script with the config file
time torchrun --standalone --nproc_per_node=${NUM_GPUS} run.py --config $1
EOF


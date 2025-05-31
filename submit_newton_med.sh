#!/bin/bash
NAME="$1"
NUM_GPUS="$2"

sbatch <<EOF
#!/bin/bash
#SBATCH -J med-Newton5-${NAME}
#SBATCH -p gpu
#SBATCH --ntasks=${NUM_GPUS}
#SBATCH --nodes=1
#SBATCH --constraint=h100
#SBATCH --mem=200G
#SBATCH --gpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH -o output/slurm_logs/med-Newton5-${NAME}.log
#SBATCH --mail-type=BEGIN  # Send an email when the job starts
#SBATCH --mail-user=rgower@flatironinstitute.org  # Your email address

export OMP_NUM_THREADS=1

module load modules/2.4-alpha2
source nano11/bin/activate
module list 

# Run the Python script with the config file
time torchrun --standalone --nproc_per_node=${NUM_GPUS} train_gpt_medium.py --mat_sign newton --name med-Newton5-${NAME}  --num_iterations 10
EOF

#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=0-01:00
#SBATCH --job-name=QR_A10
#SBATCH --mail-user=patrickgc.deng@mail.utoronto.ca
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-plavoie
source /home/p/plavoie/denggua1/.virtualenvs/pdenv/bin/activate
# Change this to wherever your “compute_velocity_gradient_core” package lives:
export PYTHONPATH="/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/PostProc/Compute_Velocity_Gradient"

# Parent directory where your Cut_<cut>_VGT folders live:
#PARENT_DIR="/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_5AOA_LES/Isosurface"
PARENT_DIR="/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_10AOA_LES/Isosurface"
# Number of processes and blocks you want:
NPROC=10
NBLOCKS=1200

# List your cuts here:
CUTS=( "095_TE")

for CUT in "${CUTS[@]}"; do
  echo "Processing cut: $CUT"
  python -m compute_velocity_gradient_core \
    --cut       "$CUT" \
    --parent-dir "$PARENT_DIR" \
    --nproc     "$NPROC" \
    --nblocks   "$NBLOCKS" \
    --output-dir "./" \
    --velocity 30 \
    --data-type "LES"\
    --angle-of-attack 10\
    --reload
  echo
done

echo "All cuts complete."
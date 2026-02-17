#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=0-13:59
#SBATCH --job-name=QR_A10U50
#SBATCH --mail-user=patrickgc.deng@mail.utoronto.ca
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-plavoie
source /project/rrg-plavoie/denggua1/pd_env.sh
# Change this to wherever your “compute_velocity_gradient_core” package lives:
export PYTHONPATH="/scratch/denggua1/Bombardier_LES/B_10AOA_U50_LES/PostProc_Fine/Compute_Velocity_Gradient"

# Parent directory where your Cut_<cut>_VGT folders live:
#PARENT_DIR="/home/p/plavoie/denggua1/scratch/Bombardier_LES/B_5AOA_LES/Isosurface"
PARENT_DIR="/scratch/denggua1/Bombardier_LES/B_10AOA_U50_LES/Isosurface/Extract_Cutplane_Fine"
# Number of processes and blocks you want:
NPROC=40
NBLOCKS=1200

# List your cuts here:
CUTS=("030_TE" "PIV1" "PIV2" "085_TE" "095_TE" "PIV3")

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
#!/bin/bash
#SBATCH --job-name=pda_rad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Run the RadImageNet PDA experiments (5 pooling methods x training strategy 1 or 3).
#   POOL     in avg | max | attn | tanh | tran
#   STRATEGY 1 = fine-tune the whole network, 3 = frozen RadImageNet encoder
# single run:  sbatch --export=ALL,POOL=attn,STRATEGY=3 launch_radimagenet.sh
# all 10 runs: bash launch_radimagenet.sh submit_all
# Edit the three paths below for your environment.

CSV=/path/to/pda_train_val_test.csv                 # exam/frame catalog CSV
OUT=/path/to/outputs/pda              # results go to $OUT/rad_<POOL>_<STRATEGY>/
WEIGHTS=pretrained/radimagenet/ResNet50.pt  # RadImageNet ResNet50.pt (Lab-Rasool/RadImageNet on Hugging Face)

if [ "$1" == "submit_all" ]; then
    mkdir -p $OUT
    script=$(realpath "$0")
    for pool in avg max attn tanh tran; do
        for strategy in 1 3; do
            sbatch -D $OUT -J pda_rad_${pool}_${strategy} --export=ALL,POOL=$pool,STRATEGY=$strategy "$script"
        done
    done
    exit 0
fi

case $POOL in
    tanh) pooling=tanh_attn ;;
    tran) pooling=transformer ;;
    *)    pooling=$POOL ;;
esac

cd "$(dirname "$(realpath "$0")")"

plotdir="${OUT}/rad_${POOL}_${STRATEGY}/"
artifactfolder="${plotdir}model_run_artifacts/"
mkdir -p $artifactfolder

freeze_flag=""
if [ "$STRATEGY" == "3" ]; then freeze_flag="--freeze-encoder"; fi

python train_cv_radimagenet.py \
    config_radimagenet.yaml \
    --artifact-folder ${artifactfolder} \
    --plot-dir ${plotdir} \
    --frame-csv "$CSV" \
    --radimagenet-weights "$WEIGHTS" \
    --device 'cuda:0' \
    --pooling-method ${pooling} \
    --epochs 12 \
    ${freeze_flag} \
    2>&1 | tee ${artifactfolder}log.txt

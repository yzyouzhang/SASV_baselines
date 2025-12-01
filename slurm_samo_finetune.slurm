#!/bin/bash
#SBATCH --job-name=samo_ft
#SBATCH --account=bfdc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=8:00:00
#SBATCH --output=slurm_outputs/slurm-%j.out
#SBATCH --error=slurm_outputs/slurm-%j.err

# SAMO Fine-tuning Script
# This script fine-tunes a pretrained model using SAMO loss with:
# 1. Speaker attractors built from training data
# 2. Frozen SAMO centers (only embedding network updates)
# 3. Lower learning rate for fine-tuning

# Use your personal miniconda3 instead of module
source /u/yzyouzhang/miniconda3/etc/profile.d/conda.sh
conda activate wildspoof_sasv

# Configuration
INITIAL_MODEL="exp/sasv_redimnet/model/model000000008.model"  # Your best pretrained model
SAVE_PATH="exp/sasv_samo_finetune"

# Training data
TRAIN_LIST="corpus/spoofceleb/metadata/train.csv"
TRAIN_PATH="corpus/spoofceleb/flac/train"
SPK_META_TRAIN="spk_meta/spk_meta_trn_spoofceleb.pk"

# Validation data
EVAL_LIST="corpus/spoofceleb/protocol/sasv_development_evaluation_protocol.csv"
EVAL_PATH="corpus/spoofceleb/flac/development"
SPK_META_EVAL="spk_meta/spk_meta_dev_spoofceleb.pk"

# MUSAN and RIR for data augmentation (update paths if different)
MUSAN_PATH="/path/to/dataset/MUSAN/musan_split"
RIR_PATH="/path/to/dataset/RIRS_NOISES/simulated_rirs"

echo "=============================================="
echo "SAMO Fine-tuning Configuration"
echo "=============================================="
echo "Initial model: $INITIAL_MODEL"
echo "Save path: $SAVE_PATH"
echo "Training data: $TRAIN_LIST"
echo "Validation data: $EVAL_LIST"
echo "=============================================="
echo ""

# Fine-tune with SAMO loss and frozen centers
python trainSASVNet.py \
    --initial_model $INITIAL_MODEL \
    --save_path $SAVE_PATH \
    --train_list $TRAIN_LIST \
    --eval_list $EVAL_LIST \
    --train_path $TRAIN_PATH \
    --eval_path $EVAL_PATH \
    --spk_meta_train $SPK_META_TRAIN \
    --spk_meta_eval $SPK_META_EVAL \
    --musan_path $MUSAN_PATH \
    --rir_path $RIR_PATH \
    --model ReDimNet \
    --redimnet_model b2 \
    --redimnet_train_type ptn \
    --redimnet_dataset vox2 \
    --redimnet_pretrained True \
    --redimnet_repo yzyouzhang/redimnet \
    --trainfunc samo_sasv \
    --num_class 1160 \
    --num_out 192 \
    --build_train_attractors \
    --freeze_samo_centers \
    --max_frames 400 \
    --num_spk 40 \
    --num_utt 2 \
    --batch_size 40 \
    --max_epoch 20 \
    --test_interval 1 \
    --optimizer adamW \
    --scheduler cosine_annealing_warmup_restarts \
    --lr 1e-5 \
    --lr_t0 5 \
    --lr_tmul 1.0 \
    --lr_max 1e-5 \
    --lr_min 5e-6 \
    --lr_wstep 0 \
    --lr_gamma 0.9 \
    --weight_decay 1e-7 \
    --margin 0.2 \
    --scale 30 \
    --num_thread 16

echo ""
echo "=============================================="
echo "Fine-tuning completed!"
echo "=============================================="
echo ""
echo "To evaluate with SAMO scoring:"
echo "python trainSASVNet.py \\"
echo "  --eval \\"
echo "  --use_enroll \\"
echo "  --use_samo_scoring \\"
echo "  --initial_model ${SAVE_PATH}/model/best_model.model \\"
echo "  --eval_list $EVAL_LIST \\"
echo "  --eval_path $EVAL_PATH \\"
echo "  --trainfunc samo_sasv \\"
echo "  --model ReDimNet \\"
echo "  --redimnet_model b2 \\"
echo "  --num_class 1160 \\"
echo "  --save_path ${SAVE_PATH}_eval"

#!/bin/bash

########################
# SLURM options
########################
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfdc-delta-gpu
#SBATCH --job-name=spoofceleb
#SBATCH --time=6:59:59

# Per-task logs
#SBATCH -e slurm_outputs/slurm-%A_%a.err
#SBATCH -o slurm_outputs/slurm-%A_%a.out

### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none

#SBATCH --mail-user="you.zhang@rochester.edu"
##SBATCH --mail-type="ALL"

########################
# Environment setup
########################

module load anaconda
# conda init
# conda activate samo2

module list


conda init
conda activate wildspoof_sasv

# CUDA_VISIBLE_DEVICES=0 python trainSASVNet.py   --max_frames 400   --num_spk 400   --num_utt 2   --batch_size 40   --trainfunc sasv_e2e_v1   --optimizer adamW   --scheduler cosine_annealing_warmup_restarts   --lr_t0 8   --lr_tmul 1.0   --lr_max 1e-4   --lr_min 0   --lr_wstep 0   --lr_gamma 0.8   --margin 0.2   --scale 30   --num_class 1160   --save_path exp/sasv_baseline   --train_list corpus/spoofceleb/metadata/train.csv   --eval_list corpus/spoofceleb/protocol/sasv_development_evaluation_protocol.csv   --train_path corpus/spoofceleb/flac/train   --eval_path corpus/spoofceleb/flac/development   --spk_meta_train spk_meta/spk_meta_trn_spoofceleb.pk   --spk_meta_eval spk_meta/spk_meta_dev_spoofceleb.pk   --musan_path /path/to/dataset/MUSAN/musan_split   --rir_path /path/to/dataset/RIRS_NOISES/simulated_rirs   --model SKA_TDNN

# CUDA_VISIBLE_DEVICES=0 python trainSASVNet.py \
#   --max_frames 400 \
#   --num_spk 400 \
#   --num_utt 2 \
#   --batch_size 40 \
#   --trainfunc sasv_e2e_v1 \
#   --optimizer adamW \
#   --scheduler cosine_annealing_warmup_restarts \
#   --lr_t0 8 \
#   --lr_tmul 1.0 \
#   --lr_max 1e-4 \
#   --lr_min 0 \
#   --lr_wstep 0 \
#   --lr_gamma 0.8 \
#   --margin 0.2 \
#   --scale 30 \
#   --num_class 1160 \
#   --save_path exp/sasv_redimnet \
#   --train_list corpus/spoofceleb/metadata/train.csv \
#   --eval_list corpus/spoofceleb/protocol/sasv_development_evaluation_protocol.csv \
#   --train_path corpus/spoofceleb/flac/train \
#   --eval_path corpus/spoofceleb/flac/development \
#   --spk_meta_train spk_meta/spk_meta_trn_spoofceleb.pk \
#   --spk_meta_eval spk_meta/spk_meta_dev_spoofceleb.pk \
#   --musan_path /path/to/dataset/MUSAN/musan_split \
#   --rir_path /path/to/dataset/RIRS_NOISES/simulated_rirs \
#   --model ReDimNet \
#   --redimnet_model b2 \
#   --redimnet_train_type ptn \
#   --redimnet_dataset vox2 \
#   --redimnet_pretrained True \
#   --redimnet_repo yzyouzhang/redimnet


# # Evaluate on validation/development set
# CUDA_VISIBLE_DEVICES=0 python trainSASVNet.py \
#   --eval \
#   --initial_model exp/sasv_redimnet/model/model000000008.model \
#   --eval_list corpus/spoofceleb/protocol/sasv_development_evaluation_protocol.csv \
#   --eval_path corpus/spoofceleb/flac/development \
#   --spk_meta_eval spk_meta/spk_meta_dev_spoofceleb.pk \
#   --save_path exp/sasv_redimnet_eval \
#   --model ReDimNet \
#   --redimnet_model b2 \
#   --num_class 1160 \
#   --batch_size 40 \
#   --num_thread 8

# # Evaluate with SAMO scoring on validation set (debug mode)
# CUDA_VISIBLE_DEVICES=0 python trainSASVNet.py \
#   --eval \
#   --debug \
#   --initial_model exp/sasv_redimnet/model/model000000008.model \
#   --eval_list corpus/spoofceleb/protocol/sasv_development_evaluation_protocol.csv \
#   --eval_path corpus/spoofceleb/flac/development \
#   --spk_meta_eval spk_meta/spk_meta_dev_spoofceleb.pk \
#   --save_path exp/sasv_samo_eval \
#   --model ReDimNet \
#   --redimnet_model b2 \
#   --trainfunc samo_sasv \
#   --num_class 1160 \
#   --batch_size 40 \
#   --num_thread 8 \
#   --use_samo_scoring \
#   --use_enroll

# # Evaluate with SAMO scoring on validation set
# CUDA_VISIBLE_DEVICES=0 python trainSASVNet.py \
#   --eval \
#   --initial_model exp/sasv_samo_finetune2/model/model000000007.model \
#   --eval_list corpus/spoofceleb/protocol/sasv_development_evaluation_protocol.csv \
#   --eval_path corpus/spoofceleb/flac/development \
#   --spk_meta_eval spk_meta/spk_meta_dev_spoofceleb.pk \
#   --save_path exp/sasv_samo_eval \
#   --model ReDimNet \
#   --redimnet_model b2 \
#   --trainfunc samo_sasv \
#   --num_class 1160 \
#   --batch_size 40 \
#   --num_thread 8 \
#   --use_samo_scoring \
#   --use_enroll

# ========================================
# Score on WildSpoof test set
# ========================================

# Step 1: Generate scores with cosine similarity (using TSV files directly)
CUDA_VISIBLE_DEVICES=0 python trainSASVNet.py \
  --scoring \
  --initial_model exp/sasv_redimnet/model/model000000008.model \
  --enroll_tsv /work/hdd/bfdc/yzyouzhang/spoofceleb_test/protocol_enroll.tsv \
  --test_tsv /work/hdd/bfdc/yzyouzhang/spoofceleb_test/protocol_test.tsv \
  --eval_path /work/hdd/bfdc/yzyouzhang/spoofceleb_test/data_v2.0 \
  --save_path exp/sasv_redimnet_test_final \
  --model ReDimNet \
  --redimnet_model b2 \
  --num_class 1160 \
  --batch_size 1 \
  --num_thread 8

echo "=========================================="
echo "Cosine similarity scoring complete!"
echo "Scores file: exp/sasv_redimnet_test_final/result/1_scorefile"
echo "Format: speaker_id TAB utt_id TAB score"
echo "=========================================="

# Step 2: Generate scores with SAMO (using TSV files directly)
# The system will:
# 1. Build speaker-level enrollment embeddings (averaged from enrollment TSV)
# 2. Compare each test file with its speaker's enrollment embedding
# 3. Use SAMO inference if --use_samo_scoring is set
CUDA_VISIBLE_DEVICES=0 python trainSASVNet.py \
  --scoring \
  --initial_model exp/sasv_redimnet/model/model000000008.model \
  --enroll_tsv /work/hdd/bfdc/yzyouzhang/spoofceleb_test/protocol_enroll.tsv \
  --test_tsv /work/hdd/bfdc/yzyouzhang/spoofceleb_test/protocol_test.tsv \
  --eval_path /work/hdd/bfdc/yzyouzhang/spoofceleb_test/data_v2.0 \
  --save_path exp/sasv_samo_test_final2_finetune_newo \
  --model ReDimNet \
  --redimnet_model b2 \
  --trainfunc samo_sasv \
  --num_class 1160 \
  --batch_size 1 \
  --num_thread 8 \
  --use_samo_scoring 

echo "=========================================="
echo "SAMO scoring complete!"
echo "Scores file: exp/sasv_samo_test_final2_finetune/result/1_scorefile"
echo "Format: speaker_id TAB utt_id TAB score"
echo "=========================================="

# Step 4: Analyze and optionally merge the two scoring methods
echo ""
echo "To analyze and compare the two scoring methods:"
echo "python analyze_scores.py \\"
echo "  --score1 exp/sasv_redimnet_test_final/result/1_scorefile \\"
echo "  --score2 exp/sasv_samo_test_final2_finetune/result/1_scorefile \\"
echo "  --name1 'Cosine' --name2 'SAMO'"
echo ""
echo "To merge the scores with equal weights:"
echo "python analyze_scores.py \\"
echo "  --score1 exp/sasv_redimnet_test_final/result/1_scorefile \\"
echo "  --score2 exp/sasv_samo_test_final2_finetune/result/1_scorefile \\"
echo "  --merge --output merged_scores.txt \\"
echo "  --name1 'Cosine' --name2 'SAMO'"
echo "=========================================="

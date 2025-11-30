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


# Evaluate on validation/development set
CUDA_VISIBLE_DEVICES=0 python trainSASVNet.py \
  --eval \
  --scoring \
  --initial_model exp/sasv_redimnet/model/model000000008.model \
  --eval_list corpus/spoofceleb/protocol/sasv_development_evaluation_protocol.csv \
  --eval_path corpus/spoofceleb/flac/development \
  --spk_meta_eval spk_meta/spk_meta_dev_spoofceleb.pk \
  --save_path exp/sasv_redimnet_eval \
  --model ReDimNet \
  --redimnet_model b2 \
  --num_class 1160 \
  --batch_size 40 \
  --num_thread 8


# # ========================================
# # Score on WildSpoof test set
# # ========================================

# # Step 1: Prepare test protocol (convert TSV to CSV format)
# # This creates a CSV file with enrollment and test file pairs
# python prepare_test_protocol.py \
#   --enroll_file /work/hdd/bfdc/yzyouzhang/spoofceleb_test/protocol_enroll.tsv \
#   --test_file /work/hdd/bfdc/yzyouzhang/spoofceleb_test/protocol_test.tsv \
#   --output_file /work/hdd/bfdc/yzyouzhang/spoofceleb_test/protocol_test.csv \
#   --use_first_enroll

# # Step 2: Generate scores on test set
# # Note: No speaker metadata needed for scoring mode
# CUDA_VISIBLE_DEVICES=0 python trainSASVNet.py \
#   --scoring \
#   --initial_model exp/sasv_redimnet/model/model000000050.model \
#   --eval_list /work/hdd/bfdc/yzyouzhang/spoofceleb_test/protocol_test.csv \
#   --eval_path /work/hdd/bfdc/yzyouzhang/spoofceleb_test/data_v2.0 \
#   --save_path exp/sasv_redimnet_test \
#   --model ReDimNet \
#   --redimnet_model b2 \
#   --num_class 1160 \
#   --batch_size 1 \
#   --num_thread 8

# # Step 3: Format scores for submission
# # The output will be in exp/sasv_redimnet_test/result/*_scorefile
# # You may need to post-process to match submission format:
# # SPK_ID UTT_ID SCORE
# python format_submission.py \
#   --score_file exp/sasv_redimnet_test/result/1_scorefile \
#   --test_protocol /work/hdd/bfdc/yzyouzhang/spoofceleb_test/protocol_test.tsv \
#   --output_file exp/sasv_redimnet_test/submission.tsv \
#   --aggregate_enrollments

# echo "=========================================="
# echo "Scoring complete!"
# echo "Submission file: exp/sasv_redimnet_test/submission.tsv"
# echo "=========================================="

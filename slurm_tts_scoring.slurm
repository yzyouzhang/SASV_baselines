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
#SBATCH --job-name=tts_scoring
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
module list

conda init
conda activate wildspoof_sasv

########################
# Score TTS evaluation sets
########################

TTS_BASE_DIR="/work/hdd/bfdc/yzyouzhang/spoofceleb_test/TTS_SASV_eval_anony"
ENROLL_TSV="/work/hdd/bfdc/yzyouzhang/spoofceleb_test/protocol_enroll.tsv"
ENROLL_PATH="/work/hdd/bfdc/yzyouzhang/spoofceleb_test/data_v2.0"
TTS_PROTOCOL="${TTS_BASE_DIR}/protocol_tts_posteval_v2.0.tsv"

# Check if shared protocol file exists
if [ ! -f "${TTS_PROTOCOL}" ]; then
    echo "Error: Protocol file ${TTS_PROTOCOL} does not exist!"
    exit 1
fi

# Loop through all 7 TTS folders
for TTS_NUM in 1 2 3 4 5 6 7; do
    TTS_DIR="${TTS_BASE_DIR}/TTS_${TTS_NUM}"
    
    echo ""
    echo "========================================================================"
    echo "Processing TTS_${TTS_NUM}"
    echo "========================================================================"
    
    # Check if TTS directory exists
    if [ ! -d "${TTS_DIR}" ]; then
        echo "Warning: Directory ${TTS_DIR} does not exist, skipping..."
        continue
    fi
    
    # Step 1: Map TTS files to target speakers
    echo "Mapping TTS files to target speakers..."
    python map_tts_to_speakers.py \
        --tts_protocol "${TTS_PROTOCOL}" \
        --enroll_protocol "${ENROLL_TSV}" \
        --output_file "${TTS_BASE_DIR}/protocol_tts${TTS_NUM}_test.tsv"
    
    echo ""
    echo "----------------------------------------"
    echo "Scoring TTS_${TTS_NUM} with Cosine Similarity"
    echo "----------------------------------------"
    
    # Step 2a: Score with cosine similarity
    CUDA_VISIBLE_DEVICES=0 python trainSASVNet.py \
        --scoring \
        --initial_model exp/sasv_redimnet/model/model000000008.model \
        --enroll_tsv "${ENROLL_TSV}" \
        --test_tsv "${TTS_BASE_DIR}/protocol_tts${TTS_NUM}_test.tsv" \
        --eval_path "${TTS_DIR}" \
        --enroll_path "${ENROLL_PATH}" \
        --save_path "exp/sasv_redimnet_tts${TTS_NUM}" \
        --model ReDimNet \
        --redimnet_model b2 \
        --num_class 1160 \
        --batch_size 1 \
        --num_thread 8
    
    echo "Cosine similarity scores saved to: exp/sasv_redimnet_tts${TTS_NUM}/result/1_scorefile"
    
    echo ""
    echo "----------------------------------------"
    echo "Scoring TTS_${TTS_NUM} with SAMO"
    echo "----------------------------------------"
    
    # Step 2b: Score with SAMO
    CUDA_VISIBLE_DEVICES=0 python trainSASVNet.py \
        --scoring \
        --initial_model exp/sasv_samo_finetune2/model/model000000007.model \
        --enroll_tsv "${ENROLL_TSV}" \
        --test_tsv "${TTS_BASE_DIR}/protocol_tts${TTS_NUM}_test.tsv" \
        --eval_path "${TTS_DIR}" \
        --enroll_path "${ENROLL_PATH}" \
        --save_path "exp/sasv_samo_tts${TTS_NUM}" \
        --model ReDimNet \
        --redimnet_model b2 \
        --trainfunc samo_sasv \
        --num_class 1160 \
        --batch_size 1 \
        --num_thread 8 \
        --use_samo_scoring
    
    echo "SAMO scores saved to: exp/sasv_samo_tts${TTS_NUM}/result/1_scorefile"
    echo ""
    
done

echo ""
echo "========================================================================"
echo "All TTS scoring complete!"
echo "========================================================================"
echo ""
echo "Score files generated:"
for TTS_NUM in 1 2 3 4 5 6 7; do
    echo "  TTS_${TTS_NUM}:"
    echo "    Cosine: exp/sasv_redimnet_tts${TTS_NUM}/result/1_scorefile"
    echo "    SAMO:   exp/sasv_samo_tts${TTS_NUM}/result/1_scorefile"
done
echo ""
echo "To analyze scores for a specific TTS system:"
echo "  python analyze_scores.py \\"
echo "    --score1 exp/sasv_redimnet_tts1/result/1_scorefile \\"
echo "    --score2 exp/sasv_samo_tts1/result/1_scorefile \\"
echo "    --name1 'Cosine' --name2 'SAMO'"
echo "========================================================================"

# SAMO Fine-tuning Commands

## Overview
This workflow fine-tunes a pretrained model using SAMO loss with speaker-aware attractors:
1. Load pretrained model checkpoint
2. Build speaker attractors from training data bonafide samples
3. Freeze SAMO centers (only update embedding network)
4. Train with lower learning rate

## Step 1: SAMO Fine-tuning

### Using SLURM
```bash
sbatch slurm_samo_finetune.slurm
```

### Manual command (interactive)
```bash
python trainSASVNet.py \
  --initial_model exp/sasv_redimnet/model/model000000008.model \
  --save_path exp/sasv_samo_finetune \
  --train_list corpus/spoofceleb/metadata/train.csv \
  --eval_list corpus/spoofceleb/protocol/sasv_development_evaluation_protocol.csv \
  --train_path corpus/spoofceleb/flac/train \
  --eval_path corpus/spoofceleb/flac/development \
  --spk_meta_train spk_meta/spk_meta_trn_spoofceleb.pk \
  --spk_meta_eval spk_meta/spk_meta_dev_spoofceleb.pk \
  --musan_path /path/to/dataset/MUSAN/musan_split \
  --rir_path /path/to/dataset/RIRS_NOISES/simulated_rirs \
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
```

## Step 2: Evaluate with SAMO scoring

```bash
python trainSASVNet.py \
  --eval \
  --use_enroll \
  --use_samo_scoring \
  --initial_model exp/sasv_samo_finetune/model/model000000010.model \
  --eval_list corpus/spoofceleb/protocol/sasv_development_evaluation_protocol.csv \
  --eval_path corpus/spoofceleb/flac/development \
  --trainfunc samo_sasv \
  --model ReDimNet \
  --redimnet_model b2 \
  --num_class 1160 \
  --num_out 192 \
  --save_path exp/sasv_samo_finetune_eval
```

## Step 3: Debug mode (test first few batches)

### Debug fine-tuning (10 batches, 1 epoch)
```bash
python trainSASVNet.py \
  --debug \
  --initial_model exp/sasv_redimnet/model/model000000008.model \
  --save_path exp/sasv_samo_finetune_debug \
  --train_list corpus/spoofceleb/metadata/train.csv \
  --eval_list corpus/spoofceleb/protocol/sasv_development_evaluation_protocol.csv \
  --train_path corpus/spoofceleb/flac/train \
  --eval_path corpus/spoofceleb/flac/development \
  --spk_meta_train spk_meta/spk_meta_trn_spoofceleb.pk \
  --spk_meta_eval spk_meta/spk_meta_dev_spoofceleb.pk \
  --musan_path /path/to/dataset/MUSAN/musan_split \
  --rir_path /path/to/dataset/RIRS_NOISES/simulated_rirs \
  --model ReDimNet \
  --redimnet_model b2 \
  --trainfunc samo_sasv \
  --num_class 1160 \
  --num_out 192 \
  --build_train_attractors \
  --freeze_samo_centers \
  --max_frames 400 \
  --num_spk 40 \
  --num_utt 2 \
  --batch_size 40 \
  --max_epoch 1 \
  --lr 1e-5
```

### Debug evaluation (9 trials)
```bash
python trainSASVNet.py \
  --eval \
  --debug \
  --use_enroll \
  --use_samo_scoring \
  --initial_model exp/sasv_samo_finetune/model/model000000010.model \
  --eval_list corpus/spoofceleb/protocol/sasv_development_evaluation_protocol.csv \
  --eval_path corpus/spoofceleb/flac/development \
  --trainfunc samo_sasv \
  --model ReDimNet \
  --redimnet_model b2 \
  --num_class 1160 \
  --num_out 192 \
  --save_path exp/sasv_samo_finetune_debug
```

## Step 4: Full training and evaluation (without --debug)

## Key Flags Explained

- `--build_train_attractors`: Build speaker attractors from **training set** bonafide samples
  * Extracts embeddings for all bonafide samples per speaker
  * Averages embeddings to create one attractor per speaker
  * Used during training to enable speaker-aware SAMO loss

- `--freeze_samo_centers`: Freeze SAMO centers during training
  * Centers remain fixed (not updated by gradients)
  * Only the embedding network (ReDimNet) is fine-tuned
  * Recommended for fine-tuning to avoid overfitting

- `--use_enroll`: Build enrollment dictionary from **evaluation set** for scoring
  * Used during evaluation/testing (not training)
  * Builds enrollments from validation/test protocols

- `--use_samo_scoring`: Use SAMO inference method instead of cosine similarity
  * Computes SAMO scores: max similarity to centers
  * Can use speaker-specific attractors if available

- `--debug`: Enable debug mode for quick testing
  * **For training**: Builds attractors for only 5 speakers, trains for 10 batches per epoch, stops after 1 epoch
  * **For evaluation**: Uses only 9 trials (3 target + 3 nontarget + 3 spoof) for balanced a-DCF
  * Shows detailed information: embeddings, speaker IDs, losses, scores
  * Use to verify pipeline works correctly before full training/evaluation

## Training vs Evaluation Modes

### Training with speaker attractors:
```
--build_train_attractors --freeze_samo_centers
```
- Uses training set bonafide samples
- Attractors are averaged embeddings per speaker
- Centers frozen, only embeddings updated

### Evaluation with enrollment:
```
--use_enroll --use_samo_scoring
```
- Uses eval/test set enrollment files from protocol
- Builds enrollment dictionary per speaker
- Uses SAMO scoring for trials

## Hyperparameter Recommendations

For fine-tuning:
- Learning rate: `--lr 1e-5` (10x lower than initial training)
- Epochs: `--max_epoch 10-20` (shorter than full training)
- Scheduler: Keep same cosine annealing settings but smaller T0
- Optimizer: AdamW with low weight decay

## Monitor Training

Check validation a-DCF every epoch:
```bash
tail -f exp/sasv_samo_finetune/result/metrics
```

## Compare Results

Compare different models/epochs:
```bash
# Original pretrained model
grep "a-DCF" exp/sasv_redimnet_eval/result/metrics

# SAMO fine-tuned
grep "a-DCF" exp/sasv_samo_finetune_eval/result/metrics
```

# Adapted version -- Baseline of the WildSpoof challenge SASV track.
- Adapted by Hye-jin Shim.

This version is an adaptation to be used as a baseline in the [WildSpoof](https://wildspoof.github.io/), Spoofing-robust Automatic Speaker Verification (SASV) track.

This baseline is based on the system proposed in a paper [Towards single integrated spoofing-aware speaker verification embeddings](https://www.isca-archive.org/interspeech_2023/mun23_interspeech.pdf), which was presented at Interspeech 2023.
This adapted version trains a Deep Neural Network (DNN) that has two output layers, one for speaker identification (multi-class classification) and the other for anti-spoofing (bonafide/spoof classification).


Please send an email to `shimhz6.6@gmail.com` for questions related to this adapted version for WildSpoof challenge SASV track.

## Prerequisites

### Activate a conda environment
- Not mandatory, but we recommend to initialize a conda envionment and match the development environment. This is because Deep Neural Network (DNN) models' results are not deterministic when trained multiple times. Here is the environment we used.
```bash
conda create --name {wildspoof_sasv}
conda install -y conda "python=${3.9.19}"
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia\n
```
- Afterward, install the packages via `pip install -r requirements.txt`.

## Models
This adaptation employs [SKA-TDNN](https://ieeexplore.ieee.org/iel7/10022052/10022330/10023305.pdf). You can also select other models implemented in the main branch (or your own model) using the `--model` option:

## Training
Train using the command below.

```bash
CUDA_VISIBLE_DEVICES=0 python trainSASVNet.py \
  --max_frames 400 \
  --num_spk 400 \
  --num_utt 2 \
  --batch_size 40 \
  --trainfunc sasv_e2e_v1 \
  --optimizer adamW \
  --scheduler cosine_annealing_warmup_restarts \
  --lr_t0 8 \
  --lr_tmul 1.0 \
  --lr_max 1e-4 \
  --lr_min 0 \
  --lr_wstep 0 \
  --lr_gamma 0.8 \
  --margin 0.2 \
  --scale 30 \
  --num_class 1160 \
  --save_path exp/sasv_baseline \
  --train_list corpus/spoofceleb/metadata/train.csv \
  --eval_list corpus/spoofceleb/protocol/sasv_development_evaluation_protocol.csv \
  --train_path corpus/spoofceleb/flac/train \
  --eval_path corpus/spoofceleb/flac/development \
  --spk_meta_train spk_meta/spk_meta_trn_spoofceleb.pk \
  --spk_meta_eval spk_meta/spk_meta_dev_spoofceleb.pk \
  --musan_path /path/to/dataset/MUSAN/musan_split \
  --rir_path /path/to/dataset/RIRS_NOISES/simulated_rirs \
  --model SKA_TDNN
```

## Evaluation
You can evaluate your checkpoint of a model using:
```bash
CUDA_VISIBLE_DEVICES=0 python trainSASVNet.py \
        --eval \
        --eval_frames 0 \
        --num_eval 1 \
        --eval_list corpus/spoofceleb/protocol/sasv_evaluation_evaluation_protocol.csv \
        --eval_path corpus/spoofceleb/flac/evaluation/ \
        --model SKA_TDNN \
        --initial_model /path/to/your_model/pretrained_weight.model
```

### Metric
We use the [Agnostic Detection Cost Function (a-DCF)](https://arxiv.org/abs/2403.01355) as the main metric, the primary metric used in the challenge to rank the submissions.

## Reference

```bibtex

@article{jung2025spoofceleb,
  title={SpoofCeleb: Speech deepfake detection and SASV in the wild},
  author={Jung, Jee-weon and Wu, Yihan and Wang, Xin and Kim, Ji-Hoon and Maiti, Soumi and Matsunaga, Yuta and Shim, Hye-jin and Tian, Jinchuan and Evans, Nicholas and Chung, Joon Son and others},
  journal={IEEE Open Journal of Signal Processing},
  year={2025},
  publisher={IEEE}
}

@article{jung2024text,
  title={The Text-to-speech in the Wild (TITW) Database},
  author={Jung, Jee-weon and Zhang, Wangyou and Maiti, Soumi and Wu, Yihan and Wang, Xin and Kim, Ji-Hoon and Matsunaga, Yuta and Um, Seyun and Tian, Jinchuan and Shim, Hye-jin and others},
  journal={ISCA Interspeech 2025},
  year={2024}
}

@inproceedings{mun2022frequency,
  title={Frequency and Multi-Scale Selective Kernel Attention for Speaker Verification},
  author={Mun, Sung Hwan and Jung, Jee-weon and Han, Min Hyun and Kim, Nam Soo},
  booktitle={Proc. IEEE SLT},
  year={2022}
}

@inproceedings{mun2023towards
  title={Towards single integrated spoofing-aware speaker verification embeddings},
  author={Mun, Sung Hwan and Shim, Hye-jin and Tak, Hemlata and Wang, Xin and Liu, Xuechen and Sahidullah, Md and Jeong, Myeonghun and Han, Min Hyun and Todisco, Massimiliano and Lee, Kong Aik and others},
  booktitle={Proc. Interspeech},
  year={2023}
}

@inproceedings{shim2024an
  title={a-DCF: an architecture agnostic metric with application to spoofing-robust speaker verification},
  author={Shim, Hye-jin and Jung, Jee-weon and Kinnunen, Tomi and Evans, Nicholas and Bonastre, Jean-Francois and Lapidot, Itshak},
  booktitle={Proc. Speaker Odyssey},
  year={2024}
}
```

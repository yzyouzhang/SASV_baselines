#!/usr/bin/python
#-*- coding: utf-8 -*-
import os
import sys
import time
import glob
import torch
import zipfile
import warnings
import argparse
import datetime
import torch.distributed as dist
import torch.multiprocessing as mp
from metrics import *
from SASVNet import *
from DatasetLoader import *
from tuneThreshold import *
import numpy as np

# Compatibility shim: older code (or third-party packages) may use deprecated
# NumPy aliases like `np.float`. Newer NumPy versions removed these aliases
# which causes AttributeError. Provide safe aliases here so packages such as
# `a_dcf` continue to work without modifying site-packages.
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

from a_dcf import a_dcf

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description = "SASVNet")
## Data loader
parser.add_argument('--max_frames',     type=int,   default=500,    help='Input length to the network for training')
parser.add_argument('--eval_frames',    type=int,   default=0,      help='Input length to the network for testing. 0 uses the whole files')
parser.add_argument('--num_eval',       type=int,   default=1,      help='Number of segments of input utterence for testing')
parser.add_argument('--num_spk',        type=int,   default=40,     help='Number of non-overlapped bona-fide speakers within a batch')
parser.add_argument('--num_utt',        type=int,   default=2,      help='Number of utterances per speaker within a batch')
parser.add_argument('--batch_size',     type=int,   default=160,    help='batch_size = num_spk*num_utt + num_spf, num_spf = batch_size - num_spk*num_utt')
parser.add_argument('--max_seg_per_spk',type=int,   default=10000,  help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--num_thread',     type=int,   default=10,     help='Number of loader threads')
parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')
parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')

## Training details
parser.add_argument('--test_interval',  type=int,   default=1,      help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=100,    help='Maximum number of epochs')
parser.add_argument('--trainfunc',      type=str,   default="aamsoftmax",     help='Loss function')

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam",     help='sgd, adam, adamW, or adamP')
parser.add_argument('--scheduler',      type=str,   default="cosine_annealing_warmup_restarts",     help='Learning rate scheduler')
parser.add_argument('--weight_decay',   type=float, default=1e-7,   help='Weight decay in the optimizer')
parser.add_argument('--lr',             type=float, default=1e-4,   help='Initial learning rate')
parser.add_argument('--lr_t0',          type=int,   default=8,      help='Cosine sched: First cycle step size')
parser.add_argument('--lr_tmul',        type=float, default=1.0,    help='Cosine sched: Cycle steps magnification.')
parser.add_argument('--lr_max',         type=float, default=1e-4,   help='Cosine sched: First cycle max learning rate')
parser.add_argument('--lr_min',         type=float, default=0,      help='Cosine sched: First cycle min learning rate')
parser.add_argument('--lr_wstep',       type=int,   default=0,      help='Cosine sched: Linear warmup step size')
parser.add_argument('--lr_gamma',       type=float, default=0.8,    help='Cosine sched: Decrease rate of max learning rate by cycle')

## Loss functions
parser.add_argument('--margin',         type=float, default=0.2,    help='Loss margin, only for some loss functions')
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
parser.add_argument('--num_class',      type=int,   default=41,     help='Number of speakers in the softmax layer, 1159 (speaker-classes) + 1 (spoofing-class)') # 41
parser.add_argument('--build_enroll',   dest='build_enroll', action='store_true', help='Build enrollment dictionary for SAMO loss from bonafide samples')
parser.add_argument('--use_enroll',     dest='use_enroll', action='store_true', help='Use enrollment dictionary with SAMO loss (requires pre-trained model)')
parser.add_argument('--use_samo_scoring', dest='use_samo_scoring', action='store_true', help='Use SAMO inference scoring instead of cosine similarity')

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
parser.add_argument('--save_path',      type=str,   default="./exp",     help='Path for model and logs')

## Training and test data
parser.add_argument('--train_list',     type=str,   default="",     help='Train list')
parser.add_argument('--eval_list',      type=str,   default="",     help='Evaluation list')
parser.add_argument('--train_path',     type=str,   default="",     help='Absolute path to the train set')
parser.add_argument('--eval_path',      type=str,   default="",     help='Absolute path to the test set')
parser.add_argument('--spk_meta_train', type=str,   default="",     help='')
parser.add_argument('--spk_meta_eval',  type=str,   default="",     help='')
parser.add_argument('--musan_path',     type=str,   default="",     help='Absolute path to the test set')
parser.add_argument('--rir_path',       type=str,   default="",     help='Absolute path to the test set')

## Model definition
parser.add_argument('--num_mels',       type=int,   default=80,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      type=bool,  default=True,   help='Log input features')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition')
parser.add_argument('--pooling_type',   type=str,   default="ASP",  help='Type of encoder')
parser.add_argument('--num_out',        type=int,   default=192,    help='Embedding size in the last FC layer')
parser.add_argument('--eca_c',          type=int,   default=1024,   help='ECAPA-TDNN channel')
parser.add_argument('--eca_s',          type=int,   default=8,      help='ECAPA-TDNN model-scale')

## ReDimNet-specific arguments
parser.add_argument('--redimnet_model',      type=str,   default='b2',                    help='ReDimNet model variant: b0, b1, b2, b3, b5, b6')
parser.add_argument('--redimnet_train_type', type=str,   default='ptn',                   help='ReDimNet training type: ptn, ft_lm, ft_mix')
parser.add_argument('--redimnet_dataset',    type=str,   default='vox2',                  help='ReDimNet dataset: vox2, vb2, vb2+vox2+cnc')
parser.add_argument('--redimnet_pretrained', type=bool,  default=True,                    help='Load pretrained ReDimNet weights')
parser.add_argument('--redimnet_repo',       type=str,   default='yzyouzhang/redimnet',   help='ReDimNet repository: yzyouzhang/redimnet or IDRnD/ReDimNet')

## Evaluation types
parser.add_argument('--eval',           dest='eval', action='store_true', help='Eval only')
parser.add_argument('--scoring',        dest='scoring', action='store_true', help='Scoring')
parser.add_argument('--enroll_list',      type=str,   default="corpus/ASVspoof5.dev.enroll.txt",     help='Evaluation enroll list')

args = parser.parse_args()

def main_worker(args):
    # args.gpu = gpu

    ## Load models
    s = SASVNet(**vars(args))
    s = WrappedModel(s).cuda()

    it = 1
    ## Write args to scorefile
    scorefile   = open(args.result_save_path+"/scores.txt", "a+", buffering=1)
    ## Print params
    pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())
    print('Total parameters: {:.2f}M'.format(float(pytorch_total_params)/1024/1024))

    trainer = ModelTrainer(s, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
    modelfiles.sort()
    if(args.initial_model != ""):
        trainer.loadParameters(args.initial_model)
        print("Model {} loaded!".format(args.initial_model))

    elif len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1])
        print("Model {} loaded from previous state!".format(modelfiles[-1]))
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1

    ## Build enrollment dictionary if using SAMO with speaker attractors
    if args.use_enroll and args.trainfunc in ["samo_sasv"]:
        if not args.eval_list or not args.eval_path:
            print("\n" + "="*50)
            print("WARNING: --use_enroll requires --eval_list and --eval_path")
            print("Skipping enrollment dictionary building.")
            print("The model will use SAMO without speaker-aware attractors.")
            print("="*50 + "\n")
        else:
            print("\n" + "="*50)
            print("Building enrollment dictionary from evaluation set...")
            print("(Training and evaluation speakers are disjoint)")
            print("="*50)
            
            # Build enrollment dict from evaluation/validation bonafide samples
            # Parse eval protocol to extract bonafide enrollment files
            import csv
            from collections import defaultdict
            
            eval_bonafide_files = defaultdict(list)
            with open(args.eval_list, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split(',')
                    if len(parts) >= 3:
                        enroll_file = parts[0]
                        label = parts[2]
                        # Only use bonafide (target) enrollments
                        if label == 'target':
                            if '/' in enroll_file:
                                speaker_id = enroll_file.split('/')[1]
                                eval_bonafide_files[speaker_id].append(enroll_file)
            
            # Get unique bonafide files
            all_bonafide = []
            for spk, files in eval_bonafide_files.items():
                all_bonafide.extend(list(set(files)))
            
            print(f"Found {len(all_bonafide)} unique bonafide enrollment files from {len(eval_bonafide_files)} speakers")
            
            # Extract embeddings for bonafide files
            from DatasetLoader import test_dataset_loader
            import torch.nn.functional as F
            
            test_dataset = test_dataset_loader(all_bonafide, args.eval_path, eval_frames=0, num_eval=1, 
                                              num_mels=args.num_mels, log_input=args.log_input)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                                     num_workers=args.num_thread, drop_last=False)
            
            # Set model to eval mode for embedding extraction
            s.eval()
            
            embeddings = {}
            import time
            tstart = time.time()
            for idx, data in enumerate(test_loader):
                inp = data[0][0].cuda()
                with torch.no_grad():
                    embed = s(inp).detach().cpu()
                embeddings[data[1][0]] = embed
                
                telapsed = time.time() - tstart
                sys.stdout.write(f"\r Extracting embeddings: {idx+1}/{len(all_bonafide)} - {(idx+1)/telapsed:.2f} Hz      ")
                sys.stdout.flush()
            
            print("\nAveraging embeddings per speaker...")
            
            # Average embeddings per speaker and create numeric mapping
            enroll_dict = {}
            speaker_id_map = {}
            unique_speakers = sorted(eval_bonafide_files.keys())
            
            for idx, speaker_str in enumerate(unique_speakers):
                speaker_embeds = []
                for audio_file in eval_bonafide_files[speaker_str]:
                    if audio_file in embeddings:
                        speaker_embeds.append(embeddings[audio_file])
                
                if speaker_embeds:
                    # Average and normalize
                    avg_embed = torch.stack(speaker_embeds).mean(dim=0).squeeze()
                    avg_embed = F.normalize(avg_embed.unsqueeze(0), p=2, dim=1).squeeze()
                    
                    # Map string speaker ID to numeric (extract number from "id10349" -> 10349)
                    try:
                        numeric_id = int(speaker_str.replace('id', ''))
                    except:
                        numeric_id = idx
                    
                    enroll_dict[numeric_id] = avg_embed
                    speaker_id_map[speaker_str] = numeric_id
            
            # Update the loss function with enrollment dictionary
            s.module.__L__.enroll_dict = enroll_dict
            s.module.__L__.use_speaker_attractor = True
            print(f"Enrollment dictionary built with {len(enroll_dict)} speakers")
            print(f"Enrollment embedding dimension: {list(enroll_dict.values())[0].shape}")
            print("="*50 + "\n")

    ## Scoring only
    if args.scoring == True:
        print('Test list',args.eval_list)
        sc, trials = trainer.evaluateFromList(**vars(args))

        savescore_file=os.path.join(args.result_save_path,f"{it}_scorefile")
        
        # Write submission format: speaker_id \t utterance_id \t score
        with open(savescore_file, "w") as tmp_scorefile:
            for _s, _trial in zip(sc, trials):
                # Extract scalar from array if needed
                score_val = _s[0] if hasattr(_s, '__len__') and len(_s) > 0 else _s
                tmp_scorefile.write(f"{_trial}\t{score_val}\n")

        msg = f"Complete scoring. save at " + savescore_file
        cur_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print('\n', cur_time, msg)
        scorefile.write(cur_time + " " + msg + "\n")
        scorefile.flush()
        scorefile.close()
        return
    ## Evaluation only
    if args.eval == True:
        print('Test list',args.eval_list)
        sc, lab, trials = trainer.evaluateFromList(**vars(args))

        # Write tmp_scorefile for a-DCF calculation (fixed format)
        with open("tmp_scorefile", "w") as tmp_scorefile:
            for _s, _l in zip(sc, lab):
                # Extract scalar from array if needed
                score_val = _s[0] if hasattr(_s, '__len__') and len(_s) > 0 else _s
                tmp_scorefile.write(f"s t {score_val} {_l}\n")
        
        # Write detailed score file with trial IDs
        detailed_file = os.path.join(args.result_save_path, f"{it}_eval_scores_detailed")
        with open(detailed_file, "w") as detail_file:
            for _s, _l, _trial in zip(sc, lab, trials):
                score_val = _s[0] if hasattr(_s, '__len__') and len(_s) > 0 else _s
                detail_file.write(f"{_trial} {score_val} {_l}\n")

        metric = a_dcf.calculate_a_dcf("tmp_scorefile")
        # os.remove("tmp_scorefile")

        msg = f"a-DCF {metric['min_a_dcf']:2.4f}, threshold: {metric['min_a_dcf_thresh']:2.4f}"
        cur_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print('\n', cur_time, msg)
        
        # Write to both metrics and scores.txt files
        with open(args.result_save_path + "/metrics", "a") as f_res:
            f_res.write(cur_time + "\n")
            f_res.write(msg + "\n")
        
        scorefile.write(cur_time + " " + msg + "\n")
        scorefile.flush()
        scorefile.close()

        return

    ## Initialise trainer and data loader
    train_dataset = train_dataset_loader(**vars(args))
    train_sampler = train_dataset_sampler(train_dataset, **vars(args))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size//2, #args.num_spk,
        num_workers=args.num_thread,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    ## Update learning rate
    for ii in range(1,it):
        trainer.__scheduler__.step()

    ## Save training code and params
    pyfiles = glob.glob('./*.py')
    strtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    zipf = zipfile.ZipFile(args.result_save_path+ '/run%s.zip'%strtime, 'w', zipfile.ZIP_DEFLATED)
    for file in pyfiles:
        zipf.write(file)
    zipf.close()
    with open(args.result_save_path + '/run%s.cmd'%strtime, 'w') as f:
        f.write('%s'%args)

    ## Core training script
    a_dcfs = []
    for it in range(it,args.max_epoch+1):

        ## Training
        train_sampler.set_epoch(it)
        loss, traineer, lr = trainer.train_network(train_loader, it)
        print('')

        ## Evaluating
        if it % args.test_interval == 0:
            sc, lab, trials = trainer.evaluateFromList(epoch=it, **vars(args))

            with open("tmp_scorefile", "w") as tmp_scorefile:
                for _s, _l in zip(sc, lab):
                    tmp_scorefile.write(f"s t {_s[0]} {_l}\n")
            
            # Write detailed scores for this epoch
            detailed_file = os.path.join(args.result_save_path, f"epoch{it:03d}_scores_detailed")
            with open(detailed_file, "w") as detail_file:
                for _s, _l, _trial in zip(sc, lab, trials):
                    detail_file.write(f"{_trial} {_s[0]} {_l}\n")

            metric = a_dcf.calculate_a_dcf("tmp_scorefile")
            os.remove("tmp_scorefile")
            a_dcfs.append(metric['min_a_dcf'])

            msg = f"a-DCF {metric['min_a_dcf']:2.4f}, threshold: {metric['min_a_dcf_thresh']:2.4f}"
            cur_time = time.strftime("%Y-%m-%d %H:%M:%S")
            print('\n', cur_time, msg)
            with open(args.result_save_path + "/metrics", "a") as f_res:
                f_res.write(cur_time + "\n")
                f_res.write(msg)

            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, ACC {:2.2f}, TLOSS {:f}, LR {:2.8f}, a-DCF {:2.4f}, Best a-DCF {:2.4f}".format(it, traineer, loss, lr, metric['min_a_dcf'], min(a_dcfs)))
            scorefile.write("Epoch {:d}, ACC {:2.2f}, TLOSS {:f}, LR {:2.8f}, a-DCF {:2.4f}, Best a-DCF {:2.4f}\n".format(it, traineer, loss, lr, metric['min_a_dcf'], min(a_dcfs)))
            scorefile.flush()
            trainer.saveParameters(args.model_save_path+"/model%09d.model"%it)
            print('')

    scorefile.close()


def main():

    args.model_save_path  = args.save_path+"/model"
    args.result_save_path = args.save_path+"/result"

    if os.path.exists(args.model_save_path): print("[Folder {} already exists...]".format(args.save_path))

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:',args.save_path)

    main_worker(args)

if __name__ == '__main__':
    main()

#!/usr/bin/python
#-*- coding: utf-8 -*-
import os
import sys
import time
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from DatasetLoader import test_dataset_loader
from torch.cuda.amp import autocast, GradScaler


class WrappedModel(nn.Module):
    
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x, label)


class SASVNet(nn.Module):

    def __init__(self, model, trainfunc, num_utt, **kwargs):
        super(SASVNet, self).__init__()
        SASVNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__S__ = SASVNetModel(**kwargs)
        LossFunction = importlib.import_module('loss.'+trainfunc).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**kwargs)
        self.num_utt = num_utt

    def forward(self, data, label=None, target_speaker=None):
        if label == None:
            return self.__S__.forward(data.reshape(-1, data.size()[-1]).cuda(), aug=False) 
        else:
            data = data.reshape(-1, data.size()[-1]).cuda() 
            outp = self.__S__.forward(data, aug=True)
            outp = outp.reshape(self.num_utt, -1, outp.size()[-1]).transpose(1,0).squeeze(1)
            nloss, prec1 = self.__L__.forward(outp, label, target_speaker=target_speaker)
            return nloss, prec1


class ModelTrainer(object):
    
    def __init__(self, speaker_model, optimizer, scheduler, **kwargs):
        self.__model__  = speaker_model
        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)
        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.__scheduler__, _ = Scheduler(self.__optimizer__, **kwargs)

        self.scaler = GradScaler() 
        self.gpu = 0
        self.ngpu = 1
        self.ndistfactor = int(kwargs.get('num_utt') * self.ngpu)
        
    def build_enrollment_dict(self, train_list, train_path, num_thread, eval_frames=0, num_eval=1, **kwargs):
        """
        Build enrollment dictionary by averaging bonafide embeddings per speaker.
        This should be called after loading a pre-trained model.
        
        Args:
            train_list: Path to training list file (CSV format: audio_path,speaker_id,label)
            train_path: Base path to audio files
            num_thread: Number of data loader threads
            eval_frames: Number of frames for evaluation (0 = full file)
            num_eval: Number of segments per utterance
            
        Returns:
            enroll_dict: Dictionary mapping speaker_id (int) to enrollment embedding (torch.Tensor)
        """
        print("\nBuilding enrollment dictionary from bonafide utterances...")
        self.__model__.eval()
        
        # Parse training list to get bonafide utterances per speaker
        speaker_files = {}  # {speaker_id: [list of audio files]}
        
        with open(train_list) as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 3:
                continue
            
            audio_file, speaker_id, label = parts[0], parts[1], parts[2]
            
            # Only use bonafide samples (label == speaker_id)
            if label == speaker_id:
                if speaker_id not in speaker_files:
                    speaker_files[speaker_id] = []
                speaker_files[speaker_id].append(audio_file)
        
        print(f"Found {len(speaker_files)} speakers with bonafide utterances")
        
        # Extract embeddings for all bonafide files
        all_files = []
        file_to_speaker = {}
        for speaker_id, files in speaker_files.items():
            for audio_file in files:
                all_files.append(audio_file)
                file_to_speaker[audio_file] = speaker_id
        
        # Create data loader
        test_dataset = test_dataset_loader(all_files, train_path, eval_frames=eval_frames, num_eval=num_eval, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_thread, drop_last=False)
        
        # Extract embeddings
        embeddings = {}  # {filename: embedding}
        tstart = time.time()
        
        for idx, data in enumerate(test_loader):
            inp = data[0][0].cuda()
            with torch.no_grad():
                embed = self.__model__(inp).detach().cpu()
            embeddings[data[1][0]] = embed
            
            telapsed = time.time() - tstart
            sys.stdout.write(f"\r Extracting embeddings: {idx+1}/{len(all_files)} - {(idx+1)/telapsed:.2f} Hz      ")
            sys.stdout.flush()
        
        print("\nAveraging embeddings per speaker...")
        
        # Average embeddings per speaker
        enroll_dict = {}
        speaker_id_map = {}  # Map string speaker IDs to integers
        
        # Build speaker ID mapping (assuming numeric speaker IDs in training)
        unique_speakers = sorted(speaker_files.keys())
        for idx, spk in enumerate(unique_speakers):
            speaker_id_map[spk] = idx
        
        for speaker_id, files in speaker_files.items():
            speaker_embeds = []
            for audio_file in files:
                if audio_file in embeddings:
                    speaker_embeds.append(embeddings[audio_file])
            
            if speaker_embeds:
                # Average all embeddings for this speaker
                avg_embed = torch.stack(speaker_embeds).mean(dim=0).squeeze()
                # Normalize the enrollment embedding
                avg_embed = F.normalize(avg_embed.unsqueeze(0), p=2, dim=1).squeeze()
                
                # Store with integer speaker ID
                numeric_id = speaker_id_map[speaker_id]
                enroll_dict[numeric_id] = avg_embed
        
        print(f"Built enrollment dictionary with {len(enroll_dict)} speakers")
        print(f"Enrollment embedding dimension: {enroll_dict[0].shape}")
        
        return enroll_dict, speaker_id_map

    def train_network(self, loader, epoch):
        self.__model__.train()
        self.__scheduler__.step(epoch-1)

        bs = loader.batch_size
        df = self.ndistfactor
        cnt, idx, loss, top1 = 0, 0, 0, 0
        tstart = time.time()

        for data, data_label, target_speaker in loader:

            self.__model__.zero_grad()
            data = data.transpose(1,0)
            label = torch.LongTensor(data_label).cuda()
            target_spk = torch.LongTensor(target_speaker).cuda()

            with autocast():
                nloss, prec1 = self.__model__(data, label, target_speaker=target_spk)

            self.scaler.scale(nloss).backward()
            self.scaler.step(self.__optimizer__)
            self.scaler.update()

            loss += nloss.detach().cpu().item()
            top1 += prec1.detach().cpu().item()
            cnt += 1
            idx += bs
            lr = self.__optimizer__.param_groups[0]['lr']
            telapsed = time.time() - tstart
            tstart = time.time()

            sys.stdout.write("\rProcessing {:d} of {:d}: Loss {:f}, ACC {:2.3f}%, LR {:.8f} - {:.2f} Hz  ".format(idx*df, loader.__len__()*bs*df, loss/cnt, top1/cnt, lr, bs*df/telapsed))
            sys.stdout.flush()

        return (loss/cnt, top1/cnt, lr)

    def evaluateFromList(self, eval_list, eval_path, num_thread, eval_frames=0, num_eval=1, **kwargs):

        rank = 0
        self.__model__.eval()

        ## Test loader ##
        tstart = time.time()
        
        # Check if eval_list file exists
        if not os.path.exists(eval_list):
            raise FileNotFoundError(f"Evaluation list file not found: {eval_list}")
        
        with open(eval_list) as f:
            lines_eval = f.readlines()
        
        if not lines_eval:
            raise ValueError(f"Evaluation list file is empty: {eval_list}")
            
        files = []
        for line in lines_eval:
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue
            parts = line.split(',')
            if len(parts) < 2:  # Skip malformed lines
                continue
            utt1, utt2 = parts[0], parts[1]
            files.append(utt1)
            files.append(utt2)
        setfiles = list(set(files))
        setfiles.sort()

        test_dataset = test_dataset_loader(setfiles, eval_path, eval_frames=eval_frames, num_eval=num_eval, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_thread, drop_last=False, sampler=None)

        ds = test_loader.__len__()
        gs = self.ngpu

        embeds_tst = {}
        for idx, data in enumerate(test_loader):
            inp1 = data[0][0].cuda()
            with torch.no_grad():
                ref_embed = self.__model__(inp1).detach().cpu()
            #embeds_tst[data[1][0][:-5]] = ref_embed
            embeds_tst[data[1][0]] = ref_embed
            telapsed = time.time() - tstart
            if rank == 0:
                sys.stdout.write("\r Reading {:d} of {:d}: {:.2f} Hz, embedding size {:d}      ".format(idx*gs, ds*gs, idx*gs/telapsed, ref_embed.size()[1]))
                sys.stdout.flush()

        ## Compute verification scores ##
        all_scores, all_labels, all_trials = [], [], []
        if rank == 0:
            tstart = time.time()
            print('')

            ## Read files and compute all scores
            for idx, line in enumerate(lines_eval):
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                data = line.split(",")
                if len(data) < 2:  # Skip malformed lines
                    continue
                    
                enr = embeds_tst[data[0]].cuda()
                tst = embeds_tst[data[1]].cuda()
                if self.__model__.module.__L__.test_normalize:
                    enr = F.normalize(enr, p=2, dim=1)
                    tst = F.normalize(tst, p=2, dim=1)

                # Choose scoring method based on configuration
                use_samo_scoring = kwargs.get('use_samo_scoring', False)
                if use_samo_scoring and hasattr(self.__model__.module.__L__, 'inference'):
                    # Extract speaker ID from enrollment file for SAMO scoring
                    enroll_path = data[0]
                    speaker_id = None
                    if '/' in enroll_path:
                        speaker_str = enroll_path.split('/')[1]  # e.g., "id10349"
                        # Try to map speaker string to numeric ID if speaker_id_map exists
                        # For now, we'll extract numeric part or use hash
                        # This assumes speaker IDs are in format "id10349" -> 10349
                        try:
                            speaker_id = int(speaker_str.replace('id', ''))
                        except:
                            speaker_id = None
                    
                    score = self.__model__.module.__L__.inference(enr, tst, enroll_speaker=speaker_id)
                    score = torch.tensor([score])  # Convert to tensor for consistency
                else:
                    # Default: cosine similarity
                    score = F.cosine_similarity(enr, tst)

                all_scores.append(score.detach().cpu().numpy())
                
                # Extract speaker ID from enrollment file path (e.g., "a00/id10349/..." -> "id10349")
                # Extract test utterance filename without extension
                enroll_path = data[0]
                test_path = data[1]
                speaker_id = enroll_path.split('/')[1] if '/' in enroll_path else enroll_path.split('_')[0]
                test_utt = os.path.splitext(os.path.basename(test_path))[0]
                
                all_trials.append(f"{speaker_id}\t{test_utt}")  # Tab-separated for submission format
                
                if (len(data) == 3): #kwargs["eval"]):
                    all_labels.append(data[2])

                telapsed = time.time() - tstart

                sys.stdout.write("\r Computing {:d} of {:d}: {:.2f} Hz      ".format(idx, len(lines_eval), idx/telapsed))
                sys.stdout.flush()

        if (kwargs["scoring"]):
            return all_scores, all_trials
        else:
            return (all_scores, all_labels, all_trials)

    def saveParameters(self, path):
        torch.save(self.__model__.module.state_dict(), path)

    def loadParameters(self, path):
        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path, map_location="cuda:%d"%self.gpu)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("{} is not in the model.".format(origname))
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)

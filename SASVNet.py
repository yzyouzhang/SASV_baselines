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

    def forward(self, x, label=None, target_speaker=None):
        return self.module(x, label, target_speaker=target_speaker)


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
        
    def build_enrollment_dict(self, train_list, train_path, num_thread, eval_frames=0, num_eval=1, debug=False, **kwargs):
        """
        Build enrollment dictionary by averaging bonafide embeddings per speaker.
        This should be called after loading a pre-trained model.
        
        Args:
            train_list: Path to training list file (CSV format: audio_path,speaker_id,label)
            train_path: Base path to audio files
            num_thread: Number of data loader threads
            eval_frames: Number of frames for evaluation (0 = full file)
            num_eval: Number of segments per utterance
            debug: If True, only process speakers from first 5 trials
            
        Returns:
            enroll_dict: Dictionary mapping speaker_id (int) to enrollment embedding (torch.Tensor)
        """
        print("\nBuilding enrollment dictionary from bonafide utterances...")
        self.__model__.eval()
        
        # In debug mode, first determine which speakers are needed
        debug_speakers = set()
        if debug:
            print("Debug mode: determining required speakers from first 5 trials...")
            with open(train_list) as f:
                lines = f.readlines()
            
            trial_count = 0
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) < 3:
                    continue
                
                # Extract speaker ID from enrollment file path or CSV column
                enroll_file = parts[0]
                if len(parts) >= 4:  # Test protocol format with speaker_id column
                    speaker_id = parts[2]
                else:  # Extract from path
                    speaker_id = enroll_file.split('/')[1] if '/' in enroll_file else enroll_file.split('_')[0]
                
                debug_speakers.add(speaker_id)
                trial_count += 1
                if trial_count >= 5:
                    break
            
            print(f"Debug mode: will build enrollment dict for {len(debug_speakers)} speakers: {sorted(debug_speakers)}")
        
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
            
            # In debug mode, skip speakers not in the debug set
            if debug and speaker_id not in debug_speakers:
                continue
            
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

    def train_network(self, loader, epoch, debug=False):
        self.__model__.train()
        self.__scheduler__.step(epoch-1)

        bs = loader.batch_size
        df = self.ndistfactor
        cnt, idx, loss, top1 = 0, 0, 0, 0
        tstart = time.time()

        for batch_idx, (data, data_label, target_speaker) in enumerate(loader):
            # Debug mode: limit to first 10 batches
            if debug and batch_idx >= 10:
                print(f"\n[DEBUG MODE] Stopping after {batch_idx} batches")
                break

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

            # Debug mode: show more detailed info
            if debug and batch_idx < 5:
                print(f"\n[DEBUG] Batch {batch_idx+1}:")
                print(f"  Data shape: {data.shape}")
                print(f"  Labels (speakers): {label.tolist()}")
                print(f"  Target speakers: {target_spk.tolist()}")
                print(f"  Loss: {nloss.item():.4f}, Acc: {prec1.item():.2f}%")
            
            if not debug or batch_idx % 5 == 0:
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
        
        # Debug mode: sample trials 
        debug_mode = kwargs.get('debug', False)
        if debug_mode:
            # Check if protocol has labels (validation) or not (test)
            first_line_parts = lines_eval[0].strip().split(',')
            has_labels = len(first_line_parts) >= 3 and first_line_parts[2] in ['target', 'nontarget', 'spoof']
            
            if has_labels:
                # Validation protocol: sample by label to get balanced set
                debug_lines = []
                label_lines = {'target': [], 'nontarget': [], 'spoof': []}
                for line in lines_eval:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        label = parts[2]
                        if label in label_lines:
                            label_lines[label].append(line)
                
                # Sample 3 from each category
                for label, lines_list in label_lines.items():
                    debug_lines.extend(lines_list[:3])
                
                lines_eval = debug_lines
                print(f"\n[DEBUG MODE] Limited to {len(lines_eval)} trials (3 target + 3 nontarget + 3 spoof)")
            else:
                # Test protocol: just take first N trials
                lines_eval = lines_eval[:10]
                print(f"\n[DEBUG MODE] Limited to {len(lines_eval)} trials")
            
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
        if rank == 0:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[{current_time}] Starting to read embeddings for {ds*gs} files...")
        
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
        
        if rank == 0:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"\n[{current_time}] Finished reading embeddings. Total time: {telapsed:.2f}s")

        ## Compute verification scores ##
        all_scores, all_labels, all_trials = [], [], []
        if rank == 0:
            tstart = time.time()
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[{current_time}] Starting to compute scores for {len(lines_eval)} trials...")

            # Choose scoring method based on configuration
            use_samo_scoring = kwargs.get('use_samo_scoring', False)
            debug_mode = kwargs.get('debug', False)
            
            # For SAMO scoring: track unique speaker+test pairs (skip duplicate enrollments)
            # For cosine similarity: accumulate scores per test utterance to average them
            from collections import defaultdict
            if use_samo_scoring:
                processed_trials = set()  # Track unique (speaker_id, test_utt) pairs
            else:
                trial_scores = defaultdict(list)  # Key: (speaker_id, test_utt), Value: list of scores
                trial_labels = {}  # Key: (speaker_id, test_utt), Value: label (if available)

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

                # Extract speaker ID and utterance ID first (needed for both methods)
                if len(data) >= 4:
                    speaker_id = data[2]
                    test_utt = data[3]
                else:
                    # Fallback: extract from file paths
                    enroll_path = data[0]
                    test_path = data[1]
                    speaker_id = enroll_path.split('/')[1] if '/' in enroll_path else enroll_path.split('_')[0]
                    test_utt = os.path.splitext(os.path.basename(test_path))[0]
                
                if use_samo_scoring and hasattr(self.__model__.module.__L__, 'inference'):
                    # SAMO scoring: uses speaker-level attractors
                    # Skip duplicate trials (same speaker+test with different enrollment files)
                    trial_key = (speaker_id, test_utt)
                    if trial_key in processed_trials:
                        if debug_mode:
                            print(f"\n[DEBUG] Trial {idx+1}: Skipping duplicate (already processed)")
                            print(f"  Speaker: {speaker_id}, Test: {test_utt}")
                        continue
                    
                    processed_trials.add(trial_key)
                    speaker_str = speaker_id
                    
                    # Convert speaker string to numeric ID
                    numeric_id = None
                    if speaker_str:
                        try:
                            # Try to extract numeric part (e.g., "id10349" -> 10349 or "spk001" -> 1)
                            numeric_id = int(''.join(filter(str.isdigit, speaker_str)))
                        except:
                            # If no digits, use the string itself if it's already numeric
                            try:
                                numeric_id = int(speaker_str)
                            except:
                                numeric_id = None
                    
                    if debug_mode:
                        print(f"\n[DEBUG] Trial {idx+1}: SAMO scoring")
                        print(f"  Enroll: {data[0][:60]}...")
                        print(f"  Test: {data[1][:60]}...")
                        print(f"  Speaker: {speaker_str} -> numeric_id={numeric_id}")
                        print(f"  Test embed norm: {tst.norm():.4f}")
                    
                    # SAMO inference uses pre-built enrollment dictionary (already averaged)
                    score = self.__model__.module.__L__.inference(enr, tst, enroll_speaker=numeric_id)
                    score = torch.tensor([score])  # Convert to tensor for consistency
                    
                    if debug_mode:
                        print(f"  SAMO score: {score.item():.6f}")
                    
                    # SAMO produces one score per speaker-test pair (enrollments already averaged)
                    all_scores.append(score.detach().cpu().numpy())
                    all_trials.append(f"{speaker_id}\t{test_utt}")
                    
                    if (len(data) == 3):
                        all_labels.append(data[2])
                else:
                    # Cosine similarity: compute score for this enrollment-test pair
                    score = F.cosine_similarity(enr, tst)
                    
                    if debug_mode:
                        print(f"\n[DEBUG] Trial {idx+1}: Cosine similarity")
                        print(f"  Enroll: {data[0][:60]}...")
                        print(f"  Test: {data[1][:60]}...")
                        print(f"  Speaker: {speaker_id}, Test: {test_utt}")
                        print(f"  Score: {score.item():.6f}")
                    
                    # Accumulate scores for this test utterance (will average later)
                    trial_key = (speaker_id, test_utt)
                    trial_scores[trial_key].append(score.detach().cpu().numpy())
                    
                    # Store label if available (same for all enrollments of this test utterance)
                    if (len(data) == 3) and trial_key not in trial_labels:
                        trial_labels[trial_key] = data[2]

                telapsed = time.time() - tstart
                sys.stdout.write("\r Computing {:d} of {:d}: {:.2f} Hz      ".format(idx, len(lines_eval), idx/telapsed))
                sys.stdout.flush()
            
            # For cosine similarity: average scores across multiple enrollments per test utterance
            if not use_samo_scoring:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"\n[{current_time}] Averaging scores across {len(trial_scores)} unique test utterances...")
                for trial_key in sorted(trial_scores.keys()):
                    speaker_id, test_utt = trial_key
                    scores = trial_scores[trial_key]
                    avg_score = sum(scores) / len(scores)  # Average across enrollment files
                    
                    if debug_mode and len(scores) > 1:
                        print(f"  {speaker_id}\t{test_utt}: {len(scores)} enrollments, scores={[f'{s[0]:.4f}' for s in scores]}, avg={avg_score[0]:.4f}")
                    
                    all_scores.append(avg_score)
                    all_trials.append(f"{speaker_id}\t{test_utt}")
                    
                    if trial_key in trial_labels:
                        all_labels.append(trial_labels[trial_key])
                
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"[{current_time}] Finished averaging. Final trial count: {len(all_scores)}")

        if (kwargs["scoring"]):
            return all_scores, all_trials
        else:
            return (all_scores, all_labels, all_trials)

    def evaluateFromTSV(self, enroll_tsv, test_tsv, eval_path, num_thread, eval_frames=0, num_eval=1, **kwargs):
        """
        Evaluate from TSV protocol files (for scoring mode).
        This builds speaker-level enrollment embeddings and compares them with test utterances.
        
        Args:
            enroll_tsv: Path to enrollment TSV (format: speaker_id enroll_file1,enroll_file2,...)
            test_tsv: Path to test TSV (format: speaker_id test_file)
            eval_path: Base path to test audio files
            num_thread: Number of data loader threads
            eval_frames: Number of frames for evaluation (0 = full file)
            num_eval: Number of segments per utterance
            enroll_path: (optional) Separate base path for enrollment audio files. If not provided, uses eval_path
        
        Returns:
            (scores, trials) tuple where trials are formatted as "speaker_id\tutt_id"
        """
        rank = 0
        self.__model__.eval()
        
        # Get separate enrollment path if provided
        enroll_path = kwargs.get('enroll_path', eval_path)
        
        tstart = time.time()
        
        # Step 1: Load enrollment protocol
        print("\n" + "="*70)
        print("Loading enrollment protocol from TSV...")
        speaker_enrollments = {}  # {speaker_id: [enroll_file1, enroll_file2, ...]}
        
        with open(enroll_tsv, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                
                speaker_id = parts[0]
                enroll_files = parts[1].split(',')
                # Add .flac extension if not present
                enroll_files = [f if f.endswith('.flac') else f + '.flac' for f in enroll_files]
                speaker_enrollments[speaker_id] = enroll_files
        
        print(f"Loaded enrollment data for {len(speaker_enrollments)} speakers")
        
        # Step 2: Load test protocol
        print("Loading test protocol from TSV...")
        test_trials = []  # [(speaker_id, test_file), ...]
        
        with open(test_tsv, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                
                speaker_id = parts[0]
                test_file = parts[1]
                # Add .flac extension if not present
                if not test_file.endswith('.flac'):
                    test_file = test_file + '.flac'
                
                if speaker_id in speaker_enrollments:
                    test_trials.append((speaker_id, test_file))
                else:
                    print(f"Warning: Speaker {speaker_id} not found in enrollment data")
        
        print(f"Loaded {len(test_trials)} test trials")
        print("="*70 + "\n")
        
        # Step 3: Separate enrollment and test files
        enroll_files = set()
        test_files = set()
        
        for enroll_list in speaker_enrollments.values():
            enroll_files.update(enroll_list)
        for _, test_file in test_trials:
            test_files.add(test_file)
        
        enroll_files = sorted(list(enroll_files))
        test_files = sorted(list(test_files))
        
        print(f"Total enrollment files: {len(enroll_files)}")
        print(f"Total test files: {len(test_files)}")
        
        # Step 4: Extract embeddings for enrollment files
        from DatasetLoader import test_dataset_loader
        embeddings = {}
        
        if rank == 0:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[{current_time}] Extracting enrollment embeddings from {enroll_path}...")
        
        enroll_dataset = test_dataset_loader(enroll_files, enroll_path, eval_frames=eval_frames, num_eval=num_eval, **kwargs)
        enroll_loader = torch.utils.data.DataLoader(enroll_dataset, batch_size=1, shuffle=False, 
                                                     num_workers=num_thread, drop_last=False, sampler=None)
        
        for idx, data in enumerate(enroll_loader):
            inp1 = data[0][0].cuda()
            with torch.no_grad():
                ref_embed = self.__model__(inp1).detach().cpu()
            embeddings[data[1][0]] = ref_embed
            
            telapsed = time.time() - tstart
            if rank == 0:
                sys.stdout.write("\r Reading enrollment {:d} of {:d}: {:.2f} Hz      ".format(
                    idx+1, len(enroll_files), (idx+1)/telapsed))
                sys.stdout.flush()
        
        if rank == 0:
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] Finished extracting enrollment embeddings")
        
        # Step 5: Extract embeddings for test files
        if rank == 0:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[{current_time}] Extracting test embeddings from {eval_path}...")
        
        test_dataset = test_dataset_loader(test_files, eval_path, eval_frames=eval_frames, num_eval=num_eval, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                                   num_workers=num_thread, drop_last=False, sampler=None)
        
        for idx, data in enumerate(test_loader):
            inp1 = data[0][0].cuda()
            with torch.no_grad():
                ref_embed = self.__model__(inp1).detach().cpu()
            embeddings[data[1][0]] = ref_embed
            
            telapsed = time.time() - tstart
            if rank == 0:
                sys.stdout.write("\r Reading test {:d} of {:d}: {:.2f} Hz      ".format(
                    idx+1, len(test_files), (idx+1)/telapsed))
                sys.stdout.flush()
        
        if rank == 0:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"\n[{current_time}] Finished extracting test embeddings")
        
        # Step 5: Build speaker-level enrollment embeddings (average across enrollment files)
        print("\nBuilding speaker-level enrollment embeddings...")
        speaker_embeds = {}
        
        for speaker_id, enroll_files in speaker_enrollments.items():
            enroll_embeds = []
            for enroll_file in enroll_files:
                if enroll_file in embeddings:
                    enroll_embeds.append(embeddings[enroll_file])
            
            if enroll_embeds:
                # Average enrollment embeddings
                avg_embed = torch.stack(enroll_embeds).mean(dim=0).squeeze()
                speaker_embeds[speaker_id] = avg_embed
        
        print(f"Built enrollment embeddings for {len(speaker_embeds)} speakers")
        
        # Step 6: Compute scores for each test trial
        print("\nComputing scores...")
        all_scores = []
        all_trials = []
        
        use_samo_scoring = kwargs.get('use_samo_scoring', False)
        
        if use_samo_scoring:
            print("Using SAMO scoring")
        else:
            print("Using cosine similarity scoring")
        
        for idx, (speaker_id, test_file) in enumerate(test_trials):
            # Get speaker enrollment embedding
            if speaker_id not in speaker_embeds:
                continue
            
            enr = speaker_embeds[speaker_id].cuda()
            tst = embeddings[test_file].cuda()
            
            # Normalize if needed
            if self.__model__.module.__L__.test_normalize:
                enr = F.normalize(enr.unsqueeze(0), p=2, dim=1).squeeze()
                tst = F.normalize(tst, p=2, dim=1)
            else:
                enr = enr.unsqueeze(0)
            
            # Compute score
            if use_samo_scoring and hasattr(self.__model__.module.__L__, 'inference'):
                # Extract numeric speaker ID for SAMO
                try:
                    numeric_id = int(''.join(filter(str.isdigit, speaker_id)))
                except:
                    numeric_id = None
                
                score = self.__model__.module.__L__.inference(enr, tst, enroll_speaker=numeric_id)
                score = torch.tensor([score])
            else:
                # Cosine similarity
                score = F.cosine_similarity(enr, tst)
            
            # Extract utterance ID from test file
            test_utt_id = os.path.basename(test_file)
            if test_utt_id.endswith('.flac'):
                test_utt_id = test_utt_id[:-5]
            
            all_scores.append(score.detach().cpu().numpy())
            all_trials.append(f"{speaker_id}\t{test_utt_id}")
            
            if (idx + 1) % 1000 == 0:
                sys.stdout.write(f"\r Computed {idx+1}/{len(test_trials)} scores      ")
                sys.stdout.flush()
        
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"\n[{current_time}] Finished computing {len(all_scores)} scores")
        
        return all_scores, all_trials


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

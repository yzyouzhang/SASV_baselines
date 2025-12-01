#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import loss.aamsoftmax as aamsoftmax
import loss.angleproto_sasv as angleproto_sasv

def generate_unique_two_hot(N, K):
    # Step 1: Generate all possible index pairs (i, j) where i < j
    index_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # # Step 2: Select the first K pairs (since the pairs are already unique and in lexicographical order)
    # selected_pairs = index_pairs[:K]

    # Step 2: Randomly select K pairs without repetition
    selected_pairs = random.sample(index_pairs, K)
    
    # Step 3: Create the K x N matrix using torch
    matrix = torch.zeros(K, N, dtype=torch.int)

    # Step 4: Set the selected pairs to 1 using torch.eye
    for idx, (i, j) in enumerate(selected_pairs):
        matrix[idx] = torch.eye(N, dtype=torch.int)[i] + torch.eye(N, dtype=torch.int)[j]
    
    return matrix

class SAMO_revised(nn.Module):
    def __init__(self, feat_dim=160, m_real=0.5, m_fake=0.2, alpha=20.0,
                 num_centers=20, initialize_centers="two_hot", addNegEntropy=False, device=torch.device("cuda")):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_centers = num_centers
        self.m_real = m_real
        self.m_fake = m_fake
        self.alpha = alpha
        self.addNegEntropy = addNegEntropy
        self.softplus = nn.Softplus()

        # ---- centers as BUFFER so device/state_dict are handled ----
        if initialize_centers == "one_hot":
            C = torch.eye(self.feat_dim)[:self.num_centers]
        elif initialize_centers == "two_hot":
            # assumes you have this util; ensure each row has two 1s
            C = generate_unique_two_hot(self.feat_dim, self.num_centers).float()
            # inference mode
            # self.center = nn.Parameter(generate_unique_two_hot(self.feat_dim, self.num_centers).float())
            # training mode
            # self.center = generate_unique_two_hot(self.feat_dim, self.num_centers).float()
        elif initialize_centers == "random":
            # learnable centers variant (not paper-faithful SAMO)
            self.center = nn.Parameter(torch.randn(self.num_centers, self.feat_dim))
            nn.init.normal_(self.center, std=0.02)
            self._is_param_centers = True
        else:
            raise ValueError("Unknown center init")

        if initialize_centers != "random":
            self.register_buffer("center", C.to(device))
            self._is_param_centers = False

        if self.addNegEntropy:
            self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def _safe_normalize(x, dim, eps=1e-8):
        return x / (x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))

    def forward(self, x, labels, spk=None, enroll=None, attractor=0):
        # x: [B, D] (already an embedding)
        x = self._safe_normalize(x, dim=1)
        w = self._safe_normalize(self.center, dim=1).to(x.device)  # [K, D]
        # raw similarities to all centers
        all_scores = x @ w.t()                        # [B, K]
        max_cos, _ = all_scores.max(dim=1, keepdim=True)  # [B, 1]

        # ======== EVALUATION MODE ========
        if labels is None:
            # return RAW similarity for scoring (not margin-shifted)
            if attractor == 1 and spk is not None and enroll is not None:
                # Override with speaker enrollment when available
                final_scores = max_cos.clone()
                for idx in range(len(spk)):
                    if spk[idx] in enroll:
                        tmp_w = self._safe_normalize(enroll[spk[idx]].unsqueeze(0), dim=1).to(x.device)
                        final_scores[idx] = (x[idx] @ tmp_w.t()).squeeze()
                return final_scores.squeeze(1), None
            else:
                return max_cos.squeeze(1), None

        # ======== TRAINING MODE ========
        # Build the similarity vector used for loss computation
        # Start with max_cos for all samples
        loss_scores = max_cos.clone()  # [B, 1]

        if attractor == 1 and spk is not None and enroll is not None:
            # Compute speaker-specific scores for ALL samples
            # Note: This FIXES a bug in original SAMO where tmp_w_lst is initialized
            # inside the loop, causing only the last speaker's embedding to be used.
            # Here we correctly build one embedding per sample.
            tmp_w_lst = []
            for id in spk:
                if id in enroll:
                    tmp_w_lst.append(enroll[id].clone())
                else:
                    tmp_w_lst.append(torch.zeros(self.feat_dim, device=x.device))
            tmp_w = torch.stack(tmp_w_lst)  # [B, D]
            tmp_w = F.normalize(tmp_w, p=2, dim=1).to(x.device)
            speaker_scores = (x * tmp_w).sum(dim=1, keepdim=True)  # [B, 1]

            # Only override bonafide samples with speaker scores
            # Spoofs always use max_cos
            if (labels == 1).any():
                loss_scores[labels == 1] = speaker_scores[labels == 1]

        # Apply margins for OC-Softmax loss
        loss_scores[labels == 1] = self.m_real - loss_scores[labels == 1]   # bonafide
        loss_scores[labels == 0] = loss_scores[labels == 0] - self.m_fake   # spoof
        emb_loss = self.softplus(self.alpha * loss_scores).mean()

        if self.addNegEntropy:
            # Encourage spoofs to spread across centers (match SAMO implementation)
            scores_softmax = self.softmax(all_scores[labels == 0])
            p = scores_softmax.sum(0).view(-1)
            p /= p.sum()

            dist_loss = np.log(w.shape[0]) + (p * p.log()).sum()  # using num_centers

            loss = dist_loss * 1e5 + emb_loss
        else:
            loss = emb_loss

        # Return raw similarity (pre-margin) for downstream scoring use
        return max_cos.squeeze(1), loss

    @torch.no_grad()
    def inference(self, x, spk=None, enroll=None, attractor=0):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).

            Able to deal with samples without enrollment in scenario args.target=0
        """
        x = self._safe_normalize(x, dim=1)
        w = self._safe_normalize(self.center, dim=1)
        all_scores = x @ w.t()
        max_cos, _ = all_scores.max(dim=1, keepdim=True)

        if attractor == 1 and spk is not None and enroll is not None:
            e_rows = []
            for i in range(x.shape[0]):
                if spk[i] in enroll:
                    e_rows.append(enroll[spk[i]].to(x.device))
                else:
                    e_rows.append(torch.zeros(self.feat_dim, device=x.device))
            E = torch.stack(e_rows, dim=0)
            E = self._safe_normalize(E, dim=1)
            enroll_cos = (x * E).sum(dim=1, keepdim=True)
            has_enroll = (E.abs().sum(dim=1, keepdim=True) > 0)
            final = torch.where(has_enroll, enroll_cos, max_cos)
        else:
            final = max_cos

        return final.squeeze(1)

class SAMO_SASV(nn.Module):
    def __init__(self, enroll_dict=None, use_speaker_attractor=False, **kwargs):
        super(SAMO_SASV, self).__init__()
        self.test_normalize = True
        self.aamsoftmax = aamsoftmax.LossFunction(**kwargs)
        
        # Initialize SAMO_revised for spoofing detection
        feat_dim = kwargs.get('num_out', 192)
        self.samo = SAMO_revised(
            feat_dim=feat_dim,
            m_real=0.7,
            m_fake=0.0,
            alpha=20.0,
            num_centers=1058,
            initialize_centers="two_hot",
            addNegEntropy=True,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        self.num_class = kwargs.get('num_class')
        self.enroll_dict = enroll_dict  # Store enrollment dictionary
        self.use_speaker_attractor = use_speaker_attractor  # Whether to use speaker-aware attractors
        self.feat_dim = feat_dim  # Store for inference
        print('Initialized SASV with AAM-Softmax + SAMO Loss')

    def forward(self, x, label=None, target_speaker=None):
        # Sample i (bonafide speaker A):
        # ├── Utterance 1 from speaker A → embedding [192]
        # └── Utterance 2 from speaker A → embedding [192]
        # Sample j (spoof targeting speaker B):
        # ├── Spoof sample 1 → embedding [192]  
        # └── Spoof sample 2 → embedding [192]
        assert x.size()[1] == 2
        # AAM-Softmax loss for speaker classification
        nlossS, prec = self.aamsoftmax(x.reshape(-1, x.size()[-1]), label.repeat_interleave(2))
        
        # Prepare data for SAMO loss
        # Reshape: [B, 2, D] -> [B*2, D]
        x_flat = x.reshape(-1, x.size()[-1])  # [B*2, D]
        
        # Create binary labels for SAMO: 1=bonafide, 0=spoof
        # label: [B] where last class (num_class-1) is spoof
        label_repeated = label.repeat_interleave(2)  # [B*2]
        samo_labels = (label_repeated != self.num_class-1).long()  # 1 for bonafide, 0 for spoof
        
        # Prepare speaker list for SAMO if using speaker-aware attractors
        spk_list = None
        attractor_mode = 0
        
        if self.use_speaker_attractor and target_speaker is not None and self.enroll_dict is not None:
            # Create speaker list for flattened embeddings [B*2]
            # Each speaker appears twice (for 2 utterances)
            spk_list = [spk_id.item() if isinstance(spk_id, torch.Tensor) else spk_id 
                       for spk_id in target_speaker for _ in range(2)]
            attractor_mode = 1
        
        # SAMO loss for spoofing detection with speaker-aware attractor
        _, nlossM = self.samo(
            x_flat, 
            samo_labels, 
            spk=spk_list,
            enroll=self.enroll_dict,
            attractor=attractor_mode
        )
        
        return nlossS + nlossM, prec
    
    @torch.no_grad()
    def inference(self, enroll_embed, test_embed, enroll_speaker=None):
        """
        SAMO-based scoring for evaluation/testing.
        Computes SAMO similarity scores between enrollment and test embeddings.
        
        Args:
            enroll_embed: Enrollment embedding [1, D]
            test_embed: Test embedding [1, D]
            enroll_speaker: Speaker ID (int) for speaker-aware scoring (optional)
            
        Returns:
            score: Similarity score (higher = more likely bonafide)
        """
        # Use SAMO inference for test embedding
        attractor_mode = 0
        spk_list = None
        
        if self.use_speaker_attractor and enroll_speaker is not None and self.enroll_dict is not None:
            # Use speaker-aware attractor
            spk_list = [enroll_speaker]
            attractor_mode = 1
        
        # Get SAMO score for test embedding
        test_score = self.samo.inference(
            test_embed,
            spk=spk_list,
            enroll=self.enroll_dict,
            attractor=attractor_mode
        )
        
        # Also get score for enrollment (should be high if bonafide)
        enroll_score = self.samo.inference(
            enroll_embed,
            spk=spk_list,
            enroll=self.enroll_dict,
            attractor=attractor_mode
        )
        
        # Combine scores: average of enrollment and test scores
        # This gives a measure of how "bonafide-like" both embeddings are
        combined_score = (enroll_score + test_score) / 2.0
        
        return combined_score.item()


# Alias for compatibility with training script
LossFunction = SAMO_SASV

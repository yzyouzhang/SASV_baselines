#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze and compare scores from different scoring methods (e.g., cosine vs SAMO).
Can also merge scores by averaging them.

Usage:
  # Analyze two score files
  python analyze_scores.py --score1 exp/model1/result/scores.txt --score2 exp/model2/result/scores.txt

  # Merge two score files by averaging
  python analyze_scores.py --score1 exp/model1/result/scores.txt --score2 exp/model2/result/scores.txt \
                          --merge --output merged_scores.txt

  # Analyze with weights for merging
  python analyze_scores.py --score1 exp/model1/result/scores.txt --score2 exp/model2/result/scores.txt \
                          --merge --weights 0.6 0.4 --output merged_scores.txt
"""

import argparse
import numpy as np
from collections import defaultdict
import sys

def load_scores(score_file):
    """
    Load scores from file.
    Expected format: speaker_id \t utterance_id \t score
    or: speaker_id utterance_id score (space-separated)
    
    Returns:
        dict: {(speaker_id, utt_id): score}
        list: [(speaker_id, utt_id, score)] in order
    """
    scores_dict = {}
    scores_list = []
    
    with open(score_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Try tab-separated first, then space-separated
            parts = line.split('\t') if '\t' in line else line.split()
            
            if len(parts) < 3:
                print(f"Warning: Skipping malformed line {line_num} in {score_file}: {line}")
                continue
            
            speaker_id = parts[0]
            utt_id = parts[1]
            try:
                score = float(parts[2])
            except ValueError:
                print(f"Warning: Invalid score on line {line_num} in {score_file}: {parts[2]}")
                continue
            
            key = (speaker_id, utt_id)
            scores_dict[key] = score
            scores_list.append((speaker_id, utt_id, score))
    
    return scores_dict, scores_list

def analyze_scores(scores_dict, name="Scores"):
    """
    Compute statistics for a set of scores.
    """
    scores = np.array(list(scores_dict.values()))
    
    print(f"\n{'='*60}")
    print(f"{name} Statistics")
    print(f"{'='*60}")
    print(f"  Number of trials: {len(scores)}")
    print(f"  Mean:             {np.mean(scores):.6f}")
    print(f"  Std:              {np.std(scores):.6f}")
    print(f"  Min:              {np.min(scores):.6f}")
    print(f"  Max:              {np.max(scores):.6f}")
    print(f"  Median:           {np.median(scores):.6f}")
    print(f"  25th percentile:  {np.percentile(scores, 25):.6f}")
    print(f"  75th percentile:  {np.percentile(scores, 75):.6f}")

def compare_scores(scores1_dict, scores2_dict, name1="Method 1", name2="Method 2"):
    """
    Compare two sets of scores.
    """
    # Find common trials
    keys1 = set(scores1_dict.keys())
    keys2 = set(scores2_dict.keys())
    common_keys = keys1 & keys2
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    
    print(f"\n{'='*60}")
    print(f"Comparison: {name1} vs {name2}")
    print(f"{'='*60}")
    print(f"  Trials in {name1}: {len(keys1)}")
    print(f"  Trials in {name2}: {len(keys2)}")
    print(f"  Common trials:     {len(common_keys)}")
    print(f"  Only in {name1}:   {len(only_in_1)}")
    print(f"  Only in {name2}:   {len(only_in_2)}")
    
    if only_in_1:
        print(f"\n  Warning: {len(only_in_1)} trials only in {name1}")
        if len(only_in_1) <= 10:
            for key in list(only_in_1)[:10]:
                print(f"    {key[0]} {key[1]}")
    
    if only_in_2:
        print(f"\n  Warning: {len(only_in_2)} trials only in {name2}")
        if len(only_in_2) <= 10:
            for key in list(only_in_2)[:10]:
                print(f"    {key[0]} {key[1]}")
    
    if len(common_keys) == 0:
        print("\n  ERROR: No common trials found!")
        return None, None
    
    # Compute correlation and differences for common trials
    scores1 = np.array([scores1_dict[k] for k in common_keys])
    scores2 = np.array([scores2_dict[k] for k in common_keys])
    
    correlation = np.corrcoef(scores1, scores2)[0, 1]
    diff = scores1 - scores2
    
    print(f"\n  Correlation:       {correlation:.4f}")
    print(f"  Mean difference:   {np.mean(diff):.6f}")
    print(f"  Std difference:    {np.std(diff):.6f}")
    print(f"  Max abs diff:      {np.max(np.abs(diff)):.6f}")
    print(f"  Median difference: {np.median(diff):.6f}")
    
    # Find trials with largest differences
    abs_diff = np.abs(diff)
    top_indices = np.argsort(abs_diff)[-10:][::-1]
    
    print(f"\n  Top 10 trials with largest score differences:")
    print(f"    {'Speaker':<20} {'Utterance':<20} {name1:<12} {name2:<12} {'Diff':>8}")
    print(f"    {'-'*75}")
    common_keys_list = list(common_keys)
    for idx in top_indices:
        key = common_keys_list[idx]
        s1 = scores1_dict[key]
        s2 = scores2_dict[key]
        print(f"    {key[0]:<20} {key[1]:<20} {s1:>12.6f} {s2:>12.6f} {s1-s2:>8.4f}")
    
    return scores1, scores2

def merge_scores(scores1_dict, scores2_dict, weights=(0.5, 0.5), output_file=None):
    """
    Merge two score files by weighted average.
    Only includes trials present in both score files.
    
    Args:
        scores1_dict: First score dictionary
        scores2_dict: Second score dictionary
        weights: Tuple of (weight1, weight2), should sum to 1.0
        output_file: Output file path
    
    Returns:
        dict: Merged scores
    """
    w1, w2 = weights
    
    # Normalize weights
    total = w1 + w2
    w1, w2 = w1/total, w2/total
    
    # Find common trials
    common_keys = set(scores1_dict.keys()) & set(scores2_dict.keys())
    
    if len(common_keys) == 0:
        print("ERROR: No common trials found for merging!")
        return None
    
    print(f"\n{'='*60}")
    print(f"Merging Scores")
    print(f"{'='*60}")
    print(f"  Weight 1: {w1:.3f}")
    print(f"  Weight 2: {w2:.3f}")
    print(f"  Common trials: {len(common_keys)}")
    
    merged_scores = {}
    for key in common_keys:
        merged_score = w1 * scores1_dict[key] + w2 * scores2_dict[key]
        merged_scores[key] = merged_score
    
    # Write to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            for key in sorted(merged_scores.keys()):
                speaker_id, utt_id = key
                score = merged_scores[key]
                f.write(f"{speaker_id}\t{utt_id}\t{score:.6f}\n")
        print(f"  Merged scores written to: {output_file}")
    
    return merged_scores

def main():
    parser = argparse.ArgumentParser(description='Analyze and compare scoring methods')
    parser.add_argument('--score1', type=str, required=True,
                        help='First score file')
    parser.add_argument('--score2', type=str, required=True,
                        help='Second score file')
    parser.add_argument('--name1', type=str, default='Method 1',
                        help='Name for first scoring method')
    parser.add_argument('--name2', type=str, default='Method 2',
                        help='Name for second scoring method')
    parser.add_argument('--merge', action='store_true',
                        help='Merge the two score files by averaging')
    parser.add_argument('--weights', type=float, nargs=2, default=[0.5, 0.5],
                        help='Weights for merging (default: 0.5 0.5)')
    parser.add_argument('--output', type=str, default='merged_scores.txt',
                        help='Output file for merged scores')
    
    args = parser.parse_args()
    
    # Load scores
    print(f"Loading scores from {args.score1}...")
    scores1_dict, scores1_list = load_scores(args.score1)
    print(f"  Loaded {len(scores1_dict)} trials")
    
    print(f"Loading scores from {args.score2}...")
    scores2_dict, scores2_list = load_scores(args.score2)
    print(f"  Loaded {len(scores2_dict)} trials")
    
    # Analyze individual score files
    analyze_scores(scores1_dict, name=args.name1)
    analyze_scores(scores2_dict, name=args.name2)
    
    # Compare the two score files
    compare_scores(scores1_dict, scores2_dict, name1=args.name1, name2=args.name2)
    
    # Merge if requested
    if args.merge:
        merged_scores = merge_scores(scores1_dict, scores2_dict, 
                                     weights=tuple(args.weights), 
                                     output_file=args.output)
        if merged_scores:
            analyze_scores(merged_scores, name="Merged Scores")
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()

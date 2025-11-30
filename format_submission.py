#!/usr/bin/env python3
"""
Format SASV scores for WildSpoof submission.
Converts from internal score format to: SPK_ID UTT_ID SCORE
"""
import argparse

def main():
    parser = argparse.ArgumentParser(description='Format scores for submission')
    parser.add_argument('--score_file', type=str, required=True,
                       help='Input score file from scoring (e.g., exp/sasv_redimnet_test/result/*_scorefile)')
    parser.add_argument('--test_protocol', type=str, required=True,
                       help='Original test protocol TSV file (protocol_test.tsv)')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output submission file')
    parser.add_argument('--aggregate_enrollments', action='store_true',
                       help='Average scores when multiple enrollment files exist for same trial')
    args = parser.parse_args()
    
    # Read original test protocol to get speaker IDs
    test_trials = []
    with open(args.test_protocol, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                spk_id, test_utt = parts
                test_trials.append((spk_id, test_utt))
    
    # Read scores
    scores = []
    with open(args.score_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('s ') or line.startswith('s\t'):
                parts = line.split()
                if len(parts) >= 2:
                    score = float(parts[1])
                    scores.append(score)
    
    # Check if we need to aggregate scores
    if args.aggregate_enrollments and len(scores) != len(test_trials):
        print(f"Aggregating scores: {len(scores)} scores for {len(test_trials)} trials")
        # Group scores by trial (assuming they appear in order)
        ratio = len(scores) // len(test_trials)
        aggregated_scores = []
        for i in range(len(test_trials)):
            trial_scores = scores[i*ratio:(i+1)*ratio]
            avg_score = sum(trial_scores) / len(trial_scores)
            aggregated_scores.append(avg_score)
        scores = aggregated_scores
    
    # Verify alignment
    if len(scores) != len(test_trials):
        print(f"Warning: Mismatch between scores ({len(scores)}) and trials ({len(test_trials)})")
        min_len = min(len(scores), len(test_trials))
        scores = scores[:min_len]
        test_trials = test_trials[:min_len]
    
    # Write submission file
    with open(args.output_file, 'w') as f:
        for (spk_id, test_utt), score in zip(test_trials, scores):
            f.write(f"{spk_id}\t{test_utt}\t{score:.5f}\n")
    
    print(f"Submission file saved to {args.output_file}")
    print(f"Total trials: {len(test_trials)}")

if __name__ == '__main__':
    main()

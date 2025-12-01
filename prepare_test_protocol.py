#!/usr/bin/env python3
"""
Convert WildSpoof test protocol (TSV format with enrollment) to SASV evaluation format (CSV).
"""
import argparse

def main():
    parser = argparse.ArgumentParser(description='Prepare test protocol for scoring')
    parser.add_argument('--enroll_file', type=str, required=True, 
                       help='Enrollment protocol TSV file (e.g., protocol_enroll.tsv)')
    parser.add_argument('--test_file', type=str, required=True,
                       help='Test protocol TSV file (e.g., protocol_test.tsv)')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output CSV file for scoring')
    parser.add_argument('--use_first_enroll', action='store_true',
                       help='Use only the first enrollment file when multiple are available')
    args = parser.parse_args()
    
    # Read enrollment data
    enroll_dict = {}
    with open(args.enroll_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                spk_id = parts[0]
                enroll_files = parts[1].split(',')
                # Add .flac extension if not present
                enroll_files = [f if f.endswith('.flac') else f + '.flac' for f in enroll_files]
                if args.use_first_enroll:
                    enroll_dict[spk_id] = enroll_files[0]
                else:
                    # Use all enrollment files (will create multiple trials)
                    enroll_dict[spk_id] = enroll_files
    
    # Read test protocol and create CSV with speaker and utterance IDs
    with open(args.test_file, 'r') as f_in, open(args.output_file, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) == 2:
                spk_id = parts[0]
                test_file = parts[1]
                
                # Extract utterance ID (filename without extension)
                utt_id = test_file.split('/')[-1] if '/' in test_file else test_file
                if utt_id.endswith('.flac'):
                    utt_id = utt_id[:-5]
                
                # Add .flac extension if not present
                if not test_file.endswith('.flac'):
                    test_file = test_file + '.flac'
                
                if spk_id in enroll_dict:
                    enroll_files = enroll_dict[spk_id]
                    if isinstance(enroll_files, str):
                        # Single enrollment file
                        # Format: enroll_file,test_file,speaker_id,utt_id
                        f_out.write(f"{enroll_files},{test_file},{spk_id},{utt_id}\n")
                    else:
                        # Multiple enrollment files - create trial for each
                        for enroll_file in enroll_files:
                            f_out.write(f"{enroll_file},{test_file},{spk_id},{utt_id}\n")
                else:
                    print(f"Warning: Speaker {spk_id} not found in enrollment file")
    
    print(f"Converted protocol saved to {args.output_file}")

if __name__ == '__main__':
    main()

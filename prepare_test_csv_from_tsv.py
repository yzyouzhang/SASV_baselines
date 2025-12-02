#!/usr/bin/env python3
"""
Convert test TSV protocol to CSV format compatible with evaluation protocol.
This directly reads the enrollment and test TSV files without needing pickle files.

Input formats:
- Enrollment TSV: speaker_id enroll_file1,enroll_file2,...
- Test TSV: speaker_id test_file

Output CSV format: enroll_file,test_file,speaker_id,test_utt_id

This matches the --eval protocol format (enroll_file,test_file,label) except:
- No label column (since this is for test/scoring)
- Added speaker_id and test_utt_id columns for output formatting
"""
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Convert test TSV to CSV evaluation format')
    parser.add_argument('--enroll_file', type=str, required=True,
                       help='Enrollment protocol TSV file (format: speaker_id enroll_file1,enroll_file2,...)')
    parser.add_argument('--test_file', type=str, required=True,
                       help='Test protocol TSV file (format: speaker_id test_file)')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output CSV file')
    parser.add_argument('--expand_enrollments', action='store_true',
                       help='Create one row per enrollment file (for averaging). If not set, use only first enrollment.')
    args = parser.parse_args()
    
    # Load enrollment metadata from TSV
    spk_enrollments = {}
    with open(args.enroll_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) != 2:
                print(f"Warning: Skipping malformed enrollment line: {line}")
                continue
            
            speaker_id = parts[0]
            enroll_files = parts[1].split(',')
            
            # Add .flac extension if not present
            enroll_files = [f if f.endswith('.flac') else f + '.flac' for f in enroll_files]
            spk_enrollments[speaker_id] = enroll_files
    
    print(f"Loaded enrollment data for {len(spk_enrollments)} speakers")
    
    # Parse test protocol and generate CSV
    output_lines = []
    missing_speakers = set()
    
    with open(args.test_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) != 2:
                print(f"Warning: Skipping malformed test line: {line}")
                continue
            
            speaker_id = parts[0]
            test_file = parts[1]
            
            # Add .flac extension if not present
            if not test_file.endswith('.flac'):
                test_file = test_file + '.flac'
            
            # Extract test utterance ID (filename without extension)
            test_utt_id = os.path.basename(test_file)
            if test_utt_id.endswith('.flac'):
                test_utt_id = test_utt_id[:-5]
            
            # Get enrollment files for this speaker
            if speaker_id not in spk_enrollments:
                if speaker_id not in missing_speakers:
                    print(f"Warning: Speaker {speaker_id} not found in enrollment data")
                    missing_speakers.add(speaker_id)
                continue
            
            enroll_files = spk_enrollments[speaker_id]
            
            if args.expand_enrollments:
                # Create one row per enrollment file (these will be averaged during scoring)
                for enroll_file in enroll_files:
                    output_lines.append(f"{enroll_file},{test_file},{speaker_id},{test_utt_id}")
            else:
                # Use only the first enrollment file
                enroll_file = enroll_files[0]
                output_lines.append(f"{enroll_file},{test_file},{speaker_id},{test_utt_id}")
    
    # Write output CSV
    with open(args.output_file, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')
    
    print(f"\nGenerated test protocol CSV: {args.output_file}")
    print(f"Total trials: {len(output_lines)}")
    if missing_speakers:
        print(f"Warning: {len(missing_speakers)} speakers not found in enrollment metadata")
    print(f"\nSample entries:")
    for line in output_lines[:3]:
        print(f"  {line}")

if __name__ == '__main__':
    main()

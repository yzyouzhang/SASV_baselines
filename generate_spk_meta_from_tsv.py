#!/usr/bin/env python3
"""
Generate speaker metadata pickle file from TSV enrollment protocol.
This creates a pickle file mapping speaker IDs to their enrollment utterances.
"""
import argparse
import pickle
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='Generate speaker metadata from TSV enrollment protocol')
    parser.add_argument('--enroll_file', type=str, required=True,
                       help='Enrollment protocol TSV file (format: speaker_id enroll_file1,enroll_file2,...)')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output pickle file (e.g., spk_meta/spk_meta_test.pk)')
    args = parser.parse_args()
    
    # Parse enrollment protocol
    spk_meta = {}
    
    with open(args.enroll_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) != 2:
                print(f"Warning: Skipping malformed line: {line}")
                continue
            
            speaker_id = parts[0]
            enroll_files = parts[1].split(',')
            
            # Add .flac extension if not present
            enroll_files = [f if f.endswith('.flac') else f + '.flac' for f in enroll_files]
            
            spk_meta[speaker_id] = enroll_files
    
    # Save as pickle file
    with open(args.output_file, 'wb') as f:
        pickle.dump(spk_meta, f)
    
    print(f"Generated speaker metadata file: {args.output_file}")
    print(f"Total speakers: {len(spk_meta)}")
    print(f"Sample entries:")
    for i, (spk_id, files) in enumerate(list(spk_meta.items())[:3]):
        print(f"  {spk_id}: {len(files)} enrollment files - {files[:2]}...")
        if i >= 2:
            break

if __name__ == '__main__':
    main()

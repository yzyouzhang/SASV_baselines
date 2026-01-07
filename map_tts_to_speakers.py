#!/usr/bin/env python3
"""
Map TTS evaluation files to their target speaker IDs using the enrollment protocol.

This script:
1. Reads the TTS protocol (protocol_tts_posteval_v2.0.tsv) to get speaker_id -> filename mapping
2. Reads the enrollment protocol to get speaker_id -> enrollment_files mapping
3. Creates a TSV file mapping each TTS file to its target speaker for scoring
"""
import argparse
import os
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='Map TTS files to target speakers')
    parser.add_argument('--tts_protocol', type=str, required=True,
                       help='TTS protocol TSV file (format: speaker_id tts_filename)')
    parser.add_argument('--enroll_protocol', type=str, required=True,
                       help='Enrollment protocol TSV file (format: speaker_id enroll_file1,enroll_file2,...)')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output TSV file (format: speaker_id tts_file)')
    args = parser.parse_args()
    
    # Step 1: Read TTS protocol to get speaker_id -> TTS files mapping
    print("Reading TTS protocol...")
    tts_trials = []  # [(speaker_id, tts_filename), ...]
    
    with open(args.tts_protocol, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) != 2:
                print(f"Warning: Skipping malformed line: {line}")
                continue
            
            speaker_id = parts[0]
            tts_filename = parts[1]
            tts_trials.append((speaker_id, tts_filename))
    
    print(f"Loaded {len(tts_trials)} TTS trials")
    
    # Step 2: Read enrollment protocol to verify speakers exist
    print("Reading enrollment protocol...")
    enrolled_speakers = set()
    
    with open(args.enroll_protocol, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) != 2:
                continue
            
            speaker_id = parts[0]
            enrolled_speakers.add(speaker_id)
    
    print(f"Found {len(enrolled_speakers)} enrolled speakers")
    
    # Step 3: Create output TSV mapping TTS files to speakers
    print("Creating output TSV...")
    output_lines = []
    missing_speakers = set()
    
    for speaker_id, tts_filename in tts_trials:
        if speaker_id not in enrolled_speakers:
            if speaker_id not in missing_speakers:
                print(f"Warning: Speaker {speaker_id} not found in enrollment protocol")
                missing_speakers.add(speaker_id)
            continue
        
        # Add .flac extension if not present
        if not tts_filename.endswith('.flac'):
            tts_filename = tts_filename + '.flac'
        
        output_lines.append(f"{speaker_id}\t{tts_filename}")
    
    # Step 4: Write output file
    with open(args.output_file, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')
    
    print(f"\nGenerated TTS test protocol: {args.output_file}")
    print(f"Total trials: {len(output_lines)}")
    if missing_speakers:
        print(f"Warning: {len(missing_speakers)} speakers not found in enrollment")
    print(f"\nSample entries:")
    for line in output_lines[:5]:
        print(f"  {line}")
    
    # Statistics by unique speakers
    unique_speakers = set([line.split('\t')[0] for line in output_lines])
    print(f"\nUnique target speakers: {len(unique_speakers)}")

if __name__ == '__main__':
    main()

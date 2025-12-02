# Revised Scoring Workflow for Test Set

## Summary of Changes

The `--scoring` mode now handles TSV protocol files directly, without requiring CSV conversion. This mirrors the `--eval` mode's behavior but with TSV input format.

## Key Understanding: How `--eval` Works

In `--eval` mode:
```
CSV: enroll_file,test_file,label
```
- Each row is a trial comparing two individual files
- Extract embedding from enroll_file
- Extract embedding from test_file  
- Compute score (cosine similarity or SAMO)
- Use label for metric calculation

## How `--scoring` Now Works with TSV

In `--scoring` mode with TSV files:
```
Enrollment TSV: speaker_id enroll_file1,enroll_file2,...
Test TSV: speaker_id test_file
```

The system:
1. Builds **speaker-level enrollment embeddings** by averaging all enrollment files per speaker
2. For each test trial, compares the speaker's enrollment embedding with the test file embedding
3. Computes score (cosine similarity or SAMO)
4. Returns scores without labels (no metrics)

## Key Difference: File-level vs Speaker-level

### `--eval` (CSV):
- **File-level comparison**: Each enrollment file is compared individually with test files
- Multiple rows can exist for the same test file (one per enrollment file)
- Scores are averaged if multiple enrollments exist for same test utterance

### `--scoring` (TSV):
- **Speaker-level comparison**: All enrollment files per speaker are averaged first
- One comparison per (speaker, test_file) pair
- The speaker's averaged enrollment embedding is compared with the test file

Both achieve the same goal but `--scoring` with TSV is more efficient and cleaner.

## Complete Test Scoring Workflow

No CSV generation needed! Use TSV files directly:

### Cosine Similarity Scoring
```bash
CUDA_VISIBLE_DEVICES=0 python trainSASVNet.py \
  --scoring \
  --initial_model exp/sasv_redimnet/model/model000000008.model \
  --enroll_tsv /work/hdd/bfdc/yzyouzhang/spoofceleb_test/protocol_enroll.tsv \
  --test_tsv /work/hdd/bfdc/yzyouzhang/spoofceleb_test/protocol_test.tsv \
  --eval_path /work/hdd/bfdc/yzyouzhang/spoofceleb_test/data_v2.0 \
  --save_path exp/sasv_redimnet_test_final \
  --model ReDimNet \
  --redimnet_model b2 \
  --num_class 1160 \
  --batch_size 1 \
  --num_thread 8
```

### SAMO Scoring
```bash
CUDA_VISIBLE_DEVICES=0 python trainSASVNet.py \
  --scoring \
  --initial_model exp/sasv_samo_finetune2/model/model000000004.model \
  --enroll_tsv /work/hdd/bfdc/yzyouzhang/spoofceleb_test/protocol_enroll.tsv \
  --test_tsv /work/hdd/bfdc/yzyouzhang/spoofceleb_test/protocol_test.tsv \
  --eval_path /work/hdd/bfdc/yzyouzhang/spoofceleb_test/data_v2.0 \
  --save_path exp/sasv_samo_test_final2_finetune \
  --model ReDimNet \
  --redimnet_model b2 \
  --trainfunc samo_sasv \
  --num_class 1160 \
  --batch_size 1 \
  --num_thread 8 \
  --use_samo_scoring
```

## How It Works Internally

When you run `--scoring` with TSV files:

1. **Load enrollment TSV**: Parse speaker → [enrollment_files] mapping
2. **Load test TSV**: Parse (speaker, test_file) pairs
3. **Extract all embeddings**: Get embeddings for all enrollment files and test files
4. **Build speaker embeddings**: For each speaker, average their enrollment file embeddings
5. **Compute scores**: For each test trial, compare speaker's enrollment embedding with test file embedding
6. **Return results**: Output format is `speaker_id TAB utt_id TAB score`

This is more efficient than `--eval`'s approach because:
- Enrollment files are processed once and averaged per speaker
- No redundant comparisons between individual enrollment files and test files
- Direct speaker-to-test comparison matches the actual evaluation scenario

## Code Changes

### New Arguments in `trainSASVNet.py`:
- `--enroll_tsv`: Path to enrollment TSV file
- `--test_tsv`: Path to test TSV file

### New Method in `SASVNet.py`:
- `evaluateFromTSV()`: Handles TSV-based evaluation
  - Builds speaker-level enrollment embeddings
  - Compares with test utterances
  - Supports both cosine similarity and SAMO scoring

## Output Format

**Scoring output** (`exp/*/result/1_scorefile`):
```
SPK_00910559_042	UTT_00780092	0.8234
SPK_00925455_252	UTT_00248047	-0.1234
...
```

Format: `speaker_id TAB utt_id TAB score`

## Notes

1. **TSV format is native**: No CSV conversion needed - the code reads TSV files directly

2. **Speaker-level enrollment**: All enrollment files per speaker are automatically averaged before comparison

3. **SAMO scoring**: Use `--use_samo_scoring` flag to use SAMO inference instead of cosine similarity. Note: `--use_enroll` flag is NOT needed with TSV mode since enrollment averaging is built-in.

4. **Backward compatibility**: The old CSV-based workflow still works if you provide `--eval_list` instead of `--enroll_tsv` and `--test_tsv`

#!/bin/bash
set -exuo pipefail
LANGS=('ar' 'bn' 'id' 'fi' 'ko' 'ru' 'sw' 'te' 'th' 'ja')

DATA_DIR="datasets"
OUTPUT_DIR="parallel-sentences"

mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

DEV_SENTENCES=1000

for lang in "${LANGS[@]}"; do
    # Download the data
    opus_read -d CCAligned -s en -t "$lang" -p raw -wm moses -w "ccaligned-en-$lang.tsv" -dl "../$DATA_DIR" -q
    # Put first DEV_SENTENCES into dev set file and gzip
    head -n "$DEV_SENTENCES" "ccaligned-en-$lang.tsv" > "ccaligned-en-$lang-dev.tsv"
    gzip "ccaligned-en-$lang-dev.tsv"
    # Put the rest into train set file and gzip
    tail -n +$((DEV_SENTENCES + 1)) "ccaligned-en-$lang.tsv" > "ccaligned-en-$lang-train.tsv"
    gzip "ccaligned-en-$lang-train.tsv"
    # Remove the original file
    rm "ccaligned-en-$lang.tsv"
done

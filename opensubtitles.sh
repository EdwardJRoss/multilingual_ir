#!/bin/bash
set -exuo pipefail
LANGS=('ar' 'bn' 'id' 'fi' 'ko' 'ru' 'sw' 'te' 'th' 'ja')
#LANGS=('ja')

DATA_DIR="datasets"
OUTPUT_DIR="parallel-sentences"

mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

DEV_SENTENCES=1000

for lang in "${LANGS[@]}"; do
    # Download the data
    opus_read -d OpenSubtitles -s en -t "$lang" -p raw -wm moses -w "opensubtitles-en-$lang.tsv" -dl "../$DATA_DIR" -q
    # Put first DEV_SENTENCES into dev set file and gzip
    head -n "$DEV_SENTENCES" "opensubtitles-en-$lang.tsv" > "opensubtitles-en-$lang-dev.tsv"
    gzip "opensubtitles-en-$lang-dev.tsv"
    # Put the rest into train set file and gzip
    tail -n +$((DEV_SENTENCES + 1)) "opensubtitles-en-$lang.tsv" > "opensubtitles-en-$lang-train.tsv"
    gzip "opensubtitles-en-$lang-train.tsv"
    # Remove the original file
    rm "opensubtitles-en-$lang.tsv"
done

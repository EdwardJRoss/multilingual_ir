#!/bin/bash
set -exuo pipefail
#LANGS=('ar' 'bn' 'id' 'fi' 'ko' 'ru' 'swh' 'te' 'th' 'ja')
LANGS=('swh' 'te' 'th' 'ja')

DATA_DIR="datasets"
OUTPUT_DIR="parallel-sentences"

mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

DEV_SENTENCES=1000

for lang in "${LANGS[@]}"; do
    # Download the data
    opus_read -d Tatoeba -s en -t "$lang" -p raw -wm moses -w "tatoeba-en-$lang.tsv" -dl "../$DATA_DIR" -q
    # Put first DEV_SENTENCES into dev set file and gzip
    head -n "$DEV_SENTENCES" "tatoeba-en-$lang.tsv" > "tatoeba-en-$lang-dev.tsv"
    gzip "tatoeba-en-$lang-dev.tsv"
    # Put the rest into train set file and gzip
    tail -n +$((DEV_SENTENCES + 1)) "tatoeba-en-$lang.tsv" > "tatoeba-en-$lang-train.tsv"
    gzip "tatoeba-en-$lang-train.tsv"
    # Remove the original file
    rm "tatoeba-en-$lang.tsv"
done

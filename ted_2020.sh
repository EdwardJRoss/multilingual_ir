#!/bin/bash
set -exuo pipefail
LANGS=('de' 'es' 'it' 'fr' 'ar' 'tr')

DATA_DIR="datasets"
OUTPUT_DIR="parallel-sentences"

mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

DEV_SENTENCES=1000

for lang in "${LANGS[@]}"; do
    # Download the data
    opus_read -d TED2020 -s en -t "$lang" -p raw -wm moses -w "TED2020-en-$lang.tsv" -dl "../$DATA_DIR" -q
    # Put first DEV_SENTENCES into dev set file and gzip
    head -n "$DEV_SENTENCES" "TED2020-en-$lang.tsv" > "TED2020-en-$lang-dev.tsv"
    gzip "TED2020-en-$lang-dev.tsv"
    # Put the rest into train set file and gzip
    tail -n +$((DEV_SENTENCES + 1)) "TED2020-en-$lang.tsv" > "TED2020-en-$lang-train.tsv"
    gzip "TED2020-en-$lang-train.tsv"
    # Remove the original file
    rm "TED2020-en-$lang.tsv"
done

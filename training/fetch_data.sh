#!/bin/sh
# Downloads the small parquet files train_models.py trains on.
set -e
cd "$(dirname "$0")"

curl -sL -o en_train.parquet "https://huggingface.co/datasets/cardiffnlp/tweet_eval/resolve/refs%2Fconvert%2Fparquet/hate/train/0000.parquet"
curl -sL -o en_val.parquet "https://huggingface.co/datasets/cardiffnlp/tweet_eval/resolve/refs%2Fconvert%2Fparquet/hate/validation/0000.parquet"
curl -sL -o de_train.parquet "https://huggingface.co/datasets/philschmid/germeval18/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
curl -sL -o de_test.parquet "https://huggingface.co/datasets/philschmid/germeval18/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"

echo "Downloaded to $(pwd)"

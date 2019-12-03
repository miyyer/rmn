#!/bin/bash

# Create directory for data and models
mkdir -p data
mkdir -p models
mkdir -p figs

# Download data, metadata, and word embeddings
curl -OL http://cs.umd.edu/~miyyer/data/relationships.csv.gz
curl -OL http://cs.umd.edu/~miyyer/data/metadata.pkl
curl -OL http://cs.umd.edu/~miyyer/data/glove.We
mv relationships.csv.gz data/
mv metadata.pkl data/
mv glove.We data/

# train rmn
python train_rmn.py

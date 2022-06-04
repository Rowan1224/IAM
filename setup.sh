#!/bin/bash

cd post/neuspell
mkdir data

cd data
mkdir models

cd models
mkdir bert-base-cased

cd bert-base-cased 

wget https://zenodo.org/record/6613955/files/pytorch_model.bin
wget https://zenodo.org/record/6613955/files/vocab.pkl
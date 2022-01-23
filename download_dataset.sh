#!/bin/bash
# Downloads a test dataset. Assumes that wget an unzip are available on PATH
set -euo pipefail
mkdir -p data
cd data
wget https://cvg.ethz.ch/research/symmetries-in-sfm/datasets/barcelona.zip
unzip barcelona.zip -d barcelona
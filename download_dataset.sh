#!/bin/bash
# Downloads a test dataset. Assumes that wget an unzip are available on PATH
set -euo pipefail
wget https://cvg.ethz.ch/research/symmetries-in-sfm/datasets/barcelona.zip -o data/
unzip barcelona.zip -d barcelona
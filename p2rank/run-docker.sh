#! /bin/bash

INPUT_PATH="$1"
OUTPUT_PATH="$2"
CURRENT_DIRECTORY=$(pwd)

if [ ! -d "$INPUT_PATH" ] ; then
    echo "Input path $INPUT_PATH does not exist or is not a directory"
    exit 1
fi

# create input.ds file for P2Rank: collect all .cif and .pdb files in the input directory
cd $INPUT_PATH
ls *.cif > input.ds
ls *.pdb >> input.ds
cd $CURRENT_DIRECTORY

# build docker image (skip if exists)
if ! sudo docker image inspect p2rank &> /dev/null; then
  sudo docker build -t p2rank -f ./Dockerfile .
fi

# run p2rank inside docker
sudo docker run -i -v "$INPUT_PATH":/opt/p2rank/input -v "$OUTPUT_PATH":/opt/p2rank/output -t p2rank ./prank predict -o output/ -c alphafold input/input.ds -visualizations 0

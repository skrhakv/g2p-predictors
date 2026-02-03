#! /bin/bash

INPUT_PATH="$1"
OUTPUT_PATH="$2"
CURRENT_DIRECTORY=$(pwd)

if [ ! -d "$INPUT_PATH" ] ; then
    echo "Input path $INPUT_PATH does not exist or is not a directory"
    exit 1
fi

# build docker image (skip if exists)
if ! sudo docker image inspect af2bind &> /dev/null; then
  sudo docker build -t af2bind -f ./Dockerfile .
fi

# run af2bind inside docker
sudo docker run -i --gpus all -v "$INPUT_PATH":/opt/af2bind/input -v "$OUTPUT_PATH":/opt/af2bind/output -t af2bind /usr/local/bin/python3.12 run.py --input_path input --output_path output
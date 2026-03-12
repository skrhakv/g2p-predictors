#! /bin/bash

INPUT_PATH="$1"
OUTPUT_PATH="$2"
CURRENT_DIRECTORY=$(pwd)

if [ ! -d "$INPUT_PATH" ] ; then
    echo "Input path $INPUT_PATH does not exist or is not a directory"
    exit 1
fi

sudo docker build -t af2bind -f ./Dockerfile .

# run af2bind inside docker
sudo docker run -v "$INPUT_PATH":/opt/af2bind/input \
                -v "$OUTPUT_PATH":/opt/af2bind/output \
                af2bind /usr/local/bin/python3.12 run.py \
                --prediction_path analysis_files/af2bind_p2rank_human_proteome_predictions_revision_with_sitemap.csv \
                --pdb_files_path input \
                --output_path output
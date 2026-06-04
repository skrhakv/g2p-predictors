#! /bin/bash

# Defaults
CONSERVATION=true
UNIREF_PATH="${HMM_SEQUENCE_FILE:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-conservation)
            CONSERVATION=false
            shift
            ;;
        --uniref)
            UNIREF_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

INPUT_PATH="$1"
OUTPUT_PATH="$2"
CURRENT_DIRECTORY=$(pwd)

if [ ! -d "$INPUT_PATH" ]; then
    echo "Input path $INPUT_PATH does not exist or is not a directory"
    exit 1
fi

if [ "$CONSERVATION" = "true" ]; then
    if [ -z "$UNIREF_PATH" ]; then
        echo "ERROR: conservation is enabled but no UniRef50 database path was given."
        echo "  Pass --uniref /path/to/uniref50.fasta or set HMM_SEQUENCE_FILE env var."
        echo "  Use --no-conservation to skip conservation."
        exit 1
    fi
    if [ ! -f "$UNIREF_PATH" ]; then
        echo "ERROR: UniRef50 database not found: $UNIREF_PATH"
        exit 1
    fi
fi

cleanup() {
    if [ "$CONSERVATION" = "true" ]; then
        echo "Stopping conservation server..."
        sudo docker rm -f p2rank-conservation-server 2>/dev/null || true
        sudo docker network rm p2rank-net 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Create input.ds for P2Rank (list of structure files)
cd "$INPUT_PATH"
{ ls *.cif 2>/dev/null; ls *.pdb 2>/dev/null; } > input.ds
cd "$CURRENT_DIRECTORY"

# Build p2rank image if not present
if ! sudo docker image inspect p2rank &>/dev/null; then
    sudo docker build -t p2rank -f ./Dockerfile .
fi

if [ "$CONSERVATION" = "true" ]; then
    # Build conservation server image if not present
    if ! sudo docker image inspect p2rank-conservation &>/dev/null; then
        sudo docker build -t p2rank-conservation -f ./Dockerfile.conservation .
    fi

    # Create inter-container network
    sudo docker network create p2rank-net 2>/dev/null || true

    # Remove any leftover container from a previous run
    sudo docker rm -f p2rank-conservation-server 2>/dev/null || true

    # Start conservation server with uniref50 database mounted
    sudo docker run -d \
        --name p2rank-conservation-server \
        --network p2rank-net \
        -v "$UNIREF_PATH":/data/uniref50.fasta:ro \
        p2rank-conservation

    # Wait until the HTTP server is accepting requests
    echo "Waiting for conservation server to start..."
    TRIES=0
    until sudo docker exec p2rank-conservation-server \
        python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8030/health')" 2>/dev/null
    do
        TRIES=$((TRIES + 1))
        if [ "$TRIES" -ge 30 ]; then
            echo "ERROR: conservation server did not become ready in time."
            exit 1
        fi
        sleep 1
    done
    echo "Conservation server ready."

    # Run p2rank, calling the conservation server for each structure
    sudo docker run \
        --network p2rank-net \
        -v "$INPUT_PATH":/opt/p2rank/input \
        -v "$OUTPUT_PATH":/opt/p2rank/output \
        p2rank \
        ./prank predict \
            -o output/ \
            -c alphafold_conservation_hmm \
            -conservation_provider hmm_server \
            -conservation_provider_url http://p2rank-conservation-server:8030 \
            input/input.ds \
            -visualizations 0
else
    sudo docker run \
        -v "$INPUT_PATH":/opt/p2rank/input \
        -v "$OUTPUT_PATH":/opt/p2rank/output \
        p2rank \
        ./prank predict -o output/ -c alphafold input/input.ds -visualizations 0
fi

# Post-processing: reformat output and clean up intermediate files
python3 ./post-processing.py --prediction_path "$OUTPUT_PATH" --pdb_files_path "$INPUT_PATH"
sudo find "$OUTPUT_PATH" -name "*_predictions.csv" -delete
sudo find "$OUTPUT_PATH" -name "*_residues.csv" -delete
sudo find "$OUTPUT_PATH" -name "params.txt" -delete
sudo find "$OUTPUT_PATH" -name "run.log" -delete

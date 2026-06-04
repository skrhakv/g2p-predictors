# LBS predictors for G2P portal
This repository summarizes the steps needed for running P2Rank and AF2Bind binding site predictors in Docker. The output of these tools is subsequently used by the G2P portal. While P2Rank accepts both *mmCIF* and *PDB*, AF2Bind accepts only the *PDB* file format.

## P2Rank
First you need to install the biotite package which is used in the post-processing phase:
```
python3 -m pip install biotite
```

Deployment of docker with P2Rank is summarized in `p2rank/run-docker.sh` script. Three parameters need to be provided: 
```
sudo bash run-docker.sh --uniref <FULL-UNIREF-PATH> <FULL-INPUT-PATH> <FULL-OUTPUT-PATH>
```
where `<FULL-INPUT-PATH>` contains all `*.cif` and `.pdb` files for the prediction and `<FULL-UNIREF-PATH>` is a FASTA file containing the UniRef50 database; see [the p2rank docs](https://github.com/rdk/prankweb/tree/conservation-server/executor-p2rank/conservation) for the download link, or try:
```
wget https://ftp.expasy.org/databases/uniprot/current_release/uniref/uniref50/uniref50.fasta.gz && gunzip uniref50.fasta.gz
```

Example input and output folders can be found at `p2rank/input` and `p2rank/output`. 

### Run for G2P structures - example
```
sudo bash run-docker.sh /media/drive1/g2p_data/shared_data/alphafold_v6/pdb /media/drive1/g2p_data/data_releases/2025_10/processed_data/p2rank_features
```

## AF2Bind
Similarly, the `AF2Bind/run-docker.sh` script can be used to extract the predicted binding sites for human proteome from [this Zenodo repository](https://zenodo.org/records/17683380):
```
sudo bash run-docker.sh <FULL-INPUT-PATH> <FULL-OUTPUT-PATH>
```
where `<FULL-INPUT-PATH>` contains all `.pdb` files for the prediction. 


## Future work
1. large portion of the sequences is in disordered regions. It would it make sense to implement some kind of filter that would exclude results from regions with low pLDDT.
2. For now, the setup is accepting the AF-sourced structure models. Expand the framework to accept also the PDB-sourced structures.
  
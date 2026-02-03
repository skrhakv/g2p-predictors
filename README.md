# LBS predictors for G2P portal
This repository summarizes the steps needed for running P2Rank and AF2Bind binding site predictors in Docker. The output of these tools is subsequently used by the G2P portal. While P2Rank accepts both *mmCIF* and *PDB*, AF2Bind accepts only the *PDB* file format.

## P2Rank
Deployment of docker with P2Rank is done summarized in `p2rank/run-docker.sh` script. Two parameters need to be provided: 
```
sudo bash run-docker.sh <FULL-INPUT-PATH> <FULL-OUTPUT-PATH>
```
where `<FULL-INPUT-PATH>` contains all `*.cif` and `.pdb` files for the prediction. 

Example input and output folders can be found at `p2rank/input` and `p2rank/output`. 

## AF2Bind
Similarly, the `AF2Bind/run-docker.sh` script can be used to run the AF2Bind predictor:
```
sudo bash run-docker.sh <FULL-INPUT-PATH> <FULL-OUTPUT-PATH>
```
where `<FULL-INPUT-PATH>` contains all `.pdb` files for the prediction. 

## Configuration
Currently, both P2Rank and AF2bind are configured for AlphaFold-predicted structures. For predicting on the experimental PDB structures, the following steps must be conducted:
1. the `-c alphafold` parameter must be removed when running P2Rank (see `p2rank/run-docker.sh`).
2. the correct chain ID needs to be selected in the `AF2Bind/src/run.py` script.


## Future work
1. large portion of the sequences is in disordered regions. It would it make sense to implement some kind of filter that would exclude results from regions with low pLDDT.
2. For now, the setup is accepting the AF-sourced structure models. Expand the framework to accept also the PDB-sourced structures.
  
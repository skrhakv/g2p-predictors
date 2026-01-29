# LBS predictors for G2P portal
## P2Rank
Deployment of docker with P2Rank is done summarized in `p2rank/run-docker.sh` script. Two parameters need to be provided: 
```
sudo bash run-docker.sh <FULL-INPUT-PATH> <FULL-OUTPUT-PATH>
```
where `<FULL-INPUT-PATH>` contains all `*.cif` and `.pdb` files for the prediction. 

Example input and output folders can be found at `p2rank/input` and `p2rank/output`. 

### Configuration
Currently, P2Rank is configured for AlphaFold-predicted structures. For predicting on the experimental PDB structures, contact me to change the configuration.

## AF2Bind

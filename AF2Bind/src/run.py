import os
import numpy as np
import pandas as pd
from biotite.structure.io.pdb import PDBFile, get_structure
from biotite.structure import filter_peptide_backbone

class AF2BindPocketResidue:
    def __init__(self, residue_number, amino_acid, average_pLDDT, p_bind, pocket_number):
        self.residue_number = residue_number
        self.amino_acid = amino_acid
        self.p_bind = p_bind
        self.average_pLDDT = average_pLDDT
        self.pocket_number = pocket_number

def parse_pbind(pbind_str):
    pbind = pbind_str[1:-1] # remove the square brackets
    pbind_values = [float(x.strip()) for x in pbind.split(',')]
    return pbind_values

def parse_resnums(resnums_str):
    resnums_str = resnums_str.split('+')
    return np.array([int(i) for i in resnums_str])

def read_pLDDTs(pdb_file_path):
    pdb_file = PDBFile.read(pdb_file_path)

    try:
        protein = get_structure(pdb_file, model=1, extra_fields=["b_factor"])
    except Exception as _: # possibly the file is not a valid PDB file (the download failed or the file is empty) - typical content of the file is "<?xml version='1.0' encoding='UTF-8'?><Error><Code>NoSuchKey</Code><Message>The specified key does not exist.</Message></Error>"
        return [], []
    
    protein = protein[filter_peptide_backbone(protein) & 
                           (protein.atom_name == "CA") &
                           (protein.element == "C") ]
    pLDDTs = np.array([i.b_factor for i in protein])        
    sequence = [i.res_name for i in protein]
    return pLDDTs, sequence

def main(prediction_path, pdb_files_path, output_path):
    # NOTE: 'resnum' is one-based indexing (1=first residue)
    binding_pockets_df = pd.read_csv(prediction_path, sep=",", header=0)
    # filter the CSV before processing sequentially
    binding_pockets_df = binding_pockets_df[["uniprot", "AF2BIND_cluster", "AF2BIND_cluster_resnums", "AF2BIND_pbind"]]
    binding_pockets_df = binding_pockets_df.dropna(how='all', subset=["AF2BIND_cluster", "AF2BIND_cluster_resnums", "AF2BIND_pbind"]) # if all values in the row are NaN, drop the row
    grouped = binding_pockets_df.groupby(['uniprot'])

    
    for uniprot, group in grouped:
        uniprot = uniprot[0]
        print(f"Processing uniprot {uniprot}...")
        AF_filepath = os.path.join(pdb_files_path, f'AF-{uniprot}-F1-model_v6.pdb')
        if not os.path.exists(AF_filepath):
            continue
        pLDDTs, sequence = read_pLDDTs(AF_filepath)
        if len(pLDDTs) == 0 or len(sequence) == 0:
            print(f"Warning: Could not read pLDDTs or sequence for uniprot {uniprot}. Skipping this uniprot.")
            continue
        valid_predictions = True
        binding_residues = {}
        for _, row in group.iterrows():
            pocket_number = int(row['AF2BIND_cluster'])
            p_bind = parse_pbind(row['AF2BIND_pbind'])
            pocket_residue_numbers = parse_resnums(row['AF2BIND_cluster_resnums'])
            
            # check if prediction residue numbering isn't longer than our AF-predicted PDB structure
            if max(pocket_residue_numbers) > len(sequence):
                print(f"Warning: For uniprot {uniprot}, pocket {pocket_number}, the maximum residue number in the predicted pocket ({max(pocket_residue_numbers)}) exceeds the length of the sequence ({len(sequence)}). Skipping this pocket.")
                valid_predictions = False
                break
            
            # pocket-wise average pLDDT
            average_pLDDT = np.mean(pLDDTs[pocket_residue_numbers - 1]) # -1 because of one-based indexing
            
            # save each pocket residue
            for (this_residue_number, this_pbind) in zip(pocket_residue_numbers, p_bind):
                binding_residues[this_residue_number] = AF2BindPocketResidue(
                    residue_number=this_residue_number,
                    amino_acid=sequence[this_residue_number - 1], # -1 because of one-based indexing
                    average_pLDDT=average_pLDDT, 
                    p_bind=this_pbind, 
                    pocket_number=pocket_number)
        
        if not valid_predictions:
            print(f"Skipping uniprot {uniprot} due to invalid predictions.")
            continue
        
        # loop over all residues in the sequence
        with open(f'{output_path}/{uniprot}.csv', 'w') as f:
            f.write("residue_number\tamino_acid\tchain\tpocket_number\tp_bind\taverage_pLDDT\n")
            for residue_number in range(1, len(sequence) + 1): # one-based indexing
                amino_acid = sequence[residue_number - 1] # -1 because of one-based indexing
                if residue_number in binding_residues.keys():
                    f.write(f"{residue_number}\t{amino_acid}\tA\t{binding_residues[residue_number].pocket_number}\t{binding_residues[residue_number].p_bind:.2f}\t{binding_residues[residue_number].average_pLDDT:.2f}\n")
                else:
                    f.write(f"{residue_number}\t{amino_acid}\tA\t\t\t\n")
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_path', type=str, required=True)
    parser.add_argument('--pdb_files_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    main(args.prediction_path, args.pdb_files_path, args.output_path)

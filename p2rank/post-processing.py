import argparse
import os
import csv
from statistics import mean
import numpy as np
from biotite.structure.io.pdb import PDBFile, get_structure
from biotite.structure import filter_peptide_backbone

def read_pLDDTs(pdb_file_path):
    pdb_file = PDBFile.read(pdb_file_path)

    try:
        protein = get_structure(pdb_file, model=1, extra_fields=["b_factor"])
    except Exception as _: # possibly the file is not a valid PDB file (the download failed or the file is empty) - typical content of the file is "<?xml version='1.0' encoding='UTF-8'?><Error><Code>NoSuchKey</Code><Message>The specified key does not exist.</Message></Error>"
        return []
    
    protein = protein[filter_peptide_backbone(protein) & 
                           (protein.atom_name == "CA") &
                           (protein.element == "C") ]
    pLDDTs = [i.b_factor for i in protein]        
    return pLDDTs
    
def main(prediction_path, pdb_files_path):
    print(f"Reformating pocket predictions in {prediction_path} and writing to txt files...")
    for file in os.listdir(prediction_path):
        if not file.endswith("_residues.csv"):
            continue
        
        # load pocket info
        pocket_info = {}
        with open(os.path.join(prediction_path, file.replace('_residues.csv', '_predictions.csv')), 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            next(csv_reader)  # skip header
            for i, row in enumerate(csv_reader):
                rank = row[1].strip()
                probability = row[3].strip()
                binding_residues = [int(i.split('_')[1]) for i in row[9].strip().split(' ')]
                pocket_info[rank] = {
                    "probability": float(probability),
                    "binding_residues": binding_residues
                }

        pLDDTs = read_pLDDTs(os.path.join(pdb_files_path, file.replace("_residues.csv", "")))
        
        if len(pLDDTs) != 0:
            for pocket_rank in pocket_info.keys():
                this_pLDDTs = [pLDDTs[res_id - 1] for res_id in pocket_info[pocket_rank]['binding_residues']]
                pocket_info[pocket_rank]['pLDDTs'] = mean(this_pLDDTs)
        # if the pLDDT loading failed, we set the pLDDT value to "N/A"
        else:
            for pocket_rank in pocket_info.keys():
                pocket_info[pocket_rank]['pLDDTs'] = 'N/A'

        with open(os.path.join(prediction_path, file), 'r') as f:
            with open(os.path.join(prediction_path, f'{file.replace(".pdb_residues.csv", ".txt")}'), 'w') as f_pred:
                # write header
                f_pred.write("residue_number\tamino_acid\tchain\tpocket_rank\tpocket_probability\tmean_pLDDT\n")

                csv_reader = csv.reader(f, delimiter=',')
                next(csv_reader)  # skip header
                for row in csv_reader:
                    chain = row[0].strip()
                    residue_number = row[1].strip()
                    amino_acid = row[2].strip()
                    pocket_number = row[6].strip()
                    # print(f'"{pocket_number}"', pocket_info.keys())
                    
                    if pocket_number in pocket_info:
                        f_pred.write(f"{residue_number}\t{amino_acid}\t{chain}\t{pocket_number}\t{pocket_info[pocket_number]['probability']:.2f}\t{pocket_info[pocket_number]['pLDDTs']:.2f}\n")
                    else:
                        f_pred.write(f"{residue_number}\t{amino_acid}\t{chain}\t\t\t\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_path', type=str, required=True)
    parser.add_argument('--pdb_files_path', type=str, required=True)
    args = parser.parse_args()
    main(args.prediction_path, args.pdb_files_path)
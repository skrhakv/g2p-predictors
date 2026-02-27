import argparse
import os
import csv

def main(path):
    for file in os.listdir(path):
        if not file.endswith("_residues.csv"):
            continue
        
        # load pocket info
        pocket_info = {}
        with open(os.path.join(path, file.replace('_residues.csv', '_predictions.csv')), 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            next(csv_reader)  # skip header
            for row in csv_reader:
                rank = row[1].strip()
                score = row[2].strip()
                probability = row[3].strip()
                sas_points = row[4].strip()
                surf_atoms = row[5].strip()
                pocket_info[rank] = {
                    "score": score,
                    "probability": probability,
                    "sas_points": sas_points,
                    "surf_atoms": surf_atoms
                }

        with open(os.path.join(path, file), 'r') as f:
            with open(os.path.join(path, f'{file.replace(".pdb_residues.csv", ".txt")}'), 'w') as f_pred:
                # write header
                f_pred.write("residue_number\tamino_acid\tchain\tpocket_rank\tpocket_score\tpocket_probability\tpocket_sas_points_count\tpocket_surf_atoms_count\n")

                csv_reader = csv.reader(f, delimiter=',')
                next(csv_reader)  # skip header
                for row in csv_reader:
                    chain = row[0].strip()
                    residue_number = row[1].strip()
                    amino_acid = row[2].strip()
                    pocket_number = row[6].strip()
                    # print(f'"{pocket_number}"', pocket_info.keys())
                    
                    if pocket_number in pocket_info:
                        f_pred.write(f"{residue_number}\t{amino_acid}\t{chain}\t{pocket_number}\t{pocket_info[pocket_number]['score']}\t{pocket_info[pocket_number]['probability']}\t{pocket_info[pocket_number]['sas_points']}\t{pocket_info[pocket_number]['surf_atoms']}\n")
                    else:
                        f_pred.write(f"{residue_number}\t{amino_acid}\t{chain}\t\t\t\t\t\n")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()
    main(args.path)
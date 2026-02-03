import argparse
import os
import csv
import numpy as np

def main(path):
    for file in os.listdir(path):
        if not file.endswith("_residues.csv"):
            continue
        with open(os.path.join(path, file), 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            next(csv_reader)  # skip header
            predicted_pockets = []
            for row in csv_reader:
                pocket_number = row[6]
                predicted_pockets.append(int(pocket_number))
        predicted_pockets = np.array(predicted_pockets)
        np.save(os.path.join(path, f"{'.'.join(file.split('.')[:-2])}.npy"), predicted_pockets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()
    main(args.path)
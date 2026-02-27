###### This code is adapted from the following repository: github.com/sokrypton/af2bind (see LICENSE file) ######

import numpy as np
from colabdesign import mk_afdesign_model
import jax, pickle
from scipy.special import expit as sigmoid
import gc

MASK_SIDECHAINS = True
MASK_SEQUENCE = False
MAX_RESIDUES_COUNT = 2190 # this is the maximum number of residues that can be processed without OOM error (experimentally determined)

def af2bind(outputs, mask_sidechains=True, seed=0):
    pair_A = outputs["representations"]["pair"][:-20,-20:]
    pair_B = outputs["representations"]["pair"][-20:,:-20].swapaxes(0,1)
    pair_A = pair_A.reshape(pair_A.shape[0],-1)
    pair_B = pair_B.reshape(pair_B.shape[0],-1)
    x = np.concatenate([pair_A,pair_B],-1)
   
    # get params
    if mask_sidechains:
        model_type = f"split_nosc_pair_A_split_nosc_pair_B_{seed}"
    else:
        model_type = f"split_pair_A_split_pair_B_{seed}"
    with open(f"/opt/af2bind/af2bind_params/attempt_7_2k_lam0-03/{model_type}.pickle","rb") as handle:
        params_ = pickle.load(handle)
    params_ = dict(**params_["~"], **params_["linear"])
    p = jax.tree_util.tree_map(lambda x:np.asarray(x), params_)

    # get predictions
    x = (x - p["mean"]) / p["std"]
    x = (x * p["w"][:,0]) + (p["b"] / x.shape[-1])
    p_bind_aa = x.reshape(x.shape[0],2,20,-1).sum((1,3))
    p_bind = sigmoid(p_bind_aa.sum(-1))
    return {"p_bind":p_bind, "p_bind_aa":p_bind_aa}

def load_model():
    af_model = mk_afdesign_model(protocol="binder", debug=True)
    return af_model

def predict(af_model, pdb_path, chain_id):
    try:
        af_model.prep_inputs(pdb_filename=pdb_path,
                                 chain=chain_id,
                                 binder_len=20,
                                 rm_target_sc=MASK_SIDECHAINS,
                                 rm_target_seq=MASK_SEQUENCE)
    except ValueError as e:
        print(f"Error processing {pdb_path} chain {chain_id}: {e}")
        return None
    r_idx = af_model._inputs["residue_index"][-20] + (1 + np.arange(20)) * 50
    af_model._inputs["residue_index"][-20:] = r_idx.flatten()
    
    if af_model._inputs["aatype"].shape[0] >= MAX_RESIDUES_COUNT:
        print(f'Error: {pdb_path} chain {chain_id} has more than {MAX_RESIDUES_COUNT} residues, which would cause OOM error.')
        return None
    
    print(f"Predicting binding for {pdb_path} chain {chain_id}, size {af_model._inputs['aatype'].shape}...")
    af_model.set_seq("ACDEFGHIKLMNPQRSTVWY")
    af_model.predict(verbose=False)
    
    o = af2bind(af_model.aux["debug"]["outputs"],
                mask_sidechains=MASK_SIDECHAINS)
    pred_bind = o["p_bind"].copy()
    
    return pred_bind


###### End of adapted code ######

import argparse
import os
from biotite.structure.io.pdb import PDBFile, get_structure
from biotite.structure import get_residues

def save_predictions(pred_bind, output_path, pdb_filepath, chain_id):
    with open(output_path, 'w') as f_pred:
        f_pred.write("residue_number\tamino_acid\tchain\tp_bind\n")
        
        # load PDB and get residue info
        pdb_file = PDBFile.read(pdb_filepath)    
        protein = get_structure(pdb_file, model=1)
        residue_ids, residue_types = get_residues(protein)
        
        assert len(residue_ids) == len(pred_bind), f"Length mismatch for {pdb_filepath}: {len(residue_ids)} residues vs {len(pred_bind)} predictions"

        # loop through residues (ids and types) and write their predicted binding probabilities to the output file
        for res_id, res_name, p in zip(residue_ids, residue_types, pred_bind):
            f_pred.write(f"{res_id}\t{res_name}\t{chain_id}\t{p:.4f}\n")
    
    return pred_bind

def main(input_path, output_path):
    model = load_model()

    for file in os.listdir(input_path):
        output_filename = file.replace(".pdb", ".txt")
        if not file.endswith(".pdb"):
            continue
        if output_filename in os.listdir(output_path):
            continue

        input_path_file = os.path.join(input_path, file)

        print(f"Processing {input_path_file}...")

# ----> CAUTION: the chain ID needs to be changed if the structure is not AlphaFold output
        chain_id = "A"
        gc.collect()
        # K.clear_session()

        predictions = predict(model, input_path_file, chain_id)
        if predictions is None: # this might happen if there was an error processing the PDB
            continue
        
        save_predictions(predictions, os.path.join(output_path, output_filename), input_path_file, chain_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    main(args.input_path, args.output_path)
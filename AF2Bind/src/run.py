###### This code is adapted from the following repository: github.com/sokrypton/af2bind (see LICENSE file) ######

import numpy as np
from colabdesign import mk_afdesign_model
import jax, pickle
from scipy.special import expit as sigmoid

MASK_SIDECHAINS = True
MASK_SEQUENCE = False

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
    af_model.prep_inputs(pdb_filename=pdb_path,
                         chain=chain_id,
                         binder_len=20,
                         rm_target_sc=MASK_SIDECHAINS,
                         rm_target_seq=MASK_SEQUENCE)
    
    r_idx = af_model._inputs["residue_index"][-20] + (1 + np.arange(20)) * 50
    af_model._inputs["residue_index"][-20:] = r_idx.flatten()
    
    af_model.set_seq("ACDEFGHIKLMNPQRSTVWY")
    af_model.predict(verbose=False)
    
    o = af2bind(af_model.aux["debug"]["outputs"],
                mask_sidechains=MASK_SIDECHAINS)
    pred_bind = o["p_bind"].copy()
    return pred_bind

###### End of adapted code ######



import argparse
import os

def main(input_path, output_path):
    for file in os.listdir(input_path):
        if not file.endswith(".pdb"):
            continue
        input_path_file = os.path.join(input_path, file)

        model = load_model()

# ----> CAUTION: the chain ID needs to be changed if the structure is not AlphaFold output
        chain_id = "A"
        predictions = predict(model, input_path_file, chain_id)

        np.save(os.path.join(output_path, file.replace(".pdb", ".npy")), predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    main(args.input_path, args.output_path)
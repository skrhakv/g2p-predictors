import os
import numpy as np

from colabdesign.af.alphafold.common import residue_constants
from colabdesign import mk_afdesign_model

import jax, pickle
from scipy.special import expit as sigmoid
        
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
    with open(f"/opt/af2bind/attempt_7_2k_lam0-03/{model_type}.pickle","rb") as handle:
        params_ = pickle.load(handle)
    params_ = dict(**params_["~"], **params_["linear"])
    p = jax.tree_util.tree_map(lambda x:np.asarray(x), params_)

    # get predictions
    x = (x - p["mean"]) / p["std"]
    x = (x * p["w"][:,0]) + (p["b"] / x.shape[-1])
    p_bind_aa = x.reshape(x.shape[0],2,20,-1).sum((1,3))
    p_bind = sigmoid(p_bind_aa.sum(-1))
    return {"p_bind":p_bind, "p_bind_aa":p_bind_aa}


mask_sidechains = True
mask_sequence = False

target_pdb = '2W83'
target_chain = "A"

af_model = mk_afdesign_model(protocol="binder", debug=True)
af_model.prep_inputs(pdb_filename='/opt/af2bind/2W83.pdb',
                     chain=target_chain,
                     binder_len=20,
                     rm_target_sc=mask_sidechains,
                     rm_target_seq=mask_sequence)

r_idx = af_model._inputs["residue_index"][-20] + (1 + np.arange(20)) * 50
af_model._inputs["residue_index"][-20:] = r_idx.flatten()

af_model.set_seq("ACDEFGHIKLMNPQRSTVWY")
af_model.predict(verbose=False)

o = af2bind(af_model.aux["debug"]["outputs"],
            mask_sidechains=mask_sidechains)
pred_bind = o["p_bind"].copy()
import warnings 
warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = UserWarning)
import MDAnalysis as mda
import nglview as nv
from MDAnalysis.analysis import distances
import numpy as np
import matplotlib.pyplot as plt
import os
import sys 
import glob 
import h5py 
import numpy as np 
from tqdm import tqdm 

pdb_file = '../S1_closed_cleaved_final_min_lipidsw_ions.pdb' 
dcd_files = sorted(glob.glob('../pr.*.dcd'))

def triu_to_full(cm0):
    num_res = int(np.ceil((len(cm0) * 2) ** 0.5))
    iu1 = np.triu_indices(num_res, 1)
    cm_full = np.zeros((num_res, num_res))
    cm_full[iu1] = cm0
    cm_full.T[iu1] = cm0
    np.fill_diagonal(cm_full, 1)
    return cm_full



contact_maps = []
mda_traj = mda.Universe(pdb_file, dcd_files)  
protein_ca = mda_traj.select_atoms('protein and name CA') 

for _ in tqdm(mda_traj.trajectory[:10]): 
        contact_map = triu_to_full(
                        (distances.self_distance_array(protein_ca.positions) < 8.0) * 1.0
                                )
        contact_maps.append(contact_map) 
contact_maps = np.array(contact_maps)

# padding if odd dimension occurs in image
pad_f = lambda x: (0, 0) if x % 2 == 0 else (0, 1)
padding_buffer = [(0, 0)]
for x in contact_maps.shape[1:]:
    padding_buffer.append(pad_f(x))
contact_maps = np.pad(contact_maps, padding_buffer, mode='constant')
contact_maps = contact_maps.reshape((contact_maps.shape) + (1,))

        # save formulated contact maps to hdf5
cm_h5 = h5py.File('contact_maps.h5', 'w') 
cm_h5.create_dataset('contact_maps', data=contact_maps) 
cm_h5.close() 

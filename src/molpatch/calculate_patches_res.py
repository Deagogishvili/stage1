from ResidueBased.ProteinPatch import ProteinPatch
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from os import listdir
from os.path import isfile, join
from PisiteParser import PisiteParser
import pandas as pd

hydr_residues = ['A','F', 'C', 'L', 'I', 'W', 'V', 'M', 'Y']

path = '/home/jan/Documents/BioInformatics/final_project_patch/data/pdb/hoh/'
csv_file = '../../data/patches/lp_residue_all.csv'
#
# if isfile(csv_file):
#     print('CSV already exists')
#     exit()

files = [f for f in listdir(path) if isfile(join(path, f))]

df = pd.read_csv(csv_file)

for file in files:
    try:
        proteinPatches = ProteinPatch(id,path+file,hydr_residues)
        patches = proteinPatches.patches
        for i,patch in enumerate(patches[:20]):
            result_dict = {}
            result_dict['patch_size'] = patch.patch_length()
            result_dict['on_surface'] = len(patch.residue_on_surface())
            result_dict['id'] = file[:-4]
            result_dict['residues'] = ''.join(patch.residues())
            result_dict['rank'] = int(i+1)
            result_dict['size'] = patch.size()
            df = df.append(pd.Series(result_dict), ignore_index=True)

        df.to_csv(csv_file, index=False)
    except:
        print(file, 'failed')

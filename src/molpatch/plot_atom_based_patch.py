from AtomBased.ProteinPatch import ProteinPatch
import yaml
config = yaml.safe_load(open("../config.yml"))

id = '3BIJA' #bad
# id =  '1NV8A' #perfect
# id =  '1GAKA' #perfect
# id =  '4A2LA' #perfect
id =  '4O5JA' #perfect
# id =  '4MEAA' #perfect
# id =  '1LMLA' #perfect
path = '/home/jan/Documents/BioInformatics/final_project_patch/data/pdb/hoh/'
file = id+'.ent'

id =  '2WPTA' #perfect
path = '/home/jan/Downloads/'
file = id+'.pdb'

atoms = ['C', 'S']

proteinPatch = ProteinPatch(id,path+file,atoms)
proteinPatch.plot_largest_patches()

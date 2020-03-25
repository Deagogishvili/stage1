from Bio.PDB import PDBList
from Bio import SeqIO

def download_pdb_files(FASTA_FILE, PDB_OUTPUT):
    # get protein ids
    records = list(SeqIO.parse(FASTA_FILE, "fasta"))
    protein_ids = set([x.id[:-1] for x in records])

    # download pdbs
    pdbl = PDBList()
    pdbl.download_pdb_files(protein_ids, pdir=PDB_OUTPUT)

if __name__ == '__main__':
    import yaml
    import sys

    config = yaml.safe_load(open("../config.yml"))
    if config['log']:
        log = open(config['path']['logs']+'download.log', "w")
        sys.stdout = log

    download_pdb_files(config['path']['fasta'], config['path']['protein'])

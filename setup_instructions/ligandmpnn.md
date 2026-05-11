## LigandMPNN

This package provides inference code for [LigandMPNN](https://www.biorxiv.org/content/10.1101/2023.12.22.573103v1) & [ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187) models. The code and model parameters are available under the MIT license.

Third party code: side chain packing uses helper functions from [Openfold](https://github.com/aqlaboratory/openfold).

### Running the code
```
git clone https://github.com/dauparas/LigandMPNN.git
cd LigandMPNN
bash get_model_params.sh "./model_params"

#setup your conda/or other environment
#conda create -n ligandmpnn_env python=3.11
#pip3 install -r requirements.txt

python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/default"
```

### Dependencies
To run the model you will need to have Python>=3.0, PyTorch, Numpy installed, and to read/write PDB files you will need [Prody](https://pypi.org/project/ProDy/).

For example to make a new conda environment for LigandMPNN run:
```
conda create -n ligandmpnn_env python=3.11
pip3 install -r requirements.txt
```

### Main differences compared with [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) code
- Input PDBs are parsed using [Prody](https://pypi.org/project/ProDy/) preserving protein residue indices, chain letters, and insertion codes. If there are missing residues in the input structure the output fasta file won't have added `X` to fill the gaps. The script outputs .fasta and .pdb files. It's recommended to use .pdb files since they will hold information about chain letters and residue indices.
- Adding bias, fixing residues, and selecting residues to be redesigned now can be done using residue indices directly, e.g. A23 (means chain A residue with index 23), B42D (chain B, residue 42, insertion code D).
- Model writes to fasta files: `overall_confidence`, `ligand_confidence` which reflect the average confidence/probability (with T=1.0) over the redesigned residues  `overall_confidence=exp[-mean_over_residues(log_probs)]`. Higher numbers mean the model is more confident about that sequence. min_value=0.0; max_value=1.0. Sequence recovery with respect to the input sequence is calculated only over the redesigned residues.

### Model parameters
To download model parameters run:
```
bash get_model_params.sh "./model_params"
```


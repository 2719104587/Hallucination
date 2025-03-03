Code for Hallucination
--------------------
<p align="center">
  <img src="./img/xlab.png" alt="alt text" width="600px"/>
</p>

### Setup/Installation
1. Clone the package
```
git clone https://github.com/2719104587/Hallcination.git
```

2. Download the trfold model weights.
```
https://www.modelscope.cn/models/DongZekai/TRFold-params
```

3. Download the alphafold model weights, The alphafold model version is v1, Remember that the model path is specified in the configuration.
```
https://github.com/google-deepmind/alphafold.git
```
or 
```
https://www.modelscope.cn/models/DongZekai/Alphafold-params
```

4. Python and cuda versions, python3.9 and cuda11.7 are recommended.
```
python 3.9
cuda 11.7
```

5. Install the dependencies.
```
pip install -r requirements.txt
```

### Inference
#### Follow the paper

You can use example_scripts to follow the design in the paper:

Example(Single epitope multiple display design):
```
bash example_scripts/motif_gap_same.sh
bash example_scripts/motif_gap.sh
bash C3_symmetry.sh
```

Example(Three epitope design):
```
bash example_scripts/RBD_gd.sh
bash example_scripts/RBD.sh
bash example_scripts/RBD_targe.sh
bash example_scripts/RBD_link_E2E3.sh
```

Note: To set parameters, see help? In hallucination.py, Following the overall flow of the article requires some additional tools.

Explanation of arguments:
- `length=80-120` This batch of designs will range from a length between 80-120 as the length of the protein sequence
- `motif-locs=A30-40` The A chain in the pdb file is expressed, and the amino acid id ranges from 30 to 40, and the structure of this region is conserved during the design process.
- `functional-sites=A30-35` The expression of the A chain in the pdb file, the amino acid id from 30-35 region, in the design process will not only conservative the structure of this part of the region, but also conservative the amino acid type. 36-40 is conserved and the amino acid type is variable.
- `motif-dir=./output/motif_dir` Refer to the directory where the pdb file resides and the directory where the pk file resides.
- `prefix=6xr8` The reference pdb path is./output/motif_dir/6xr8.pdb.

Expected outputs:
- `example_files/output/6xr8_A_1_134_rec.pk` Necessary design result information, such as the position of each motif in the design protein.
- `example_files/output/6xr8_A_1_134.fasta` Design protein sequence.
- `example_files/output/6xr8_A_1_134.pdb` Design the structure of the protein.
- `example_files/output/generate-1.log` Log of the design process, including detailed parameter Settings and loss changes under each step.

#### Protein design in a new scenario

You need to read the article carefully, clearly understand the meaning and correlation of each parameter in the code, and then set the parameters reasonably according to your own design goals, so that it is possible to achieve the design goals you want.

### Other tools
1. RFdiffusion
```
git clone https://github.com/RosettaCommons/RFdiffusion.git

```

2. ProteinMPNN
```
git clone https://github.com/dauparas/ProteinMPNN.git

```

3. dl_binder_design
```
git clone https://github.com/nrbennet/dl_binder_design.git

```
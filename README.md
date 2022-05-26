# ACE2_allosteric_modeling

## NCATS models

Stratified-bagging models for the prediction of ACE2 allosteric Inhibitors

The jupyter notebook for the prediction of ACE2 allosteric Inhibitors using Stratified Bagging models and Morgan fingerprints.

Installation and dependencies Anaconda (https://docs.anaconda.com/anaconda/navigator/install/) >= 1.9.12; Python >= 3.7.7; Pandas >= 1.0.3; numpy >= 1.18.1

### Usage
    1. From the anaconda navigator, start the jupyter notebook and load the file ‘ACE2_allosteric_Prediction_StratifiedBagging.ipynb’ available under the folder ‘ACE2_StratifiedBagging’
    2. Calculate the Morgan fingerprint for the test compounds (sample knime workflow provided under the folder ‘knime_workflow_descriptor_calculation’).
    3. Place the generated file with the calculated descriptor in the folder ‘test_data’. A sample file is available (sample_test_file_morgan.csv).
    4. In the jupyter notebook ‘ACE2_allosteric_Prediction_StratifiedBagging.ipynb’, change the name of the test file (3rd cell; #Path to the test file)
    5. Execute the notebook.
    6. The ACE2 allosteric Inhibitor’s prediction using Stratified Bagging models and Morgan fingerprints will be saved in the ‘output_predictions’ folder.
    
The folder ‘SB_models’ also contains Stratified Bagging models generated for Avalon Fingerprint and RDKit descriptors (physicochemical properties).

The folder ‘knime_workflow_descriptor_calculation’ contains the knime workflow to generate the descriptor files for Morgan Fingerprint, Avalon Fingerprint and RDKit descriptors (physicochemical properties). This workflow takes input a .csv file with columns: ‘Molecule name’; ‘smiles’

The folder ‘pharmacophore_models’ contains Ligand-based pharmacophore models generated for the virtual screening.

## UNC Models

### Installation:
  1. Create a conda environment created from the requirements file environment.yaml
  2. Clone PyMuDRA (https://github.com/MikolajMizera/pyMuDRA) and update its filepath in models.py
  3. Construct models by running python train_models.py. Models will be saved to the folder models, which will be created if it doesn’t already exists
  4. Run models on external molecules following the example file run_models.py

### Notes:
Exact model files used in the paper are not provided here, but these are exact model architectures and training procedures. Validation statistics produced with this code should be very similar to what is reported in the paper.

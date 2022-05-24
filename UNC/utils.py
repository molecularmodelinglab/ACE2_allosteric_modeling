import torch
import numpy as np
import pandas as pd
import glob
import mol as ModelingMol
from mol import get_descriptors
from mol import write_sdf
from models import Model, RF, GradientBoosting
from sklearn import metrics
import rdkit
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import SDMolSupplier
from itertools import product
from rdkit.Chem import Descriptors, Crippen, MolSurf, Lipinski, Fragments, EState, GraphDescriptors
from rdkit import Chem
import os
import time as time_module

from torch.utils.data import Dataset

from sklearn.neighbors import NearestNeighbors

class SirmsHolder:

    def __init__(self, mols, filename):

        f = open(filename)
        descs = []
        for i, line in enumerate(f):
            if i == 0:
                continue
            s = line.split()
            desc = np.array(s, dtype = int)
            descs.append(desc)

        matrix = np.array(descs)
        self.matrix = matrix
        ids = [mol.identifier for mol in mols]
        self.descriptor_dictionary = {}
        for i, id_val in enumerate(ids):
            self.descriptor_dictionary[id_val] = i

    def process(self):

        matrix, _, _ = remove_constant_columns(self.matrix, threshold = 0.5)
        matrix, _, _ = remove_correlated_columns(matrix, cc_threshold = 0.9)
        self.matrix = matrix


    def get_sirms_descriptor(self, mol):

        desc = self.matrix[self.descriptor_dictionary[mol.identifier]]
        return desc

#values equal to or above cutoff become 1, values below cutoff become 0
def discretize(x, cutoff = 0.5):

    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype = float)

    return torch.ge(x, cutoff).squeeze()

def classification_statistics(actual, predicted, to_string = False):

    cm = metrics.confusion_matrix(actual, predicted)

    accuracy = metrics.accuracy_score(actual, predicted).item()
    f1 = metrics.f1_score(actual, predicted).item()
    Kappa = metrics.cohen_kappa_score(actual, predicted, weights='linear').item()

    TN, FP, FN, TP = cm.ravel()
    TN = TN.item()
    FP = FP.item()
    FN = FN.item()
    TP = TP.item()
    stats = statistics_from_cm(TN,FN,TP,FP)
    return stats

def statistics_from_cm(TN,FN,TP,FP):

    # Sensitivity, hit rate, recall, or true positive rate
    try:
        SE = TP/(TP+FN)
    except:
        SE = 0

    # Specificity or true negative rate
    try:
        SP = TN/(TN+FP) 
    except:
        SP = 0

    # Precision or positive predictive value
    try:
        PPV = TP/(TP+FP)
    except:
        PPV = 0

    # Negative predictive value
    try:
        NPV = TN/(TN+FN)
    except:
        NPV = 0

    # Correct classification rate
    CCR = (SE + SP)/2

    return {'CCR': CCR, 'SP':SP, 'SE':SE ,'PPV': PPV, 'NPV': NPV, 'TN': TN, 'TP': TP, 'FN': FN, 'FP': FP }


def continuous_statistics(actual, predicted, to_string = False):
     r2 = metrics.r2_score(actual, predicted).item()
     explained_variance = metrics.explained_variance_score(actual, predicted).item()
     max_error = metrics.max_error(actual, predicted).item()
     rmse = metrics.mean_squared_error(actual, predicted, squared = False)
     return {"r2":r2, "explained_variance":explained_variance,"max_error":max_error, "rmse":rmse}


def ad_coverage(train_mols, test_mols, descriptor_function, k = 1, z = 1):

    train_descriptors, _ = get_descriptors(train_mols, descriptor_function)
    test_descriptors, _ = get_descriptors(test_mols, descriptor_function)

    predictor = NearestNeighbors(n_neighbors = k + 1, algorithm = "ball_tree")
    predictor.fit(train_descriptors)
    distances, indices = predictor.kneighbors(train_descriptors)
    distances = distances[:,1:] #trim out self distance

    d = np.mean(distances)
    s = np.std(distances)
    D = d + (z*s) #define cutoff

    distances, indices = predictor.kneighbors(test_descriptors)
    distances = distances[:,:-1]
    if distances.shape[1] > 1:
        distances = np.mean(distances, axis = 0)

    accepted = np.where(distances < D, 1, 0)

    coverage = np.sum(accepted) / len(accepted)

    return coverage


#returns k (x, y) tuples in a list
#last set will be largest by at most k-1 if not cleanly divisible
def k_fold_split(l, k = 5):

    l = np.array(l)

    np.random.shuffle(l)

    return_list = []

    step = int((1 / k) * len(l))
    for i in range(1,k + 1):
        start = (i - 1) * step
        if i == k:
            fold_l = l[start:]
            return_list.append(fold_l)

        else:
            stop = i * step
            fold_l = l[start:stop]
            return_list.append(fold_l)

    return return_list

class train_test_split:

    def __init__(self, mols, k = 5, verbose = False):
        self.folds = k_fold_split(mols, k = k)
        self.k = k
        self.fold_num = 0
        self.verbose = verbose

    def __iter__(self):
        return self

    def __next__(self):
        #print(self.fold_num, self.k)

        if self.fold_num >= self.k:
            raise StopIteration()

        train = []
        for i in range(self.k):

            if i != self.fold_num:
                m = self.folds[i]
                train.append(m)

            else:
                test = self.folds[i]

        self.fold_num = self.fold_num + 1

        train = np.concatenate(train)

        if(self.verbose):
            print(f"Train shape: {train_x.shape}")
            print(f"Test shape: {test_x.shape}")

        return (train, test)

def get_morgan_descriptor(mol, radius = 2, convert_to_np = True):

    if type(mol) == rdkit.Chem.rdchem.Mol:
        rdkit_mol = mol
    else:
        rdkit_mol = mol.mol

    fp = AllChem.GetMorganFingerprintAsBitVect(rdkit_mol, radius)

    if convert_to_np:
        arr = np.array((0,))
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol.mol, radius), arr)
        arr = np.array(arr, dtype=np.float32)
        return arr

    return fp

def get_mudra_paper_descriptor(mol, radius = 2):

    if type(mol) == rdkit.Chem.rdchem.Mol:
        rdkit_mol = mol
    else:
        rdkit_mol = mol.mol

    morgan = np.array((0,))
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(rdkit_mol, radius), morgan)

    atom_pair = np.array((0,))
    DataStructs.ConvertToNumpyArray(AllChem.GetHashedAtomPairFingerprintAsBitVect(rdkit_mol), atom_pair)

    torsion = np.array((0,))
    DataStructs.ConvertToNumpyArray(AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(rdkit_mol), torsion)

    maccs = np.array((0,))
    DataStructs.ConvertToNumpyArray(AllChem.GetMACCSKeysFingerprint(rdkit_mol), maccs)

    return (morgan, atom_pair, torsion, maccs)

def get_big_binary_descriptor(mol, radius = 2):

    morgan = np.array((0,))
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, radius), morgan)

    atom_pair = np.array((0,))
    DataStructs.ConvertToNumpyArray(AllChem.GetHashedAtomPairFingerprintAsBitVect(mol), atom_pair)

    torsion = np.array((0,))
    DataStructs.ConvertToNumpyArray(AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol), torsion)

    maccs = np.array((0,))
    DataStructs.ConvertToNumpyArray(AllChem.GetMACCSKeysFingerprint(mol), maccs)

    return np.hstack(morgan, atom_pair, torsion, maccs)


def get_super_descriptor(mol):

    morgan = np.array((0,))
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, radius = 3), morgan)

    atom_pair = np.array((0,))
    DataStructs.ConvertToNumpyArray(AllChem.GetHashedAtomPairFingerprintAsBitVect(mol), atom_pair)

    torsion = np.array((0,))
    DataStructs.ConvertToNumpyArray(AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol), torsion)

    maccs = np.array((0,))
    DataStructs.ConvertToNumpyArray(AllChem.GetMACCSKeysFingerprint(mol), maccs)

    rdkit_desc = get_all_rdkit_descriptors(mol)

    desc = np.hstack((morgan, atom_pair, torsion, maccs, rdkit_desc))
    desc = np.array(desc, dtype=np.float32)

    return desc


def get_maccs_key(mol):

    maccs = np.array((0,))
    DataStructs.ConvertToNumpyArray(AllChem.GetMACCSKeysFingerprint(mol), maccs)
    return maccs

#identifying_column_names will be used as keys, everything else will be averaged
def average_results(df, identifying_column_names, method = "classification"):

    if method == "classification":
        for_std = ["CCR", "Coverage", "PPV", "NPV", "PPV", "SE", "SP"]
    else:
        for_std = ["explained_variance", "max_error", "r2"]

    #TODO: reorder df so identifying_column_names come first
    print(identifying_column_names)

    averaged_results = pd.DataFrame()
    consensus_results = pd.DataFrame()

    possibilities = []
    for column_name in identifying_column_names:
        vals = list(df[column_name].unique())
        full_vals = [(column_name, val) for val in vals]
        possibilities.append(full_vals)

    combs = list(product(*possibilities))

    actuals = []
    predicteds = []
    labels = []
    print(df.columns)
    for comb in combs:

        temp_df = df
        for key,val in comb:
            a = temp_df[temp_df[key] == val]
            temp_df = temp_df[temp_df[key] == val]


        if not temp_df.empty:

            print(temp_df)
            num_averaged = temp_df.shape[0]
            averaged = temp_df.mean(axis = 0)
            std = temp_df.std(axis = 0)
            for key,val in comb:
                averaged[key] = val
            averaged["Num Averaged"] = num_averaged

            for col_name in for_std:
                std = temp_df[col_name].std()
                std_name = f"{col_name}_std"
                averaged[std_name] = std

            consensus = pd.Series()

            if method == "classification":
                fn = np.sum(temp_df["FN"])
                tn = np.sum(temp_df["TN"])
                fp = np.sum(temp_df["FP"])
                tp = np.sum(temp_df["TP"])

                pos = fn + tp
                neg = fp + tn

                cv_stats = statistics_from_cm(tn,fn,tp,fp)

                consensus["# Positive Examples"] = pos
                consensus["# Negative Examples"] = neg
                for key,val in cv_stats.items():
                    consensus[key] = val

            true = []
            predicted = []
            for i in range(len(temp_df)):
                s = temp_df.iloc[i]
                fold_true = s["Test labels"]
                fold_pred = s["Test predictions"]
                true.extend(fold_true)
                predicted.extend(fold_pred)

            actuals.append(true)
            predicteds.append(predicted)
            label = " ".join([x[1] for x in comb])
            labels.append(label)

            consensus["Model Name"] = averaged["Model Name"] + " (5-Fold CV)"
            consensus["Descriptor Function"] = averaged["Descriptor Function"]
            consensus_results = consensus_results.append(consensus, ignore_index = True)

            averaged["Model Name"] = averaged["Model Name"] + " (Averaged)"
            averaged_results = averaged_results.append(averaged, ignore_index = True)

    averaged_results = averaged_results.drop("Fold", axis = 1)
    return averaged_results, consensus_results

def get_all_rdkit_descriptors(mol):

    if type(mol) == rdkit.Chem.rdchem.Mol:
        rdkit_mol = mol
    else:
        rdkit_mol = mol.mol

    descriptor_functions = np.array([Crippen.MolLogP,
                                      Crippen.MolMR,
                                      Descriptors.FpDensityMorgan1,
                                      Descriptors.FpDensityMorgan2,
                                      Descriptors.FpDensityMorgan3,
                                      Descriptors.FractionCSP3,
                                      Descriptors.HeavyAtomMolWt,
                                      Descriptors.MaxAbsPartialCharge,
                                      Descriptors.MaxPartialCharge,
                                      Descriptors.MinAbsPartialCharge,
                                      Descriptors.MinPartialCharge,
                                      Descriptors.MolWt,
                                      Descriptors.NumRadicalElectrons,
                                      Descriptors.NumValenceElectrons,
                                      EState.EState.MaxAbsEStateIndex,
                                      EState.EState.MaxEStateIndex,
                                      EState.EState.MinAbsEStateIndex,
                                      EState.EState.MinEStateIndex,
                                      EState.EState_VSA.EState_VSA1,
                                      EState.EState_VSA.EState_VSA10,
                                      EState.EState_VSA.EState_VSA11,
                                      EState.EState_VSA.EState_VSA2,
                                      EState.EState_VSA.EState_VSA3,
                                      EState.EState_VSA.EState_VSA4,
                                      EState.EState_VSA.EState_VSA5,
                                      EState.EState_VSA.EState_VSA6,
                                      EState.EState_VSA.EState_VSA7,
                                      EState.EState_VSA.EState_VSA8,
                                      EState.EState_VSA.EState_VSA9,
                                      Fragments.fr_Al_COO,
                                      Fragments.fr_Al_OH,
                                      Fragments.fr_Al_OH_noTert,
                                      Fragments.fr_aldehyde,
                                      Fragments.fr_alkyl_carbamate,
                                      Fragments.fr_alkyl_halide,
                                      Fragments.fr_allylic_oxid,
                                      Fragments.fr_amide,
                                      Fragments.fr_amidine,
                                      Fragments.fr_aniline,
                                      Fragments.fr_Ar_COO,
                                      Fragments.fr_Ar_N,
                                      Fragments.fr_Ar_NH,
                                      Fragments.fr_Ar_OH,
                                      Fragments.fr_ArN,
                                      Fragments.fr_aryl_methyl,
                                      Fragments.fr_azide,
                                      Fragments.fr_azo,
                                      Fragments.fr_barbitur,
                                      Fragments.fr_benzene,
                                      Fragments.fr_benzodiazepine,
                                      Fragments.fr_bicyclic,
                                      Fragments.fr_C_O,
                                      Fragments.fr_C_O_noCOO,
                                      Fragments.fr_C_S,
                                      Fragments.fr_COO,
                                      Fragments.fr_COO2,
                                      Fragments.fr_diazo,
                                      Fragments.fr_dihydropyridine,
                                      Fragments.fr_epoxide,
                                      Fragments.fr_ester,
                                      Fragments.fr_ether,
                                      Fragments.fr_furan,
                                      Fragments.fr_guanido,
                                      Fragments.fr_halogen,
                                      Fragments.fr_hdrzine,
                                      Fragments.fr_hdrzone,
                                      Fragments.fr_HOCCN,
                                      Fragments.fr_imidazole,
                                      Fragments.fr_imide,
                                      Fragments.fr_Imine,
                                      Fragments.fr_isocyan,
                                      Fragments.fr_isothiocyan,
                                      Fragments.fr_ketone,
                                      Fragments.fr_ketone_Topliss,
                                      Fragments.fr_lactam,
                                      Fragments.fr_lactone,
                                      Fragments.fr_methoxy,
                                      Fragments.fr_morpholine,
                                      Fragments.fr_N_O,
                                      Fragments.fr_Ndealkylation1,
                                      Fragments.fr_Ndealkylation2,
                                      Fragments.fr_NH0,
                                      Fragments.fr_NH1,
                                      Fragments.fr_NH2,
                                      Fragments.fr_Nhpyrrole,
                                      Fragments.fr_nitrile,
                                      Fragments.fr_nitro,
                                      Fragments.fr_nitro_arom,
                                      Fragments.fr_nitro_arom_nonortho,
                                      Fragments.fr_nitroso,
                                      Fragments.fr_oxazole,
                                      Fragments.fr_oxime,
                                      Fragments.fr_para_hydroxylation,
                                      Fragments.fr_phenol,
                                      Fragments.fr_phenol_noOrthoHbond,
                                      Fragments.fr_phos_acid,
                                      Fragments.fr_phos_ester,
                                      Fragments.fr_piperdine,
                                      Fragments.fr_piperzine,
                                      Fragments.fr_priamide,
                                      Fragments.fr_prisulfonamd,
                                      Fragments.fr_pyridine,
                                      Fragments.fr_quatN,
                                      Fragments.fr_SH,
                                      Fragments.fr_sulfide,
                                      Fragments.fr_sulfonamd,
                                      Fragments.fr_sulfone,
                                      Fragments.fr_term_acetylene,
                                      Fragments.fr_tetrazole,
                                      Fragments.fr_thiazole,
                                      Fragments.fr_thiocyan,
                                      Fragments.fr_thiophene,
                                      Fragments.fr_unbrch_alkane,
                                      Fragments.fr_urea,
                                      GraphDescriptors.BalabanJ,
                                      GraphDescriptors.BertzCT,
                                      GraphDescriptors.Chi0,
                                      GraphDescriptors.Chi0n,
                                      GraphDescriptors.Chi0v,
                                      GraphDescriptors.Chi1,
                                      GraphDescriptors.Chi1n,
                                      GraphDescriptors.Chi1v,
                                      GraphDescriptors.Chi2n,
                                      GraphDescriptors.Chi2v,
                                      GraphDescriptors.Chi3n,
                                      GraphDescriptors.Chi3v,
                                      GraphDescriptors.Chi4n,
                                      GraphDescriptors.Chi4v,
                                      GraphDescriptors.HallKierAlpha,
                                      GraphDescriptors.Ipc,
                                      GraphDescriptors.Kappa1,
                                      GraphDescriptors.Kappa2,
                                      GraphDescriptors.Kappa3,
                                      Lipinski.HeavyAtomCount,
                                      Lipinski.NHOHCount,
                                      Lipinski.NOCount,
                                      Lipinski.NumAliphaticCarbocycles,
                                      Lipinski.NumAliphaticHeterocycles,
                                      Lipinski.NumAliphaticRings,
                                      Lipinski.NumAromaticCarbocycles,
                                      Lipinski.NumAromaticHeterocycles,
                                      Lipinski.NumAromaticRings,
                                      Lipinski.NumHAcceptors,
                                      Lipinski.NumHDonors,
                                      Lipinski.NumHeteroatoms,
                                      Lipinski.NumRotatableBonds,
                                      Lipinski.NumSaturatedCarbocycles,
                                      Lipinski.NumSaturatedHeterocycles,
                                      Lipinski.NumSaturatedRings,
                                      Lipinski.RingCount,
                                      MolSurf.LabuteASA,
                                      MolSurf.PEOE_VSA1,
                                      MolSurf.PEOE_VSA10,
                                      MolSurf.PEOE_VSA11,
                                      MolSurf.PEOE_VSA12,
                                      MolSurf.PEOE_VSA13,
                                      MolSurf.PEOE_VSA14,
                                      MolSurf.PEOE_VSA2,
                                      MolSurf.PEOE_VSA3,
                                      MolSurf.PEOE_VSA4,
                                      MolSurf.PEOE_VSA5,
                                      MolSurf.PEOE_VSA6,
                                      MolSurf.PEOE_VSA7,
                                      MolSurf.PEOE_VSA8,
                                      MolSurf.PEOE_VSA9,
                                      MolSurf.SlogP_VSA1,
                                      MolSurf.SlogP_VSA10,
                                      MolSurf.SlogP_VSA11,
                                      MolSurf.SlogP_VSA12,
                                      MolSurf.SlogP_VSA2,
                                      MolSurf.SlogP_VSA3,
                                      MolSurf.SlogP_VSA4,
                                      MolSurf.SlogP_VSA5,
                                      MolSurf.SlogP_VSA6,
                                      MolSurf.SlogP_VSA7,
                                      MolSurf.SlogP_VSA8,
                                      MolSurf.SlogP_VSA9,
                                      MolSurf.SMR_VSA1,
                                      MolSurf.SMR_VSA10,
                                      MolSurf.SMR_VSA2,
                                      MolSurf.SMR_VSA3,
                                      MolSurf.SMR_VSA4,
                                      MolSurf.SMR_VSA5,
                                      MolSurf.SMR_VSA6,
                                      MolSurf.SMR_VSA7,
                                      MolSurf.SMR_VSA8,
                                      MolSurf.SMR_VSA9,
                                      MolSurf.TPSA])


    descriptor = []
    for descriptor_function in descriptor_functions:
        try:
            desc_val = descriptor_function(rdkit_mol)
        except:
            continue
        if np.isnan(desc_val):
            desc_val = 0
        descriptor.append(desc_val)

    descriptor = np.array(descriptor)

    a = sum(np.isnan(descriptor))
    b = sum(np.isinf(descriptor))
    if a > 0 or b > 0:
        print(descriptor)
        print("BAD DESCRIPTOR")
        exit()
    return descriptor

def check_if_mols_in_set(check_mols, set_mols):

    return [c in set_mols for c in check_mols]

def np_array_to_csv(arr):
    s = ""
    for i,val in enumerate(arr):
        if i == 0:
            s = s + str(val)
        else:
            s = s + "," + str(val)

    return s

def sdf_to_fingerprints(filename, score_column_name, descriptor_function,
        output_filename, skip_first_mol = False, limit = -1):

    sdm = SDMolSupplier(filename)
    of = open(output_filename, 'w+')

    for i, mol in enumerate(sdm):

        if skip_first_mol and i == 0:
            continue

        if limit > 0 and i >= limit:
            break

        m = Mol(mol, float(mol.GetProp(score_column_name)))
        fp = descriptor_function(m.mol)
        of.write(f"{np_array_to_csv(fp)}, {m.score}\n")
        print(f"{i} mols fingerprinted\r", end = "")

def read_fingerprint_file(filename):

    fps = []
    scores = []
    f = open(filename, 'r')
    for i, line in enumerate(f):
        s = line.split(',')
        score = s[-1]
        fp = s[:-1]

        fp = np.array(fp, dtype = int)
        score = float(score)
        fps.append(fp)
        scores.append(score)
        print(f"{i} fingerprints read\r", end="")

    return fps, scores

def get_training_mols():

    filename = "data/curated/ace-2_allosteric_curated.sdf"
    ids = set()
    sup = Chem.rdmolfiles.SDMolSupplier(filename)
    mols = []
    for mol in sup:
        ids.add(mol.GetProp("Sample ID"))
        mm = ModelingMol.Mol(mol, score = int(float(mol.GetProp("Outcome"))), identifier = mol.GetProp("Sample ID"))
        mols.append(mm)

    return mols

def get_ncats_screening_mols():

    filename = "data/curated/sytravon_prepared.sdf"
    ids = set()
    sup = Chem.rdmolfiles.SDMolSupplier(filename)
    mols = []
    for mol in sup:
        ids.add(mol.GetProp("NCGC ID"))
        mm = ModelingMol.Mol(mol, score = None, identifier = mol.GetProp("NCGC ID"))
        mols.append(mm)

    filename = "data/curated/genesis_prepared.sdf"
    ids = set()
    sup = Chem.rdmolfiles.SDMolSupplier(filename)
    for mol in sup:
        ids.add(mol.GetProp("NCGC ID"))
        mm = ModelingMol.Mol(mol, score = None, identifier = mol.GetProp("NCGC ID"))
        mols.append(mm)

    return mols


#read all .pkl files from dir as models, then applies them to provided mols
def run_models_on_set(model_dirname, external_mols, output_dir, sirms_filename, evaluate_performance = False):

    if evaluate_performance:
        true_labels = [m.score for m in external_mols]
        performance = pd.DataFrame()

    ids = []
    for m in external_mols:
        if isinstance(m.identifier, list):
            s = ",".join(m.identifier)
        else:
            s = m.identifier
        ids.append(s)

    fold_dict = get_test_fold_mols(f"{model_dirname}/folds/allostericace2activity")

    sirmsholder = SirmsHolder(mols = external_mols, filename = sirms_filename)

    log_filename = f"{output_dir}/log.txt"
    prediction_filename = f"{output_dir}/predictions.csv"
    consensus_filename = f"{output_dir}/consensus_predictions.csv"
    statistics_filename= f"{output_dir}/statistics.csv"

    try:
        os.mkdir(output_dir)
    except:
        pass

    log = open(log_filename, 'w+')

    all_preds = pd.DataFrame()

    preds_by_id = pd.DataFrame(index = ids)
    pred_dict = {}
    score_dict = {}

    filenames = glob.glob(f"{model_dirname}/models/*.pkl")
    failed_models = []
    for filename in filenames:
        print(filename)
        s = filename.split("/")[-1].split(".")[0]
        s = s.split("_")
        target = "_".join(s[:-3])
        model_name = s[-3]
        descriptor = s[-2]
        fold = s[-1]

        print(target, model_name, descriptor, fold)

        if (model_name, descriptor) not in pred_dict:
            pred_dict[(model_name, descriptor)] = {}
            score_dict[(model_name, descriptor)] = {}

        try:
            m = Model.from_file(filename)
        except Exception as e:
            print(e)
            failed_models.append(filename)
            continue

        try:
            pred = m.predict(external_mols, run_discretize = False)
        except:
            try:
                pred = m.predict(external_mols)
                log.write(f"FAILED NON DISCRETIZE ({model_name}, {descriptor})\n")
            except:
                log.write("Encountered exception: str(e) on ({model_name}, {descriptor})")
                continue
        pred_dict[(model_name, descriptor)][int(fold)] = pred
        score_dict[(model_name, descriptor)][int(fold)] = external_mols

        if evaluate_performance:
            pred_for_eval = [int(x) for x in pred]
            stats = classification_statistics(true_labels, pred_for_eval)
            print(stats)

        s = {}
        s["Target"] = target
        s["Model Name"] = model_name
        s["Fold"] = fold
        s["Descriptor"] = descriptor
        s["Prediction"] = np.array(pred, dtype = int)
        name = descriptor + "_" + model_name
        ser = pd.Series(s)
        all_preds = all_preds.append(ser, ignore_index = True)

    for key in pred_dict.keys():

        pred = []
        true = []


        all_preds = []
        if len(pred_dict[key]) == 0:
            log.write(f"SKIPPING MODEL DUE TO EMPTY RESULTS: {key}\n")
            continue
        for i in range(max(pred_dict[key].keys()) + 1):
            output_filename = f"{output_dir}/{key[0]}_{key[1]}_{i}_predictions.txt"
            print(output_filename)
            ids = [mol.identifier for mol in score_dict[key][i]]
            ids = np.array(ids)
            pred = pred_dict[key][i]
            all_preds.append(pred_dict[key][i])
            f = open(output_filename, 'w')
            for i in range(len(ids)):
                f.write(f"{ids[i]},{pred[i]}\n")
            f.close()

def get_test_fold_mols(target):

    filename = f"{target}_test_for_fold_*"
    print(filename)
    fold_filenames = glob.glob(filename)
    fold_dict = {}
    for fold_filename in fold_filenames:
        fold_num  = fold_filename.split("_")[-1].split(".")[0]
        print(fold_num)
        mols = get_mols(fold_filename, id_name = "Sample ID", score_name = "Outcome")
        fold_dict[fold_num] = mols


    return fold_dict

def get_mols(filename, id_name, score_name):

    ids = set()
    sup = Chem.rdmolfiles.SDMolSupplier(filename)
    mols = []
    for mol in sup:
        ids.add(mol.GetProp(id_name))
        mm = ModelingMol.Mol(mol, score = int(float(mol.GetProp(score_name))), identifier = mol.GetProp(id_name))
        mols.append(mm)

    return mols

if __name__ == "__main__":
    main()


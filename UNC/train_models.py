import os
import sys
import torch
import pickle
import numpy as np
import pandas as pd
import glob
import time as time_module
from sklearn.cluster import KMeans
import copy
from models import Model, MUDRA, NeuralNetwork, GradientBoosting
from utils import train_test_split, write_sdf, get_all_rdkit_descriptors, get_morgan_descriptor, ad_coverage, classification_statistics, average_results, get_mudra_paper_descriptor, SirmsHolder, get_training_mols

import mol as ModelingMol
from rdkit import Chem
from rdkit import DataStructs

def write_csv(mols, filename):
    f = open(filename, 'w')
    for mol in mols:
        s = f"{mol.id_val},{Chem.MolToSmiles(mol.mol)},{mol.score}\n"
        f.write(s)

    f.close()

def build_models(mols, sirms_filename):

    sirmsholder = SirmsHolder(mols = mols, filename = sirms_filename)

    results = pd.DataFrame()
    time = int(time_module.time())

    top_dirname = "models"
    dirname = f"{top_dirname}/{time}"
    model_dir = f"{dirname}/models"
    fold_dir = f"{dirname}/folds"
    stat_dir = f"{dirname}/statistics"

    os.makedirs(top_dirname, exist_ok = True)
    os.makedirs(dirname, exist_ok = True)
    os.makedirs(model_dir, exist_ok = True)
    os.makedirs(fold_dir, exist_ok = True)
    os.makedirs(stat_dir, exist_ok = True)

    print(f"Saving models and results to '{dirname}'")

    log_file = open(f"{dirname}/log.txt", "w+")


    for target_name, mols in [("Allosteric_ACE2_Activity", mols)]:

        for foldnum, data in enumerate(train_test_split(mols)):
            train = data[0]
            test = data[1]

            if "Similarity_Balanced" in target_name:
                test = np.array(list(test) + list(sim_balanced_unused_negatives))
            elif "Balanced" in target_name:
                test = np.array(list(test) + list(balanced_unused_negatives))

            #setup y randomization scores into 'randomized_train'
            scores = np.array([mol.score for mol in train])
            np.random.shuffle(scores)
            randomized_train = []
            for i, mol in enumerate(train):
                newmol = ModelingMol.Mol(mol.mol, score = scores[i], identifier = mol.identifier)
                randomized_train.append(newmol)
            old_scores = np.array([mol.score for mol in train])
            new_scores = np.array([mol.score for mol in randomized_train])

            print(test.shape)
            write_sdf(test, filename = f"{fold_dir}/{target_name}_test_for_fold_{foldnum}.sdf")

            test_scores = [m.score for m in test]

            for y_randomize in [False]:
                for descriptor_function, needs_normalization in [(sirmsholder.get_sirms_descriptor, True), (get_all_rdkit_descriptors, True), (get_morgan_descriptor, False), (get_mudra_paper_descriptor, False)]:


                        print("----------------")
                        print(len(train))
                        print(len(test))
                        print(descriptor_function)
                        print("----------------")
                        coverage = ad_coverage(train, test, descriptor_function, k = 1)

                        models = []

                        if descriptor_function != get_mudra_paper_descriptor:

                            '''
                            models.append(("Neural Network", 
                                NeuralNetwork("classification", 
                                    descriptor_function = descriptor_function, 
                                    normalize = needs_normalization)))
                            '''

                            models.append(("Gradient Boosting", 
                                GradientBoosting("classification", 
                                    descriptor_function = descriptor_function, 
                                    normalize = needs_normalization)))

                        if descriptor_function == get_mudra_paper_descriptor:
                            models.append(("MUDRA", MUDRA("classification", 
                                descriptor_function = get_mudra_paper_descriptor, 
                                normalize = False)))

                        for model_name, model in models:

                            if y_randomize:
                                model_filename = f"{model_dir}/{target_name.lower()}_{model_name.replace(' ','').lower()}_{descriptor_function.__name__.replace('_','')}_{foldnum}_yrandomized.pkl"
                                output_target_name = target_name + "_YRandomized"
                                output_model_name = model_name = " (YRandomized)"
                            else:
                                model_filename = f"{model_dir}/{target_name.lower()}_{model_name.replace(' ','').lower()}_{descriptor_function.__name__.replace('_','')}_{foldnum}.pkl"
                                output_target_name = target_name
                                output_model_name = model_name
                            print(model_filename)
                            s = {}
                            s["Target"] = output_target_name
                            s["Model Name"] = output_model_name
                            s["Descriptor Function"] = descriptor_function.__name__
                            s["Fold"] = foldnum
                            s["Train Size"] = len(train)
                            s["Test_Size"] = len(test)
                            s["Coverage"] = coverage

                            if y_randomize:

                                model.fit(randomized_train)
                            else:
                                model.fit(train)

                            pred = model.predict(test)
                            print(output_model_name)
                            model_results = classification_statistics(test_scores, pred)

                            s.update(model_results)
                            s = pd.Series(s)
                            s["Test labels"] = test_scores
                            s["Test predictions"] = pred
                            print(s)
                            results = results.append(s, ignore_index = True)

                            model.save(model_filename)
                            print(results)

    results.to_csv(f"{stat_dir}/all_modeling_results.csv")

    averaged_results, consensus_results = average_results(results, ["Target", "Model Name", "Descriptor Function"])
    averaged_results.to_csv(f"{stat_dir}/all_modeling_results_averaged.csv")
    consensus_results.to_csv(f"{stat_dir}/all_modeling_results_consensus.csv")

def main():

    training_mols = get_training_mols()
    build_models(training_mols, sirms_filename = "data/curated/sirms_descriptors.txt")

if __name__ == "__main__":
    main()

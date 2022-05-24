from utils import run_models_on_set, get_training_mols, get_ncats_screening_mols

training_mols = get_training_mols()

run_models_on_set(model_dirname = "models/example/", external_mols = training_mols, 
        output_dir = "another_test", sirms_filename = "data/curated/sirms_descriptors.txt",
        evaluate_performance = True)

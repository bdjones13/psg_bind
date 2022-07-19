import sys
import pandas as pd
import numpy as np
from io_helpers import parse_arguments, write_to_cache, read_from_cache, build_feature_descriptions, \
    save_summary_results
from preprocess import read_pdb_biopython, get_ligand_data
from extract_features import extract_features_by_description
from itertools import repeat
from multiprocessing import Pool
from ml_model import train_test_correlation


def compute_features(df, pro_elements, pro_ele_rad, directory, lig_elements, lig_ele_rad, feature_descriptions):
    for index, row in df.iterrows():
        pdbid = index
        print(pdbid, flush=True)
        if not any(np.isnan(df.loc[pdbid, :])):  # if nothing to calculate for this complex, don't even read in
            print(pdbid, " done", flush=True)
            continue
        # read in the protein and ligand data
        protein = read_pdb_biopython(pdbid, pro_elements, pro_ele_rad)
        ligand = get_ligand_data(pdbid, directory, lig_elements, lig_ele_rad, "mol2")

        features = []
        for feature in feature_descriptions:
            all_measured = True
            for measurement in feature["measurements"]:
                if np.isnan(df.loc[pdbid, measurement["label"]]):
                    all_measured = False
                    break
            if not all_measured:
                features = features + extract_features_by_description(protein, ligand, feature, pdbid)
            else:
                for measurement in feature["measurements"]:
                    features.append(df.loc[pdbid, measurement["label"]])

        features_np = np.array(features)
        df.loc[pdbid, :] = features_np
        print(pdbid, " done", flush=True)
    return df


def main(arguments):
    use_cache = True
    save_to_cache = True

    MAX_CORES = 32  # to prevent accidentally allocating too many cores by command line
    parsed_arguments = parse_arguments(arguments, MAX_CORES)
    num_cores = parsed_arguments["num_cores"]
    test_count = parsed_arguments["test_count"]

    core_df = pd.read_csv("input/v2007_core_list.csv")
    refined_df = pd.read_csv("input/v2007_refine_list.csv")

    # un-comment to limit tests to fewer proteins
    refined_df = refined_df[0:5]
    core_df = core_df[0:5]

    # combine refined set and core set for calculating. Will be separated later for training
    #  and testing the random forest
    all_df = pd.concat([refined_df, core_df], ignore_index=True)

    # change types
    all_df["id"] = all_df["id"].astype("string")
    core_df["num"] = core_df["num"].astype(float)

    # set constants
    directory = "input/v2007"
    pro_ele_rad = [1.70, 1.55, 1.52, 1.80]
    pro_elements = ["C", "N", "O", "S"]
    lig_elements = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
    lig_ele_rad = [1.70, 1.55, 1.52, 1.80, 1.80, 1.47, 1.75, 1.85, 1.98]  # last is hydrogen when needed: 1.20

    # Get all combinations "C-C", ... "S-I"
    pro_lig_element_pairs = [f"{pro}-{lig}" for pro in pro_elements for lig in lig_elements]

    statistics_list = ["mean", "sum", "max", "SD", "Var", "Sec"]
    feature_descriptions = build_feature_descriptions(pro_lig_element_pairs, statistics_list)
    # label the columns in the dataframe
    column_labels = []
    for feature_description in feature_descriptions:
        for measurement in feature_description["measurements"]:
            column_labels.append(measurement["label"])
    if use_cache:
        all_features = read_from_cache(all_df, column_labels)
    else:
        all_features = pd.DataFrame(np.nan, index=all_df.id, columns=column_labels)

    # process the complexes in parallel
    # if one core, don't do the parallelization step. This makes for easier time profiling
    if num_cores == 1:
        all_features = compute_features(all_features, pro_elements, pro_ele_rad, directory, lig_elements, lig_ele_rad,
                                        feature_descriptions)
    else:
        df_split = np.array_split(all_features, num_cores)
        pool = Pool(num_cores)
        all_features = pd.concat(pool.starmap(compute_features, zip(df_split,
                                                                    repeat(pro_elements),
                                                                    repeat(pro_ele_rad),
                                                                    repeat(directory),
                                                                    repeat(lig_elements),
                                                                    repeat(lig_ele_rad),
                                                                    repeat(feature_descriptions))))
        pool.close()
        pool.join()

    #    use_diff_protein_complex = True
    #    if use_diff_protein_complex:
    #        all_features = diff_protein_complex(all_features)
    #
    if save_to_cache:
        write_to_cache(all_features)

    #    # Use Gradient Boosted Regressor to predict Binding Affinities
    correlations_rmse = pd.DataFrame(index=range(test_count), columns=["corr", "rmse"])
    correlations_rmse_split = np.array_split(correlations_rmse, num_cores)
    pool = Pool(num_cores)
    correlations_rmse = pd.concat(pool.starmap(train_test_correlation, zip(correlations_rmse_split,
                                                                           repeat(all_features),
                                                                           repeat(refined_df),
                                                                           repeat(core_df))))
    pool.close()
    pool.join()

    r_m = np.median(correlations_rmse["corr"])
    r_b = np.max(correlations_rmse["corr"])
    r_a = np.average(correlations_rmse["corr"])
    rmse_m = np.median(correlations_rmse["rmse"])
    rmse_b = np.min(correlations_rmse["rmse"])
    rmse_a = np.average(correlations_rmse["rmse"])
    #
    print(correlations_rmse)

    save_summary_results(r_m, r_b, r_a, rmse_m, rmse_b, rmse_a)
    return all_features


if __name__ == "__main__":
    main(sys.argv)

import os
import pandas as pd
import numpy as np
import datetime


def write_to_cache(feature_df):
    # save the calculated feature values for later use
    fname = f"input/cache/v2007.csv"
    feature_df.to_csv(fname)


def read_from_cache(all_df, column_labels):
    # blank template DataFrame
    skeleton_df = pd.DataFrame(np.nan, index=all_df.id, columns=column_labels)
    skeleton_df.index.name = "id"

    fname = f"input/cache/v2007.csv"
    if os.path.isfile(fname):
        # if file exists, read it in
        temp_df = pd.read_csv(fname, index_col="id")
        temp_skeleton_df = skeleton_df.copy()
        for column in temp_skeleton_df.columns:
            if column in temp_df.columns:
                temp_skeleton_df.loc[temp_skeleton_df.index.isin(temp_df.index), column] = temp_df.loc[:, column]
        # temp_skeleton_df[temp_skeleton_df.index.isin(temp_df.index)] = temp_df
        features = temp_skeleton_df
    else:
        # if file does not exist, make a blank template
        features = skeleton_df.copy()

    return features


def save_summary_results(r_m, r_b, r_a, rmse_m, rmse_b, rmse_a):
    result_list_float = [datetime.datetime.now(), r_m, r_b, r_a, rmse_m, rmse_b, rmse_a]
    result_list_str = [str(f) for f in result_list_float]
    paramstring = ",".join(result_list_str) + "\n"
    print(paramstring)
    with open('results.csv', 'a') as results:
        results.write(paramstring)


def parse_arguments(arguments, MAX_CORES):
    # read in input data from command line arguments
    # returns a dictionary with number of cores, num_cores, and number of
    # times to compute the GBR correlation, test_count.
    # default to 1 core and 1 test
    if len(arguments) == 1:
        parsed_arguments = {"num_cores": 1, "test_count": 1}
    elif len(arguments) == 3:
        parsed_arguments = {"num_cores": min(MAX_CORES, int(arguments[1])), "test_count": int(arguments[2])}
    else:
        raise Exception("invalid number of parameters. Use none to default to 1 core and 3 test\n"
                        "or 2 for number of cores and GBR iterations,\n"
                        "e.g. './main.py 5 50' to use 5 cores and iterate 50 times.")

    return parsed_arguments


def get_basic_feature_descriptions(pro_lig_element_pairs, statistics_list):
    feature_descriptions = []
    cutoff = 12.0
    delta_r = 0.01
    min_r = 0.0
    max_r = 4.0
    for atom_description in pro_lig_element_pairs:
        temp_description = {
            "atom_description": atom_description,
            "cutoff": cutoff,
            "delta_r": delta_r,
            "min_r": min_r,
            "max_r": max_r,
            "filtration_count": int((max_r-min_r)/delta_r),
            "measurements": []
        }
        for statistic in statistics_list:
            temp_description["measurements"].append({
                "dim": 0,
                "statistic": statistic,
                "value": "integral",
            })
        feature_descriptions.append(temp_description)
    return feature_descriptions


def add_feature_label(feature_description):
    element_types = feature_description["atom_description"]
    for measurement in feature_description["measurements"]:
        betti_i = measurement["dim"]
        statistic = measurement["statistic"]
        value = measurement["value"]
        measurement["label"] = f"{element_types}-{statistic}-betti{betti_i}-{value}"
    return


def build_feature_descriptions(pro_lig_element_pairs, statistics_list):
    feature_descriptions = []
    feature_descriptions = feature_descriptions + get_basic_feature_descriptions(pro_lig_element_pairs, statistics_list)

    for feature_description in feature_descriptions:
        add_feature_label(feature_description)

    return feature_descriptions

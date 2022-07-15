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
                temp_skeleton_df.loc[temp_skeleton_df.index.isin(temp_df.index),column] = temp_df.loc[:,column]
        # temp_skeleton_df[temp_skeleton_df.index.isin(temp_df.index)] = temp_df
        features = temp_skeleton_df
    else:
        # if file does not exist, make a blank template
        features = skeleton_df.copy()

    return features


def save_summary_results(r_m, r_b, r_a, rmse_m, rmse_b, rmse_a):
    result_list_float = [datetime.datetime.now(), r_m, r_b,r_a, rmse_m, rmse_b, rmse_a]
    result_list_str = [str(f) for f in result_list_float]
    paramstring = ",".join(result_list_str) + "\n"
    print(paramstring)
    with open('results.csv', 'a') as results:
        results.write(paramstring)


def parse_arguments(arguments, MAX_CORES):
    # read in input data from command line arguments
    # returns a dictionary with number of cores, num_cores, and number of
    # times to compute the GBR correlation, test_count.
    # default to 1 core and 3 tests
    if len(arguments) == 1:
        parsed_arguments = {"num_cores": 1, "test_count": 3}
    elif len(arguments) == 3:
        parsed_arguments = {"num_cores": min(MAX_CORES, int(arguments[1])), "test_count": int(arguments[2])}
    else:
        raise Exception("invalid number of parameters. Use none to default to 1 core and 3 test\n"
                        "or 2 for number of cores and GBR iterations,\n"
                        "e.g. './main.py 5 50' to use 5 cores and iterate 50 times.")

    return parsed_arguments


def get_connectivity_feature_descriptions(pro_lig_element_pairs,kernel, nu, tau):
    feature_descriptions = []
    # connectivity features
    cutoff = 12.0

    # element-specific sum of length of betti-0 bars
    for atom_description in pro_lig_element_pairs:
        temp_description = {
            "atom_description": atom_description,
            "cutoff": cutoff,
            "use_fri": True,  # ?
            "agst": True,
            "tau": tau,
            "nu": nu,
            "kernel": kernel,
            "is_vr": True,
            "measurements": [
                {"dim": 0,
                 "use_bins": False,
                 "metric": "sum_of_length"}
            ]
        }
        feature_descriptions.append(temp_description)

    # sum of length of Betti-0, -1 and -2 of protein, complex. Difference of the two handled after computation of features
    cutoff = 6
    temp_description = {
        "atom_description": "pro",
        "cutoff": cutoff,
        "use_fri": True,  # ?
        "agst": False,
        "tau": tau,
        "nu": nu,
        "kernel": kernel,
        "is_vr": True,
        "measurements": [],
    }
    for i in range(3):
        temp_description["measurements"].append({
           "dim": i,
           "use_bins": False,
           "metric": "sum_of_length"
        })
        temp_description["measurements"].append({
            "dim": i,
            "use_bins": False,
            "metric": "sum_of_birth"
        })
    # feature_descriptions.append(temp_description)

    temp_description = {
        "atom_description": "all",
        "cutoff": cutoff,
        "use_fri": True,  # ?
        "agst": False,
        "tau": tau,
        "nu": nu,
        "kernel": kernel,
        "is_vr": True,
        "measurements": [],
    }
    for i in range(3):
        temp_description["measurements"].append({
           "dim": i,
           "use_bins": False,
           "metric": "sum_of_length"
        })
        temp_description["measurements"].append({
            "dim": i,
            "use_bins": False,
            "metric": "sum_of_birth"
        })
    # feature_descriptions.append(temp_description)

    return feature_descriptions


def get_physical_feature_descriptions(pro_lig_element_pairs, interactions_bins):
    feature_descriptions = []
    cutoff = 12
    for atom_description in pro_lig_element_pairs:
        temp_description = {
            "atom_description": atom_description,
            "cutoff": cutoff,
            "use_fri": False,
            "agst": True,
            "is_vr": True,
            "measurements": [
                {"dim": 0,
                 "use_bins": True,
                 "bins": interactions_bins,
                 "direction": "death",
                 "metric": "count"}
            ]
        }
        feature_descriptions.append(temp_description)

    return feature_descriptions


def get_geometric_feature_descriptions(geometric_bins):
    feature_descriptions = []
    # Summation of Betti-1, Betti-2 bars with birth in each bin interval
    # Table S1 also uses Betti-0
    cutoff = 9
    temp_description = {
        "atom_description": "pro",
        "cutoff": cutoff,
        "is_vr": False,
        "measurements": []
    }
    for i in range(0, 3):
        temp_description["measurements"].append({
            "dim": i,
            "use_bins": True,
            "bins": geometric_bins,
            "direction": "birth",
            "metric": "sum_of_length"})

    feature_descriptions.append(temp_description)
    temp_description = {
        "atom_description": "all",
        "cutoff": cutoff,
        "is_vr": False,
        "measurements": []
    }
    for i in range(0, 3):
        temp_description["measurements"].append({
            "dim": i,
            "use_bins": True,
            "bins": geometric_bins,
            "direction": "birth",
            "metric": "sum_of_length"})

    feature_descriptions.append(temp_description)

    # C/C from table S1 or fig 8, only betti-1 and betti-2
    temp_description = {
        "atom_description": "C-C",
        "cutoff": cutoff,
        "is_vr": False,
        "measurements": []
    }
    for i in range(1, 3):
        temp_description["measurements"].append({
            "dim": i,
            "use_bins": True,
            "bins": geometric_bins,
            "direction": "birth",
            "metric": "sum_of_length"}) # TODO: is this the correct metric? "of the third category" = binned sum of lengths
            # TODO: Table S1 says not binned, table 1 says binned. check if corr. coeffs line up
    feature_descriptions.append(temp_description)

    # C/C unbinned
    temp_description = {
        "atom_description": "C-C",
        "cutoff": cutoff,
        "is_vr": False,
        "measurements": []
    }
    for i in range(1, 3):
        temp_description["measurements"].append({
            "dim": i,
            "use_bins": False,
            "direction": "birth",
            "metric": "sum_of_length"})
    feature_descriptions.append(temp_description)


    return feature_descriptions


def add_feature_label(feature_description):
    element_types = feature_description["atom_description"]
    if feature_description["is_vr"]:
        simplicial_complex = "VR"
        if feature_description["use_fri"]:
            if feature_description["kernel"] == "E":
                method = "FRIExp"
            else:
                method = "FRI"
            distance_function = f"""{method}({feature_description["nu"]},{feature_description["tau"]})"""
            if feature_description["agst"]:
                distance_function = distance_function + "agst"
        else:
            distance_function = "EUC"
            if feature_description["agst"]:
                distance_function = distance_function + "agst"
    else:
        simplicial_complex = "Alpha"
        distance_function = "EUC"

    for measurement in feature_description["measurements"]:
        betti_i = measurement["dim"]
        metric = measurement["metric"]
        measurement["label"] = f"{element_types}-{distance_function}-{simplicial_complex}-Betti{betti_i}-{metric}"

    return



def build_feature_descriptions(pro_lig_element_pairs, interactions_bins, geometric_bins):
    feature_descriptions = []
    FRI_paramsets = [
        ["E",1,1], # [kernel, nu, tau]
        ["L",3,2],
        ["L",3,1],
        ["L",3,0.5],
        ["L",5,1],
        ["L",2,1]]
    # for p in FRI_paramsets:
        # feature_descriptions = feature_descriptions + get_connectivity_feature_descriptions(pro_lig_element_pairs,p[0],p[1],p[2])
    ## feature_descriptions = feature_descriptions + get_connectivity_feature_descriptions(pro_lig_element_pairs,kernel,nu,tau)
    # feature_descriptions = feature_descriptions + get_physical_feature_descriptions(pro_lig_element_pairs,interactions_bins)
    feature_descriptions = feature_descriptions + get_geometric_feature_descriptions(geometric_bins)
    for feature_description in feature_descriptions:
        add_feature_label(feature_description)
    return feature_descriptions
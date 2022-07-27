import pandas as pd
from ml_model import train_test_correlation
from itertools import repeat
import numpy as np
from multiprocessing import Pool
import re
def train_test(all_features,refined_df, core_df, num_cores, test_count):
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
    group_statistics = [r_m, r_b, r_a, rmse_m, rmse_b, rmse_a]
    return group_statistics


# compute correlations by group using the regressor

pro_elements = ["C", "N", "O", "S"]
lig_elements = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
pro_lig_element_pairs = [f"{pro}-{lig}" for pro in pro_elements for lig in lig_elements]


def get_seperate_statistic_groups(all_columns,keywords):
    feature_groups = []
    for key in keywords:
        group = [column for column in all_columns if key in column]
        # r = re.compile(f".*{key}*")
        # group = list(filter(r.match,all_columns))
        feature_groups.append(group)
    return feature_groups 


def get_ESPH_EUCagst_VR_Betti0_groups():
    feature_groups = []
    physical_bins_count = 6


    for pair in pro_lig_element_pairs:
        group = []
        for i in range(physical_bins_count):
            group.append(f"""{pair}-EUCagst-VR-Betti0-countbin{i}""")
        feature_groups.append(group)

    esph_betti0_group = []
    for group in feature_groups:
        for feature in group:
            esph_betti0_group.append(feature)

    feature_groups.insert(0,esph_betti0_group)
    return feature_groups

def get_connectivity_groups():
    feature_groups = []
    FRI_paramsets = [
        ["E", 1, 1],  # [kernel, nu, tau]
        ["L", 3, 2],
        ["L", 3, 1],
        ["L", 3, 0.5],
        ["L", 5, 1],
        ["L", 2, 1]]
    for p in FRI_paramsets:
        group = []
        method = "FRIExp" if p[0] == "E" else "FRI"
        for pair in pro_lig_element_pairs:
            group.append(f"""{pair}-{method}({p[1]},{p[2]})agst-VR-Betti0-sum_of_length""")
        feature_groups.append(group)
    return feature_groups

def get_geometric_groups():
    # possibly need unbinned
    feature_groups = []
    geometric_bins_count = 6
    group = []

    # note: Betti 1 and Betti 2 all in same group for C/C
    # binned
    for i in range(1,3):
        base = f"C-C-EUC-Alpha-Betti{i}-sum_of_length"
        for j in range(geometric_bins_count):
            group.append(f"{base}bin{j}")
    feature_groups.append(group)

    # C/C unbinned
    group = []
    for i in range(1,3):
        base = f"C-C-EUC-Alpha-Betti{i}-sum_of_length"
        group.append(base)
    feature_groups.append(group)

    # pro-EUC-Alpha-Betti0-sum_of_lengthbin0,.... all-EUC-Alpha-Betti2-sum_of_lengthbin5
    group = []
    for i in range(0,3):
        base = f"EUC-Alpha-Betti{i}-sum_of_length"
        for j in range(geometric_bins_count):
            #group.append(f"pro-{base}bin{j}")
            group.append(f"all-{base}bin{j}")
    feature_groups.append(group)
    return feature_groups

# TODO: Non element specific FRI(t=2,v=3) betti0,1,2 Not binned
#feature_groups = get_ESPH_EUCagst_VR_Betti0_groups()
#feature_groups = feature_groups + get_connectivity_groups()
#feature_groups = feature_groups + get_geometric_groups()
# feature_groups = get_geometric_groups()

core_df = pd.read_csv("input/v2007_core_list.csv")
refined_df = pd.read_csv("input/v2007_refine_list.csv")
#core_df = core_df[0:2]
#refined_df = refined_df[0:2]
all_df = pd.concat([refined_df, core_df], ignore_index=True)
# change types
all_df["id"] = all_df["id"].astype("string")
core_df["num"] = core_df["num"].astype(float)

features_df = pd.read_csv("input/cache/v2007.csv",index_col="id")

statistics_list = ["mean", "sum", "max", "SD", "Var", "Sec","Top"]

feature_groups = get_seperate_statistic_groups(features_df.columns,statistics_list)
for group in feature_groups:
    print("group: ",group)


# core_df = core_df[0:10]
# refined_df = refined_df[0:10]

num_cores = 25
test_count = 25

outfile = "output/feature_analysis.csv"
results = []
print("about to train and test")
group_statistics = train_test(features_df, refined_df, core_df, num_cores, test_count)
results.append(["all",group_statistics[0],group_statistics[3]])
for group in feature_groups:
    group_df = features_df.loc[:,group]
    group_statistics = train_test(group_df, refined_df, core_df, num_cores, test_count)
    group_label = ";".join(group)
    results.append([group_label,group_statistics[0],group_statistics[3]])
    print(group_label, group_statistics,flush=True)
df = pd.DataFrame(results,columns=["label", "r_m","rmse_m"])
df = df.set_index("label")
df.to_csv(outfile)

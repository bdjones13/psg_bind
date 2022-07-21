# to investigate the distribution of distances between protein atoms and ligand atoms.
# This will help substantiate a choice of filtration value in PSG
# with randstate 0, sample size 250:
#   shape: very roughly normal
#   minimum: 0.316
#   mean: 22.38
#   std: 8.044

# Histogram roughly supports trying filtration with differing dr values changing
#  at around 10 and at around 20
# 1 <= r < 10, dr = 0.1
# 10 <= r < 20 dr = 0.5
# 20 <= r <= 40 dr = 1.0
import pandas as pd
import numpy as np
import math
from preprocess import read_pdb_biopython, get_ligand_data
import matplotlib.pyplot as plt

core_df = pd.read_csv("input/v2007_core_list.csv")
refined_df = pd.read_csv("input/v2007_refine_list.csv")


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

distances = []

cutoff = 40
cutoff2 = math.pow(cutoff,2)
# By running this experiment without a cutoff, can see there are relatively very few
# distances over 40, and these interactions would be much weaker

sampled = all_df["id"].sample(n=250,random_state=0)

for pdbid in sampled:
    this_distances = []
    print(pdbid, flush=True)
    # read in the protein and ligand data
    protein = read_pdb_biopython(pdbid, pro_elements, pro_ele_rad)
    ligand = get_ligand_data(pdbid, directory, lig_elements, lig_ele_rad, "mol2")
    # distances = distances + [dist(k,l) for k in protein for l in ligand]
    for k in protein:
        for l in ligand:
            dist2 = math.pow(k[0] - l[0],2) \
                + math.pow(k[1] - l[1],2) \
                + math.pow(k[2] - l[2], 2)
            if dist2 < cutoff2:
                dist = math.sqrt(dist2)
                distances.append(dist)
print("min distance: ", min(distances))
print("mean distance: ", np.mean(distances))
print("standard dev distance: ", np.std(distances))
plt.hist(distances)
plt.show()
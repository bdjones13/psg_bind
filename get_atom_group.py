import numpy as np


# noinspection PyPep8Naming
def get_atom_group(description, cutoff, protein, ligand):
    c2 = np.power(cutoff, 2.0)
    P_list = []  # start as list, then convert to numpy array for speed

    if description in ["all", "pro"]:
        X = protein
        Y = ligand
    else:
        elements = description.split('-')

        pro_element = elements[0]
        lig_element = elements[1]
        # TODO: time this subsetting to determine if fewer calls will make an efficiency difference
        X = protein[protein[:, 4] == pro_element]
        Y = ligand[ligand[:, 4] == lig_element]

    for a in X:
        for b in Y:
            # no need to take square root for comparison to cutoff
            dist2 = (np.power((a[0] - b[0]), 2.0)
                     + np.power((a[1] - b[1]), 2.0)
                     + np.power((a[2] - b[2]), 2.0))
            if dist2 > c2:
                continue
            P_list.append([a[0], a[1], a[2], a[3], 1])  # TODO: use booleans instead of 0 and 1 as flags in numpy array
            break  # only need to be within cutoff of 1 ligand atom

    if description != "pro":
        for b in Y:
            P_list.append([b[0], b[1], b[2], b[3], 0])

    P_array = np.array(P_list)
    return P_array

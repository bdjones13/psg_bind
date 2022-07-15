import pandas as pd
import os
import numpy as np
import math
from get_atom_group import get_atom_group
#from distance_matrix import distance_matrix
#from get_tf import get_tf


class TF_Bin:

    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum
        self.bars = set()

    def should_contain(self, bar, direction):
        if direction == "birth" and (self.minimum <= bar.birth <= self.maximum) and not math.isinf(bar.death):
            return True
        elif direction == "death" and (self.minimum < bar.death <= self.maximum):
            return True
        else:
            return False

    def add_bar(self, bar):
        self.bars.add(bar)

    def count(self):
        return len(self.bars)

    def sum_of_lengths(self):
        cumulative = 0.0
        for tf_bar in self.bars:
            cumulative = cumulative + (tf_bar.death - tf_bar.birth)
        return cumulative


def sum_betti_lengths(dgms, i):
    # sum of the lengths of the betti-i bars
    cumulative = 0.0
    dgm = dgms[i]
    for p in dgm:
        if p.death == np.inf:
            continue
        cumulative = cumulative + (p.death - p.birth)
    return cumulative

def sum_betti_births(dgms, i):
    # sum of the births of the betti-i bars
    cumulative = 0.0
    dgm = dgms[i]
    for p in dgm:
        if p.death == np.inf:
            continue
        cumulative = cumulative + p.birth
    return cumulative

def get_binned(dgms, i, bin_bounds, direction):
    dgm = dgms[i]

    # transform bin boundaries into object of class TF_Bin
    tf_bins = [TF_Bin(bounds[0], bounds[1]) for bounds in bin_bounds]

    # perform the binning
    for p in dgm:
        for tf_bin in tf_bins:
            if tf_bin.should_contain(p, direction):
                tf_bin.add_bar(p)
                break

    return tf_bins

def atom_group_to_xyz(P,filename):
    xyz = P[:,0:3]
    np.savetxt(filename,xyz)
    return 

def get_spectrum(filename):
    with open(filename) as fp:
        line_lists = []
        for line in fp.readlines():
            line_list = line.split()  # turn into list
            line_list = [round(float(line_elt),5) for line_elt in line_list]  # convert str to float
            line_lists.append(line_list)

    # make uniform length by right-padding with NaN
    max_len = max([len(line_list) for line_list in line_lists])
    for line_list in line_lists:
        while len(line_list) < max_len:  #TODO: would be more efficient to calculate number of NaN then do a for loop
            line_list.append(np.nan)

    spectrum_df = pd.DataFrame(line_lists)
    return spectrum_df
        

def get_spectra():
    directory = "test/one_complex/"
    filenames = ["snapshots_vertex.txt", "snapshots_edge.txt", "snapshots_facet.txt"]
    filenames = [directory + filename for filename in filenames]
    spectra = []

    for filename in filenames:
        spectrum_df = get_spectrum(filename)
        spectra.append(spectrum_df)
    return spectra 


def get_spectrum_statistic(spectrum,alpha):
    if alpha == "mean":
        return spectrum.mean(axis=1, skipna=True)
    elif alpha == "sum":
        return spectrum.sum(axis=1, skipna=True)
    elif alpha == "max":
        return spectrum.max(axis=1,skipna=True)
    elif alpha == "SD":
        print("placeholder")
    elif alpha == "Var":
        print("placeholder")
    elif alpha == "Sec":
        # min nonzero element in each row
        # apply to each row a function that gets the first nonzero element
        spectrum.apply(lambda x: x.iloc[x.to_numpy().nonzero() [0][0]], axis=1)
    elif alpha == "Top":
        print("placeholder")
        # count of zero elements in each row
    else:
        raise Exception("invalid spectrum statistic")

def get_area_under_plot(persistent_statistic, delta_r):
    cumulative = 0.0
    for statistic in persistent_statistic:
        if not np.isnan(statistic):
            cumulative = cumulative + statistic * delta_r
    return cumulative

def extract_feature(protein, ligand, feature):
    P = get_atom_group(feature["atom_description"], feature["cutoff"],protein,ligand)
    #TODO: write to .xyz file and call HERMES
    xyz_filename = "temp.xyz"
    filtration_filename = "test/hermes_example/filtration.txt"
    atom_group_to_xyz(P,xyz_filename)
    # os.system(f"hermes {xyz_filename} {filtration_filename} 100 0.4") 
    spectra = get_spectra()

    alpha = "mean" # mean, sum, max, SD, variance of eigen, Sec= lambda_2
    i = 1
    persistent_statistic = get_spectrum_statistic(spectra[i],alpha)
    delta_r = 0.01
    area = get_area_under_plot(persistent_statistic,delta_r)
    return [area]
    # Get topological fingerprint
#    if feature["is_vr"]:
#        # Get distance matrix
#        if feature["use_fri"]:
#            d = distance_matrix(P, use_fri=True, agst=feature["agst"], tau=feature["tau"], nu=feature["nu"], kernel=feature["kernel"])
#            dgms = get_tf(d, is_vr=True, dim=3, cutoff=2.0)  # using a cutoff for Dionysus
#        else:
#            d = distance_matrix(P, use_fri=False, agst=feature["agst"])
#            dgms = get_tf(d, is_vr=True, dim=3, cutoff=feature["cutoff"])
#    else:
#        dgms = get_tf(P[:, 0:3], is_vr=False) # no need for dim
#
#    # compute the features
#    measurements = []
#    for measurement in feature["measurements"]:
#        if measurement["use_bins"]:
#            binned = get_binned(dgms, measurement["dim"], measurement["bins"], measurement["direction"])
#            if measurement["metric"] == "count":
#                measurements = measurements + [tf_bin.count() for tf_bin in binned]
#            elif measurement["metric"] == "sum_of_length":
#                measurements = measurements + [tf_bin.sum_of_lengths() for tf_bin in binned]
#            else:
#                raise Exception("invalid measurement: ", measurement)
#        else:
#            if measurement["metric"] == "sum_of_length":
#                measurements = measurements + [sum_betti_lengths(dgms, measurement["dim"])]
#            elif measurement["metric"] == "sum_of_birth":
#                measurements = measurements + [sum_betti_births(dgms, measurement["dim"])]
#            else:
#                raise Exception("invalid measurement: ", measurement)
#
#    return measurements

def extract_features_by_description(protein, ligand, feature):
    # features = []
    # for feature in feature_descriptions:
    
    return extract_feature(protein, ligand, feature)


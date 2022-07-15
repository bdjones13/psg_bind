import numpy as np
import math
from get_atom_group import get_atom_group
from get_spectra import get_spectra


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

def get_sec(x):
    nonzero = x.to_numpy().nonzero()
    if len(nonzero[0]) == 0:
        return np.nan
    return x.iloc[nonzero[0][0]]

def get_spectrum_statistic(spectrum,alpha):
    # TODO: is this over all eigenvalues or only the nonzero?
    if alpha == "mean":
        return spectrum.mean(axis=1, skipna=True)
    elif alpha == "sum":
        return spectrum.sum(axis=1, skipna=True)
    elif alpha == "max":
        return spectrum.max(axis=1,skipna=True)
    elif alpha == "SD":
        return spectrum.std(axis=1,skipna=True)
    elif alpha == "Var":
        return spectrum.var(axis=1,skipna=True)
    elif alpha == "Sec":
        # min nonzero element in each row
        # apply to each row a function that gets the first nonzero element
        return spectrum.apply(lambda x: get_sec(x),axis=1)
        # spectrum.apply(lambda x: x.iloc[x.to_numpy().nonzero() [0][0]], axis=1)
    elif alpha == "Top":
        return -2
        # count of zero elements in each row
    else:
        raise Exception("invalid spectrum statistic")

def get_area_under_plot(persistent_statistic, delta_r):
    cumulative = 0.0
    for statistic in persistent_statistic:
        if not np.isnan(statistic):
            cumulative = cumulative + statistic * delta_r
    return cumulative


def extract_feature(protein, ligand, feature, pdbid):
    P = get_atom_group(feature["atom_description"], feature["cutoff"],protein,ligand)
    measurements = []
    # if no atoms in group, return all 0
    if len(P) == 0:
        for measurement in feature["measurements"]:
            measurements.append(0)
        return measurements

    spectra = get_spectra(P, pdbid)

    measurements = []
    for measurement in feature["measurements"]:
        persistent_statistic = get_spectrum_statistic(spectra[measurement["dim"]],measurement["statistic"])
        if measurement["value"] == "integral":
            delta_r = 0.01
            area = get_area_under_plot(persistent_statistic,delta_r)
        else:
            raise Exception("invalid measurement value (use 'integral').")
        measurements.append(area)
    return measurements
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

def extract_features_by_description(protein, ligand, feature, pdbid):
    # features = []
    # for feature in feature_descriptions:
    
    return extract_feature(protein, ligand, feature, pdbid)


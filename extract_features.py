import numpy as np
# import math
from get_atom_group import get_atom_group
from get_spectra import get_spectra, get_persistent_betti_small_points


# class TF_Bin:
#
#     def __init__(self, minimum, maximum):
#         self.minimum = minimum
#         self.maximum = maximum
#         self.observations = set()
#
#     def should_contain(self, observation):
#         if direction == "birth" and (self.minimum <= bar.birth <= self.maximum) and not math.isinf(bar.death):
#             return True
#         elif direction == "death" and (self.minimum < bar.death <= self.maximum):
#             return True
#         else:
#             return False
#
#     def add_bar(self, bar):
#         self.bars.add(bar)
#
#     def count(self):
#         return len(self.bars)
#
#     def sum_of_lengths(self):
#         cumulative = 0.0
#         for tf_bar in self.bars:
#             cumulative = cumulative + (tf_bar.death - tf_bar.birth)
#         return cumulative
#
#
# def sum_betti_lengths(dgms, i):
#     # sum of the lengths of the betti-i bars
#     cumulative = 0.0
#     dgm = dgms[i]
#     for p in dgm:
#         if p.death == np.inf:
#             continue
#         cumulative = cumulative + (p.death - p.birth)
#     return cumulative
#
# def sum_betti_births(dgms, i):
#     # sum of the births of the betti-i bars
#     cumulative = 0.0
#     dgm = dgms[i]
#     for p in dgm:
#         if p.death == np.inf:
#             continue
#         cumulative = cumulative + p.birth
#     return cumulative
#
# def get_binned(harmonic_spectrum, bin_bounds):
#     bin_indicies = np.digitize(harmonic_spectrum, bin_bounds)
#     with_bins = np.concatenate([harmonic_spectrum,bin_indicies],axis=1)
#     return with_bins

def get_sec(x):
    nonzero = x.to_numpy().nonzero()
    if len(nonzero[0]) == 0:
        return np.nan
    return x.iloc[nonzero[0][0]]


def get_spectrum_statistic(spectrum, alpha):
    # From PSG paper:
    #       "To verify our hypothesis, we compute the summation, mean, maximal, SD, variance of its eigenvalues,
    #       and \left(\tilde \lambda_2\right)_0^{r+0} of the persistent spectra of L^{r + 0}_0 over various
    #       filtration radii r"
    # so these mean, max, etc. are over the persistent spectra, which includes both harmonic and non-harmonic
    if alpha == "mean":
        return spectrum.mean(axis=1, skipna=True)
    elif alpha == "sum":
        return spectrum.sum(axis=1, skipna=True)
    elif alpha == "max":
        return spectrum.max(axis=1, skipna=True)
    elif alpha == "SD":
        return spectrum.std(axis=1, skipna=True)
    elif alpha == "Var":
        return spectrum.var(axis=1, skipna=True)
    elif alpha == "Sec":
        # min nonzero element in each row
        # apply to each row a function that gets the first nonzero element
        return spectrum.apply(lambda x: get_sec(x), axis=1)
    elif alpha == "Top":
        # count of zero elements in each row
        # get boolean index matrix, cast to int (1 if 0 value, else 0), and count by row
        # nonzero values -> False -> 0 -> do not count to sum
        # zero values -> True -> 1 -> do count to sum
        harmonic_spectrum = (spectrum == 0).astype(int).sum(axis=1)
        return list(harmonic_spectrum.values)
    else:
        raise Exception("invalid spectrum statistic")


def get_area_under_plot(persistent_statistic, filtration):
    cumulative = 0.0
    for i in range(len(persistent_statistic)-1): # -1 to not go over
        statistic = persistent_statistic.iloc[i]
        if not np.isnan(statistic):
            delta_r = filtration[i+1]-filtration[i]
            cumulative = cumulative + statistic * delta_r
    # for statistic in persistent_statistic:
    #     if not np.isnan(statistic):
    #         cumulative = cumulative + statistic * delta_r
    return cumulative


def extract_feature(protein, ligand, feature, pdbid):
    P = get_atom_group(feature["atom_description"], feature["cutoff"], protein, ligand)
    measurements = []
    # if no atoms in group, return all 0
    if len(P) == 0:
        for measurement in feature["measurements"]:
            if measurement["statistic"] == "Top":
                for _ in feature["filtration_r"]:
                    measurements.append(0)
            else:
                measurements.append(0)
        return measurements
    if len(P) <= 3:
        for measurement in feature["measurements"]:
            if measurement["statistic"] != "Top":
                # alpha complex not defined for <= 3 points, so non-harmonic specra all 0.
                measurements.append(0)
            else:
                persistent_betti = get_persistent_betti_small_points(P, feature["filtration_r"])
                measurements = measurements + persistent_betti[measurement["dim"]]
        return measurements

    print(f"""{pdbid}: get spectra {feature["atom_description"]}""", flush=True)
    spectra = get_spectra(P, pdbid,feature["filtration_r"],feature["alpha_filtration"])

    measurements = []
    for measurement in feature["measurements"]:
        persistent_statistic = get_spectrum_statistic(spectra[measurement["dim"]], measurement["statistic"])
        if measurement["statistic"] == "Top":
            measurements = measurements + persistent_statistic
            print("top")
        elif measurement["value"] == "integral":
            area = get_area_under_plot(persistent_statistic, feature["filtration_r"])
            measurements.append(area)
        else:
            raise Exception("invalid measurement value (use 'integral').")

    return measurements


def extract_features_by_description(protein, ligand, feature, pdbid):
    return extract_feature(protein, ligand, feature, pdbid)

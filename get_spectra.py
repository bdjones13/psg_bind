import math

import numpy as np
import shutil
import os
import csv
import pandas as pd


def atom_group_to_xyz(P, filename):
    xyz = P[:, 0:3]
    np.savetxt(filename, xyz)
    return


def get_spectrum(filename):
    TOL = 0.001
    with open(filename) as fp:
        line_lists = []
        for line in fp.readlines():
            line_list = line.split()  # turn into list
            line_list = [float(line_elt) for line_elt in line_list]  # convert str to float
            line_list = [0.0 if x < TOL else x for x in line_list]
            line_lists.append(line_list)

    # make uniform length by right-padding with NaN
    max_len = len(line_lists[-1])  # longest line should be the last
    for line_list in line_lists:
        amount_to_add = max_len - len(line_list)
        line_list.extend([np.nan]*amount_to_add)

    spectrum_df = pd.DataFrame(line_lists)
    return spectrum_df


def read_spectra(pdbid, atom_description):
    directory = f"temp/{pdbid}/{atom_description}"
    os.chdir(directory)
    filenames = ["snapshots_vertex.txt", "snapshots_edge.txt", "snapshots_facet.txt"]
    spectra = []

    for filename in filenames:
        spectrum_df = get_spectrum(filename)
        spectra.append(spectrum_df)
    os.chdir("../../../")
    return spectra

def generate_spectra(P,pdbid,alpha_filtration, atom_description):
    directory = f"temp/{pdbid}/{atom_description}"

    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    os.chdir(directory)

    filtration_filename = "filtration.txt"
    with open(filtration_filename, 'w') as f:
        write = csv.writer(f, delimiter=' ')
        write.writerow(alpha_filtration)
    xyz_filename = f"{pdbid}-{atom_description}.xyz"
    atom_group_to_xyz(P, xyz_filename)

    # TODO: replace with call to HERMES
    #shutil.copy("../../../test/one_complex/snapshots_vertex.txt", "snapshots_vertex.txt")
    #shutil.copy("../../../test/one_complex/snapshots_edge.txt", "snapshots_edge.txt")
    #shutil.copy("../../../test/one_complex/snapshots_facet.txt", "snapshots_facet.txt")
    os.system(f"hermes {xyz_filename} {filtration_filename} 100 0 > hermes_output.txt")
    os.chdir("../../..")

def spectra_exists(pdbid, atom_description):
    return os.path.isdir(f"temp/{pdbid}/{atom_description}")

def get_spectra(P, pdbid, filtration_r, alpha_filtration, reuse_spectra,atom_description):
    # make temporary directory, call HERMES, read in spectra, and delete the temporary files
    if not reuse_spectra or not spectra_exists(pdbid, atom_description):
        generate_spectra(P,pdbid, alpha_filtration,atom_description)

    spectra = read_spectra(pdbid, atom_description)
    for spectrum in spectra:
        spectrum["r"] = filtration_r
        spectrum.set_index(["r"],inplace=True)

    return spectra


def get_persistent_betti_small_points(P, filtration_r):
    # need 3 series of length filtration_count, betti0, betti1, betti2. Betti1 and 2 both all zero.
    filtration_count = len(filtration_r)
    betti_1 = np.zeros((filtration_count,))
    betti_2 = np.zeros((filtration_count,))
    if len(P) == 1:
        betti_0 = np.ones((filtration_count,))
    elif len(P) == 2:
        # number components = 2 until r = distance, then 1
        a = P[0,:]
        b = P[1,:]
        dist = np.sqrt(
            np.power(a[0] - b[0], 2)
            + np.power(a[1] - b[1], 2)
            + np.power(a[2] - b[2], 2)
        )
        betti_0 = np.ones((filtration_count,))
        for i in range(filtration_count):
            r = filtration_r[i]
            if r < dist:
                betti_0[i] = 2
            else:
                break
        # dist_steps = int((dist-filtration[0])/delta_r)
        # components_2 = 2*np.ones((dist_steps,))
        # components_1 = np.ones((filtration_count-dist_steps,))
        # betti_0 = np.concatenate([components_2, components_1], axis=0)
    elif len(P) == 3:
        # number components = 3 until r = min(distance, then 1
        a = P[0, :]
        b = P[1, :]
        c = P[2, :]
        dist1 = np.sqrt(
            np.power(a[0] - b[0], 2)
            + np.power(a[1] - b[1], 2)
            + np.power(a[2] - b[2], 2)
        )
        dist2 = np.sqrt(
            np.power(a[0] - c[0], 2)
            + np.power(a[1] - c[1], 2)
            + np.power(a[2] - c[2], 2)
        )
        dist3 = np.sqrt(
            np.power(b[0] - c[0], 2)
            + np.power(b[1] - c[1], 2)
            + np.power(b[2] - c[2], 2)
        )
        distances = sorted([dist1,dist2,dist3])
        betti_0 = np.ones((filtration_count,))
        for i in range(filtration_count):
            r = filtration_r[i]
            if r < distances[0]:
                betti_0[i] = 3
            elif r < distances[1]:
                betti_0[i] = 2
            else:
                break


        # dist_steps1 = int((distances[0] - min_r) / delta_r)
        # dist_steps2 = int((distances[1] - distances[0]) / delta_r)
        #
        # components_3 = 3 * np.ones((dist_steps1,))
        # components_2 = 2 * np.ones((dist_steps2,))
        # components_1 = np.ones((filtration_count - dist_steps2,))
        # betti_0 = np.concatenate([components_3, components_2, components_1], axis=0)
    else:
        raise Exception(f"get_persistent_betti_small_points called with incorrect number of points: {len(P)}")
    return [list(betti_0),list(betti_1),list(betti_2)]

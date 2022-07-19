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
    with open(filename) as fp:
        line_lists = []
        for line in fp.readlines():
            line_list = line.split()  # turn into list
            line_list = [round(float(line_elt), 5) for line_elt in line_list]  # convert str to float
            line_lists.append(line_list)

    # make uniform length by right-padding with NaN
    max_len = max([len(line_list) for line_list in line_lists])
    for line_list in line_lists:
        while len(line_list) < max_len:  # TODO: would be more efficient to calculate number of NaN then do a for loop
            line_list.append(np.nan)

    spectrum_df = pd.DataFrame(line_lists)
    return spectrum_df


def read_spectra():
    filenames = ["snapshots_vertex.txt", "snapshots_edge.txt", "snapshots_facet.txt"]
    filenames = [filename for filename in filenames]
    spectra = []

    for filename in filenames:
        spectrum_df = get_spectrum(filename)
        spectra.append(spectrum_df)
    return spectra


def get_spectra(P, pdbid, delta_r, min_r, filtration_count):
    # make temporary directory, call HERMES, read in spectra, and delete the temporary files
    if os.path.isdir(f"temp/{pdbid}"):
        shutil.rmtree(f"temp/{pdbid}")

    os.makedirs(f"temp/{pdbid}")
    os.chdir(f"temp/{pdbid}")

    # setup input for HERMES
    filtration = [min_r + delta_r * i for i in range(filtration_count)]
    filtration_filename = "filtration.txt"
    with open(filtration_filename, 'w') as f:
        write = csv.writer(f, delimiter=' ')
        write.writerow(filtration)
    xyz_filename = f"{pdbid}.xyz"
    atom_group_to_xyz(P, xyz_filename)

    # TODO: replace with call to HERMES
    shutil.copy("../../test/one_complex/snapshots_vertex.txt", "snapshots_vertex.txt")
    shutil.copy("../../test/one_complex/snapshots_edge.txt", "snapshots_edge.txt")
    shutil.copy("../../test/one_complex/snapshots_facet.txt", "snapshots_facet.txt")
    # os.system(f"hermes {xyz_filename} {filtration_filename} 100 {delta_r} > hermes_output.txt")
    spectra = read_spectra()

    # clean up
    os.chdir("../..")
    shutil.rmtree(f"temp/{pdbid}")

    return spectra


def get_persistent_betti_small_points(P, delta_r, min_r, filtration_count):
    # need 3 series of length filtration_count, betti0, betti1, betti2. Betti1 and 2 both all zero.
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
        dist_steps = int((dist-min_r)/delta_r)
        components_2 = 2*np.ones((dist_steps,))
        components_1 = np.ones((filtration_count-dist_steps,))
        betti_0 = np.concatenate([components_2, components_1], axis=0)
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

        dist_steps1 = int((distances[0] - min_r) / delta_r)
        dist_steps2 = int((distances[1] - distances[0]) / delta_r)

        components_3 = 3 * np.ones((dist_steps1,))
        components_2 = 2 * np.ones((dist_steps2,))
        components_1 = np.ones((filtration_count - dist_steps2,))
        betti_0 = np.concatenate([components_3, components_2, components_1], axis=0)
    else:
        raise Exception(f"get_persistent_betti_small_points called with incorrect number of points: {len(P)}")
    return [betti_0,betti_1,betti_2]
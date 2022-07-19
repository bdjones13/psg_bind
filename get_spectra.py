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
    print(f"{pdbid}: get spectra",flush=True)
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
    # os.system(f"hermes {xyz_filename} {filtration_filename} 100 {delta_r}")
    spectra = read_spectra()

    # clean up
    os.chdir("../..")
    shutil.rmtree(f"temp/{pdbid}")

    return spectra

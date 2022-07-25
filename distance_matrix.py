import numpy as np


def FRI(phi_func, atom_i, atom_j, tau, nu):
    eta = tau * (atom_i[3] + atom_j[3])  # atom_k[3] = r_k
    dist = atom_distance(atom_i, atom_j)
    phi_val = phi_func(dist, eta, nu)
    return 1 - phi_val


def FRI_agst(phi_func, atom_i, atom_j, tau, nu, d_inf):
    if atom_i[4] == atom_j[4]:
        return d_inf
    return FRI(phi_func, atom_i, atom_j, tau, nu)


def atom_distance(a, b):
    # also known as EUC
    return np.sqrt(np.power((a[0] - b[0]), 2.0)
                   + np.power((a[1] - b[1]), 2.0)
                   + np.power((a[2] - b[2]), 2.0))


def atom_distance_against(atom_i, atom_j, d_inf):
    if atom_i[4] == atom_j[4]:
        return d_inf
    return atom_distance(atom_i, atom_j)


def phi_lorentz(dist, eta, beta):
    # kernel method with Lorentz function
    return 1 / (1 + np.power((dist / eta), beta))


def phi_exp(dist, eta, beta):
    # kernel method with exponential function
    return np.exp(-np.power((dist / eta), beta))


def get_corr_function(use_fri, agst, tau, nu, kernel, d_inf):
    if use_fri:
        if kernel == 'L':
            phi = phi_lorentz
        else:  # kernel == 'E'
            phi = phi_exp
        if agst:
            def corr_function(atom_i, atom_j):
                return FRI_agst(phi, atom_i, atom_j, tau, nu, d_inf)
            # corr_function = lambda atom_i, atom_j: FRI_agst(phi, atom_i, atom_j, tau, nu, d_inf)
        else:
            def corr_function(atom_i, atom_j):
                return FRI(phi, atom_i, atom_j, tau, nu)
    #         corr_function = lambda atom_i, atom_j: FRI(phi, atom_i, atom_j, tau, nu)
    else:
        if agst:
            def corr_function(atom_i, atom_j):
                return atom_distance_against(atom_i, atom_j, d_inf)
            # corr_function = lambda atom_i, atom_j: atom_distance_against(atom_i, atom_j, d_inf)
        else:
            def corr_function(atom_i, atom_j):
                return atom_distance(atom_i, atom_j)
            # corr_function = lambda atom_i, atom_j: atom_distance(atom_i, atom_j)

    return corr_function


def distance_matrix(P, use_fri, agst, tau=None, nu=None, kernel=None):
    # d_inf = np.inf
    d_inf = 100
    num_atoms = len(P)
    d = np.zeros((num_atoms, num_atoms))  # TODO: consider starting this array filled with np.inf

    corr_function = get_corr_function(use_fri, agst, tau, nu, kernel, d_inf)

    for i in range(num_atoms):
        atom_i = P[i, :]
        for j in range(i + 1):  # distance matrix will be symmetric; only compute half
            if i == j:
                d[i, j] = 0
                continue
            atom_j = P[j, :]
            d[i, j] = corr_function(atom_i, atom_j)

    # make the matrix symmetric
    for i in range(num_atoms):
        for j in range(i + 1):
            d[j, i] = d[i, j]

    return d

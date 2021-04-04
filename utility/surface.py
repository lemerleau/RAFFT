"""Draw a surface from subopt structres
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LightSource
from scipy import interpolate
from matplotlib.pyplot import rcParams

import numpy as np
from sklearn import manifold
from RNA import bp_distance
from numpy import zeros, meshgrid, array, mgrid
from numpy.random import RandomState


def get_distance_matrix(structures):
    matrix = zeros((len(structures), len(structures)))
    for si, (structi, nrji) in enumerate(structures):
        for sj, (structj, nrjj) in enumerate(structures[si+1: ], start=si+1):
            dist = bp_distance(structi, structj)
            matrix[si, sj] = dist
            matrix[sj, si] = dist
    return matrix


def parse_rafft_output(infile):
    results = []
    with open(infile) as rafft_out:
        seq = rafft_out.readline().strip()
        for l in rafft_out:
            struct, nrj = l.strip().split()
            results += [(struct, float(nrj))]
    return results, seq


def parse_arguments():
    """Parsing command line
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('rafft_out', help="rafft_output")
    return parser.parse_args()


def main():
    args = parse_arguments()
    structures, seq = parse_rafft_output(args.rafft_out)
    dist_mat = get_distance_matrix(structures)
    plt.rcParams["font.family"] = "serif"
    fsize = 7
    plt.rcParams["font.size"] = fsize
    plt.rcParams['text.usetex'] = True

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    
    # map the structures on a plan
    seed = RandomState(seed=3)
    mds = manifold.MDS(n_components=2, max_iter=5000, eps=1e-9, random_state=seed,
                       dissimilarity="precomputed", n_jobs=20)
    pos = mds.fit_transform(dist_mat)

    nrjs = [nrj for st, nrj in structures]

    ti = np.linspace(np.min(pos), np.max(pos), 300)
    XI, YI = np.meshgrid(ti, ti)
    nrj_ = interpolate.Rbf(pos[:, 0], pos[:, 1], nrjs, function="thin_plate")
    p1, p2 = meshgrid(ti, ti)
    nrj_c = nrj_(p1, p2)

    surf = ax.plot_surface(p1, p2, nrj_c, rstride=1, cstride=1, linewidth=0.1,
                           antialiased=False, shade=False, alpha=0.5,
                           cmap=cm.coolwarm)

    ax.scatter(pos[:, 0], pos[:, 1], nrjs, c=nrjs, s=9, lw=0, label='MDS',
               cmap=cm.coolwarm, alpha=1.0)

    ax.set_zlabel(f"Stability (kcal/mol)")
    plt.savefig("test.png", dpi=300)


if __name__ == '__main__':
    main()

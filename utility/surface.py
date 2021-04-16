"""Draw a surface from a set of structures
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
    seen = set()
    with open(infile) as rafft_out:
        seq = rafft_out.readline().strip()
        for l in rafft_out:
            if not l.startswith("#"):
                struct, nrj = l.strip().split()
                if struct not in seen:
                    results += [(struct, float(nrj))]
                    seen.add(struct)
    return results, seq


def parse_arguments():
    """Parsing command line
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rafft_out', help="rafft_output")
    parser.add_argument('--out', "-o", help="outfile")
    return parser.parse_args()


def main():
    args = parse_arguments()
    structures, seq = parse_rafft_output(args.rafft_out)
    dist_mat = get_distance_matrix(structures)
    plt.rcParams["font.family"] = "serif"
    fsize = 13
    plt.rcParams["font.size"] = fsize
    plt.rcParams['text.usetex'] = True

    # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    fig, ax = plt.subplots()
    
    # map the structures on a plan
    seed = RandomState(seed=3)
    mds = manifold.MDS(n_components=2, max_iter=5000, eps=1e-9, random_state=seed,
                       dissimilarity="precomputed", n_jobs=20)
    pos = mds.fit_transform(dist_mat)

    nrjs = [nrj for st, nrj in structures]
    saved_st = [
        "..................................................................................",
        ".....(((((((((((.((.....)))))))))))))......................(((........))).........",
        ".....(((((((((((..........)))))))))))....................((((((.............))))))",
        ".....(((((((((((.((.....)))))))))))))(((((................)))))...................",
        ".....(((((((((((.((.....)))))))))))))....................((((((.............))))))",
        ".....(((((((((((.((.....)))))))))))))(((((..((........))..)))))...................",
        ".....(((((((((((..........))))))))))).............................................",
        ".....(((((((((((.((.....)))))))))))))((((((((.........))).)))))..................."
    ]

    id_struct = [i for i, (st, nrj) in enumerate(structures) if st in saved_st]

    ti = np.linspace(np.min(pos) -1, np.max(pos) +1, 300)
    XI, YI = np.meshgrid(ti, ti)
    nrj_ = interpolate.Rbf(pos[:, 0], pos[:, 1], nrjs, function="thin_plate")
    p1, p2 = meshgrid(ti, ti)
    nrj_c = nrj_(p1, p2)

    # surf = ax.plot_surface(p1, p2, nrj_c, rstride=1, cstride=1, linewidth=0.1,
    #                        antialiased=False, shade=False, alpha=0.5,
    #                        cmap=cm.coolwarm)

    surf = ax.contour(p1, p2, nrj_c, colors="k", linewidths=0.5, levels=7)
    surf = ax.contourf(p1, p2, nrj_c, cmap=cm.coolwarm, alpha=0.3, levels=7)

    # ax.scatter(pos[:, 0], pos[:, 1], nrjs, c=nrjs, s=12, lw=0, label='MDS',
    #            cmap=cm.coolwarm, alpha=1.0)

    # print(id_struct)
    # print(pos[id_struct, 0])


    surf = ax.scatter(pos[:, 0], pos[:, 1], c=nrjs, s=30, lw=0, label='MDS',
                      cmap=cm.coolwarm, alpha=1.0)

    ax.scatter(pos[id_struct, 0], pos[id_struct, 1], c="black", s=80, lw=0, alpha=1.0)
    ax.scatter(pos[id_struct, 0], pos[id_struct, 1], c=array(nrjs)[id_struct], s=30, lw=0,
               label='MDS', cmap=cm.coolwarm, alpha=1.0)

    for px, py, ist in zip(pos[id_struct, 0], pos[id_struct, 1], id_struct):
        if ist == 50:
            ax.text(px-1.0, py+1, ist, fontsize=20)
        elif ist == 44:
            ax.text(px-1, py-3.5, ist, fontsize=20)
        elif ist == 0:
            ax.text(px-2.2, py-1.5, ist, fontsize=20)
        else:
            ax.text(px+1.0, py-1.5, ist, fontsize=20)

    cb = fig.colorbar(surf, ax=ax)
    # cb.set_label("Stability (kcal/mol)", rotation=270)

    if args.out:
        plt.savefig(args.out, dpi=300, transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()

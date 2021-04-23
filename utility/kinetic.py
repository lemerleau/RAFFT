"""From RAFFT output, build a kinetic model. Starting from only unfolded
structures, it generates a folding kinetic trajectory.

Usage:
python kinetic.py rafft.out

"""

import argparse
import subprocess
from os.path import realpath, dirname
from utils import paired_positions, parse_rafft_output
from numpy import array, zeros, exp, matmul, reshape
from numpy.linalg import eig
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


def diag_evo(V, W, init_pop, time):
    final_pop = zeros(len(init_pop))

    for i in range(len(init_pop)):
        final_pop[i] = 0.0
        for k in range(len(init_pop)):
            for j in range(len(init_pop)):
                final_pop[i] += W[i, k] * exp(V[k] * time) * (1.0/W[k, j]) * init_pop[j]

    # for pi, pop in enumerate(init_pop):
    #     final_pop += pop * W[pi, :] * exp(-V[pi] * time)

    return final_pop


def get_connected_prev(cur_struct, prev_pos):
    "get the connected structures"
    cur_pairs = set(paired_positions(cur_struct))
    res = []
    for si, (struct, nrj) in enumerate(prev_pos):
        pairs = set(paired_positions(struct))
        if len(pairs - cur_pairs) == 0:
            res += [si]
    return res


def parse_arguments():
    """Parsing command line
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('rafft_out', help="rafft_output")
    parser.add_argument('--out', '-o', help="output file")
    parser.add_argument('--width', '-wi', help="figure width", type=int, default=8)
    parser.add_argument('--height', '-he', help="figure height", type=int, default=5)
    parser.add_argument('--n_steps', '-ns', help="integration steps", type=int, default=10000)
    parser.add_argument('--show_thres', '-st', help="threshold population to show", type=float, default=0.01)
    parser.add_argument('--font_size', '-fs', help="font size for the colors", type=int, default=15)
    parser.add_argument('--init_pop', '-ip', help="initialization of the population <POS>:<WEI>", nargs="*")
    parser.add_argument('--uni', action="store_true", help="uniform distribution")
    parser.add_argument('--temp', '-t', help="kT (kcal/mol)", type=float, default=0.6)
    parser.add_argument('--plot', action="store_true", help="plot kinetics")
    return parser.parse_args()


def main():
    args = parse_arguments()
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = args.font_size
    plt.rcParams["figure.figsize"] = args.width, args.height
    init_population = []

    if args.init_pop is not None:
        tot = 0.0
        for el in args.init_pop:
            pos, wei = el.split(":")
            init_population += [(int(pos), float(wei))]
            tot += float(wei)
        init_population = [(pos, wei/tot) for pos, wei in init_population]

    fast_paths, seq = parse_rafft_output(args.rafft_out)

    # to draw the paths
    nb_steps = len(fast_paths)
    nb_saved = len(fast_paths[-1])


    # transition matrix
    # struct_list = [st for el in fast_paths for st, _ in el]
    struct_list = []
    for el in fast_paths:
        for st, _ in el:
            if st not in struct_list:
                struct_list += [st]
    struct_map = {st: si for si, st in enumerate(struct_list)}
    map2_struct = {}
    map2_struct[0] = (0, 0)
    nb_struct = len(struct_list)
    transition_mat = zeros((nb_struct, nb_struct))

    # save position in the canvas for each structure
    actual_position, actual_sizes = {}, {}

    # save nrj differences
    nrj_changes = {}

    # width of points
    pos_hor = 0
    crop_side = 0
    # store best change
    min_change = 0
    KT = 0.6

    transition_mat[0, 0] = 1.0
    for step_i, fold_step in enumerate(fast_paths):
        if len(fold_step) > 1:
            for str_i, (struct, nrj) in enumerate(fold_step):
                lprev_co = get_connected_prev(struct, fast_paths[step_i - 1])
                nrj_changes[(step_i, str_i)] = {}
                map2_struct[struct_map[struct]] = (step_i, str_i)
                map_cur = struct_map[struct]

                for si in lprev_co:
                    prev_st, prev_nrj = fast_paths[step_i-1][si]
                    delta_nrj = nrj - prev_nrj
                    map_cur, map_prev = struct_map[struct], struct_map[prev_st]
                    if prev_st != struct:
                        transition_mat[map_cur, map_prev] = exp(delta_nrj/KT)
                        transition_mat[map_prev, map_cur] = exp(-delta_nrj/KT)

    

    # # normalize per line
    norm_h = transition_mat.sum(axis=1)
    transition_mat = (transition_mat.T/norm_h).T

    for si, struct in enumerate(struct_list):
        transition_mat[si, si] = -transition_mat[si,:].sum()

    # normalize per column
    # norm_v = transition_mat.sum(axis=0)
    # transition_mat = transition_mat/norm_v
    V, W = eig(transition_mat)
    V, W = V.real, W.real

    trajectory = []
    if args.init_pop is None:
        init_pop = array([1.0] + [0.0 for _ in range(nb_struct-1)])
    else:
        init_pop = array([0.0 for _ in range(nb_struct)])
        for p, w in init_population:
            init_pop[p] = w

    if args.uni:
        uni_wei = 1.0/nb_struct
        for p in range(nb_struct):
            init_pop[p] = uni_wei

    trajectory += [deepcopy(init_pop)]

    for st in range(args.n_steps):
        init_pop += matmul(init_pop, transition_mat) * 0.01
        trajectory += [init_pop/sum(init_pop)]

        # tmp_pop = diag_evo(V, W, init_pop, float(st))
        # trajectory += [tmp_pop/sum(tmp_pop)]
    

    res = []
    tot_pop = 0.0
    for si, final_p in enumerate(init_pop):
        step_i, str_i = map2_struct[struct_map[struct_list[si]]]
        struct, nrj = fast_paths[step_i][str_i]
        res += [(struct_list[si], final_p, nrj)]
        tot_pop += final_p

    res.sort(key=lambda el: el[1])
    for st, fp, nrj in res:
        print("{} {:6.3f} {:5.1f} {:d}".format(st, fp, nrj, struct_map[st]))

    if args.plot:
        trajectory = array(trajectory)

        left, width = 0.10, 0.85
        bottom, height = 0.10, 0.85
        spacing = 0.000
        rect_scatter = [left, bottom, width, height]
        rect_histy = [left + width + spacing, bottom, 0.2, height]
        fig = plt.figure(1)
        kin_f = fig.add_axes(rect_scatter)
        kin_f.grid(True, color="grey",linestyle="--", linewidth=0.2)

        kin_f.set_xlim([1, args.n_steps])
        kin_f.set_ylim([0.0, 1.01])

        for si, st in enumerate(struct_list):
            if any(trajectory[:, si] > args.show_thres):
                kin_f.plot(trajectory[:, si], alpha=0.8, label=si)

        kin_f.set_xscale("log")
        kin_f.legend()
        if args.out is not None:
            plt.savefig(args.out, dpi=300, transparent=True)
        else:
            plt.show()


if __name__ == '__main__':
    main()

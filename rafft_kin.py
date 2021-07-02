#!/usr/bin/env python3
"""From RAFFT output, build a kinetic model. Starting from only unfolded
structures, it generates a folding kinetic trajectory.

Usage:
python rafft_kin.py rafft.out --plot

"""

import argparse
from utils import paired_positions, parse_rafft_output
from numpy import array, zeros, exp, diag
from numpy.linalg import eig, inv
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


def get_connected_prev(cur_struct, prev_pos):
    "get the connected structures"
    cur_pairs = set(paired_positions(cur_struct))
    res = []
    for si, (struct, nrj) in enumerate(prev_pos):
        pairs = set(paired_positions(struct))
        if len(pairs - cur_pairs) == 0:
            res += [si]
    return res


def get_connected_next(cur_struct, prev_pos):
    "get the connected structures"
    cur_pairs = set(paired_positions(cur_struct))
    for si, (struct, nrj) in enumerate(prev_pos):
        pairs = set(paired_positions(struct))
        if len(cur_pairs - pairs) == 0:
            return True


def get_transition_mat(fast_paths, nb_struct, map2_struct, struct_map,
                       struct_list, temp, other_rate=False):
    KT = 0.61
    nrj_changes = {}
    transition_mat = zeros((nb_struct, nb_struct), dtype=np.double)

    for step_i, fold_step in enumerate(fast_paths):
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
                    # M_ij = k(j -> i)
                    if other_rate:
                        transition_mat[map_cur, map_prev] = exp(temp * delta_nrj/KT)
                        transition_mat[map_prev, map_cur] = exp(-temp * delta_nrj/KT)
                    else:
                        transition_mat[map_cur, map_prev] = min(1.0, exp(temp * delta_nrj/KT))
                        transition_mat[map_prev, map_cur] = min(1.0, exp(-temp * delta_nrj/KT))

    # normalize input and output flows
    if other_rate:
        norm_h = transition_mat.sum(axis=1)
        norm_v = transition_mat.sum(axis=0)
        transition_mat = (transition_mat.T/(norm_h + norm_v)).T

    for si, struct in enumerate(struct_list):
        transition_mat[si, si] = -transition_mat[si,:].sum()

    return transition_mat


def parse_arguments():
    """Parsing command line
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('rafft_out', help="rafft_output")
    parser.add_argument('--out', '-o', help="output file")
    parser.add_argument('--width', '-wi', help="figure width", type=int, default=2)
    parser.add_argument('--height', '-he', help="figure height", type=int, default=1.5)
    parser.add_argument('--n_steps', '-ns', help="integration steps", type=int, default=100)
    parser.add_argument('--show_thres', '-st', help="threshold population to show", type=float, default=0.08)
    parser.add_argument('--font_size', '-fs', help="font size for the colors", type=int, default=5)
    parser.add_argument('--init_pop', '-ip', help="initialization of the population <POS>:<WEI>", nargs="*")
    parser.add_argument('--uni', action="store_true", help="uniform distribution")
    parser.add_argument('--other_rate', action="store_true", help="use the other rate")
    parser.add_argument('--temp', '-t', help="exp(t * delta_G/kT), t is a scaling constant", type=float, default=1.0)
    parser.add_argument('--max_time', '-mt', help="max time (exp scale)", type=float, default=30)
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
        # init_population = [(pos, wei/tot) for pos, wei in init_population]
        init_population = [(pos, wei) for pos, wei in init_population]

    fast_paths, seq = parse_rafft_output(args.rafft_out)

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

    transition_mat = get_transition_mat(fast_paths, nb_struct, map2_struct,
                                        struct_map, struct_list, args.temp,
                                        args.other_rate)

    # parse_rafft_output_(args.rafft_out, struct_map)
    # np.set_printoptions(threshold=np.inf)
    V, W = eig(transition_mat.T)
    iW = inv(W)

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

    time_step = args.max_time/args.n_steps
    times = []
    times += [exp(-4)]

    # Eigen method
    for st in range(args.n_steps):

        time = exp(time_step * st-4)
        times += [time]
        tmp_pop = W @ diag(exp(V * time)) @ (iW @ init_pop)
        trajectory += [tmp_pop/tmp_pop.sum()]

    # take last population
    # init_pop = W[] @ diag(exp(V[0] * time)) @ iW @ init_pop
    init_pop = trajectory[-1]

    res = []
    tot_pop = 0.0
    for si, fin_p in enumerate(init_pop):
        step_i, str_i = map2_struct[struct_map[struct_list[si]]]
        struct, nrj = fast_paths[step_i][str_i]
        res += [(struct_list[si], fin_p, nrj)]
        tot_pop += fin_p

    res.sort(key=lambda el: el[1])
    for st, fp, nrj in res:
        print("{} {:6.3f} {:5.1f} {:d}".format(st, fp.real, nrj, struct_map[st]))

    if args.plot:
        trajectory = array(trajectory).real

        left, width = 0.10, 0.88
        bottom, height = 0.10, 0.88
        rect_scatter = [left, bottom, width, height]
        fig = plt.figure(1)
        kin_f = fig.add_axes(rect_scatter)
        kin_f.grid(True, color="grey", linestyle="--", linewidth=0.5)

        kin_f.set_xlim([times[0], times[-1]])

        for si, st in enumerate(struct_list):
            if any(trajectory[:, si] > args.show_thres):
                kin_f.plot(times, trajectory[:, si], alpha=0.8, label=si, linewidth=0.8)

        kin_f.set_xscale("log")
        kin_f.legend(ncol=2, fontsize=int(args.font_size * 0.8))
        if args.out is not None:
            plt.savefig(args.out, dpi=300, transparent=True)
        else:
            plt.show()


if __name__ == '__main__':
    main()

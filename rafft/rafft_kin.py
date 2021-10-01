#!/usr/bin/env python3
"""From RAFFT output, build a kinetic model. Starting from only unfolded
structures, it generates a folding kinetic trajectory.

Usage:
python rafft_kin.py rafft.out --plot

"""

import argparse
from rafft.utils import paired_positions, parse_rafft_output
from numpy import array, zeros, exp, diag
from numpy.linalg import eig, inv
from scipy.linalg import eig as sci_eig
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


def plot_traj(trajectory, struct_list, times, font_size, width, height,
              show_thres, out_file=None):
    """plot trajectory
    """
    trajectory = array(trajectory).real

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = font_size
    plt.rcParams["figure.figsize"] = width, height
    left, width = 0.10, 0.88
    bottom, height = 0.10, 0.88
    rect_scatter = [left, bottom, width, height]
    fig = plt.figure(1)
    kin_f = fig.add_axes(rect_scatter)
    kin_f.grid(True, color="grey", linestyle="--", linewidth=0.2)

    kin_f.set_xlim([times[0], times[-1]])

    for si, (st, nrj) in enumerate(struct_list):
        if any(trajectory[:, si] > show_thres):
            kin_f.plot(times, trajectory[:, si::int(trajectory.shape[0]/100)], alpha=0.8, label=si)

    kin_f.set_xscale("log")
    kin_f.legend(ncol=2, fontsize=int(font_size * 0.8))
    if out_file is not None:
        plt.savefig(out_file, dpi=300, transparent=True)
    else:
        plt.show()


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


def get_transition_mat(fast_paths, nb_struct, struct_map):
    transition_mat = zeros((nb_struct, nb_struct), dtype=np.double)
    KT = 0.61

    for step_i, fold_step in enumerate(fast_paths):
        for struct, nrj in fold_step:
            # get all structures connected from previous step
            lprev_co = get_connected_prev(struct, fast_paths[step_i - 1])
            map_cur, cur_nrj = struct_map[struct]

            for si in lprev_co:
                prev_st, prev_nrj = fast_paths[step_i-1][si]
                map_prev, prev_nrj = struct_map[prev_st]
                delta_nrj = cur_nrj - prev_nrj
                if prev_st != struct:
                    # M_ij = k(i -> j)
                    transition_mat[map_cur, map_prev] = min(1.0, exp(delta_nrj/KT))
                    # # M_ji = k(i -> j)
                    transition_mat[map_prev, map_cur] = min(1.0, exp(-delta_nrj/KT))

    # normalize input and output flows
    for si in range(nb_struct):
        transition_mat[si, si] = -transition_mat[si,:].sum()

    return transition_mat


def kinetics(fast_paths, max_time, n_steps, initial_pop=None):
    """fast_paths = list of list of (structure, energy)
    initial_pop = list of float giving the initial population of all/few structures
    """

    struct_list = []
    for el in fast_paths:
        for st, nrj in el:
            if (st, nrj) not in struct_list:
                struct_list += [(st, nrj)]
    
    # create a non-redundant mapping
    struct_map = {st: (si, nrj) for si, (st, nrj) in enumerate(struct_list)}
    nb_struct = len(struct_list)
    transition_mat = get_transition_mat(fast_paths, nb_struct, struct_map)

    # for el in transition_mat:
        # print(" ".join([str(eli) for eli in el]))

    # initialize the kinetic
    if initial_pop is None:
        # starts with the unfolded state
        init_pop = array([1.0] + [0.0 for _ in range(nb_struct-1)])
    else:
        init_pop = array([0.0 for _ in range(nb_struct)])
        for p, w in initial_pop:
            init_pop[p] = w

    trajectory = [deepcopy(init_pop)]

    # diagonalize the transition matrix to solve the system
    V, W = eig(transition_mat.T)
    iW = inv(W)

    residuals = abs(transition_mat.T - (W @ diag(V) @ iW)).sum(axis=0).sum()

    # # equilibrium population
    min_p = np.argmin(abs(V))
    equi_pop = W[:, min_p].real
    equi_pop /= equi_pop.sum()
    time_step = max_time / n_steps
    
    if residuals < 10**-10:
        times = [exp(-4)]
        for st in range(n_steps):
            time = exp(time_step * st-4)
            times += [time]
            tmp_pop = W @ diag(exp(V * time)) @ (iW @ init_pop)
            trajectory += [tmp_pop]
        equi_pop = trajectory[-1]
    else:
        # print("# ERROR: Using numerical integration")
        dt = 0.01
        time = 0.0
        times = [time]
        for st in range(n_steps):
            init_pop += init_pop @ transition_mat * 0.01
            trajectory += [init_pop/sum(init_pop)]
            times += [times[-1] + dt]
        equi_pop = trajectory[-1]

    # get the equilibrium population
    str_equi_pop = [(str_, nrj, ep, struct_map[str_][0]) for (str_, nrj), ep in zip(struct_list, equi_pop.real)]
    return trajectory, times, struct_list, str_equi_pop


def parse_arguments():
    """Parsing command line
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('rafft_out', help="rafft_output")
    parser.add_argument('--out', '-o', help="output file")
    parser.add_argument('--width', '-wi', help="figure width", type=int, default=7)
    parser.add_argument('--height', '-he', help="figure height", type=int, default=5)
    parser.add_argument('--n_steps', '-ns', help="integration steps", type=int, default=100)
    parser.add_argument('--show_thres', '-st', help="threshold population to show", type=float, default=0.08)
    parser.add_argument('--font_size', '-fs', help="font size for the colors", type=int, default=15)
    parser.add_argument('--init_pop', '-ip', help="initialization of the population <POS>:<WEI>", nargs="*")
    parser.add_argument('--uni', action="store_true", help="uniform distribution")
    parser.add_argument('--other_rate', action="store_true", help="use the other rate")
    parser.add_argument('--max_time', '-mt', help="max time (exp scale)", type=float, default=30)
    parser.add_argument('--plot', action="store_true", help="plot kinetics")
    return parser.parse_args()


def main():
    args = parse_arguments()
    init_population = None

    if args.init_pop is not None:
        tot = 0.0
        for el in args.init_pop:
            pos, wei = el.split(":")
            init_population += [(int(pos), float(wei))]
            tot += float(wei)
        # init_population = [(pos, wei/tot) for pos, wei in init_population]
        init_population = [(pos, wei) for pos, wei in init_population]

    fast_paths, seq = parse_rafft_output(args.rafft_out)

    trajectory, times, struct_list, equi_pop = kinetics(fast_paths, args.max_time, args.n_steps, init_population)
    # kinetics(fast_paths, args.max_time, args.n_steps, init_population)
    equi_pop.sort(key=lambda el: el[2])

    for st, nrj, fp, si in equi_pop:
        print("{} {:6.3f} {:5.1f} {:d}".format(st, fp, nrj, si))

    if args.plot:
        plot_traj(trajectory, struct_list, times, args.font_size, args.width, args.height, args.show_thres)


if __name__ == '__main__':
    main()

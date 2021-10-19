#!/usr/bin/env python3
"""From RAFFT output, build a kinetic model. Starting from only unfolded
structures, it generates a folding kinetic trajectory.

Usage:
rafft_kin rafft.out --plot

"""

from rafft.utils import paired_positions, parse_rafft_output
from numpy import array, zeros, exp, diag
from scipy.linalg import eig, inv
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

    for si, struct in enumerate(struct_list):
        if any(trajectory[:, si] > show_thres):
            kin_f.plot(times, trajectory[:, si], alpha=0.8, label=si)

    kin_f.set_xscale("log")
    kin_f.legend(ncol=2, fontsize=int(font_size * 0.8))
    if out_file is not None:
        plt.savefig(out_file, dpi=300, transparent=True)
    else:
        plt.show()


def get_connected_prev(cur_struct, prev_pos):
    "get the connected structures"
    cur_pairs = set(paired_positions(cur_struct.str_struct))
    res = []
    for si, struct in enumerate(prev_pos):
        pairs = set(paired_positions(struct.str_struct))
        if len(pairs - cur_pairs) == 0:
            res += [si]
    return res


def get_connected_next(cur_struct, prev_pos):
    "get the connected structures"
    cur_pairs = set(paired_positions(cur_struct.str_struct))
    for si, struct in enumerate(prev_pos):
        pairs = set(paired_positions(struct.str_struct))
        if len(cur_pairs - pairs) == 0:
            return True


def get_transition_mat(fast_paths, nb_struct, struct_map):
    transition_mat = zeros((nb_struct, nb_struct), dtype=np.longdouble)
    KT = 0.61

    for step_i, fold_step in enumerate(fast_paths):
        for struct in fold_step:
            # get all structures connected from previous step
            lprev_co = get_connected_prev(struct, fast_paths[step_i - 1])
            map_cur, cur_nrj = struct_map[struct.str_struct]

            for si in lprev_co:
                prev_st = fast_paths[step_i-1][si]
                map_prev, prev_nrj = struct_map[prev_st.str_struct]
                delta_nrj = cur_nrj - prev_nrj
                if map_cur != map_prev:
                    # # M_ij = k(i -> j)
                    transition_mat[map_prev, map_cur] = min(1.0, exp(-delta_nrj/KT))
                    transition_mat[map_cur, map_prev] = min(1.0, exp(delta_nrj/KT))

    # normalize input and output flows
    for si in range(nb_struct):
        transition_mat[si, si] = -transition_mat[si,:].sum()

    return transition_mat


def kinetics(fast_paths, max_time, n_steps, initial_pop=None):
    """
    input:
    fast_paths = list of list of (structure, energy)
    initial_pop = list of float giving the initial population of all/few structures

    output:
    trajectory = trajectory of population
    times = time steps
    struct_list = list of unique structures
    str_equi_pop = final population
    """
    seen = set()
    struct_list = []
    for el in fast_paths:
        for struct in el:
            if struct.str_struct not in seen:
                seen.add(struct.str_struct)
                struct_list += [struct]
    
    # create a non-redundant mapping
    struct_map = {struct.str_struct: (si, struct.energy) for si, struct in enumerate(struct_list)}
    nb_struct = len(struct_list)
    transition_mat = get_transition_mat(fast_paths, nb_struct, struct_map)

    # initialize the kinetic
    if initial_pop is None:
        # starts with the unfolded state
        init_pop = array([1.0] + [0.0 for _ in range(nb_struct-1)], dtype=np.longdouble)
    else:
        init_pop = array([0.0 for _ in range(nb_struct)], dtype=np.longdouble)
        for p, w in initial_pop:
            init_pop[p] = w

    trajectory = [deepcopy(init_pop)]

    # diagonalize the transition matrix to solve the system
    V, W = eig(transition_mat.T, check_finite=True)
    iW = inv(W)

    time_step = max_time / n_steps
    
    times = [exp(-4)]
    for st in range(n_steps):
        time = exp(time_step * st-4)
        times += [time]
        tmp_pop = W @ diag(exp(V * time)) @ (iW @ init_pop)
        trajectory += [tmp_pop/tmp_pop.sum()]

    equi_pop = trajectory[-1]

    # g = nx.from_numpy_matrix(transition_mat)
    # nx.draw(g)
    # plt.show()

    str_equi_pop = [(struct.str_struct, struct.energy, ep, struct_map[struct.str_struct][0]) for struct, ep in zip(struct_list, equi_pop.real)]
    return trajectory.real, times, struct_list, str_equi_pop

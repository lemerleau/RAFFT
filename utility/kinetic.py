"""Take the output of rafft and produce a latex file to display the fold paths.

It uses varna to produce 2ndary structures.
"""

import argparse
import subprocess
from os.path import realpath, dirname
<<<<<<< HEAD
from utils import paired_positions
from numpy import array, zeros, exp, matmul
import matplotlib.pyplot as plt


def parse_rafft_output(infile):
    results = []
    with open(infile) as rafft_out:
        seq = rafft_out.readline().strip()
        for l in rafft_out:
            if l.startswith("# --"):
                results += [[]]
            else:
                struct, nrj = l.strip().split()
                results[-1] += [(struct, float(nrj))]
    return results, seq


=======
from utils import paired_positions, parse_rafft_output
from numpy import array, zeros, exp, matmul, reshape
import numpy as np
import matplotlib.pyplot as plt

>>>>>>> origin/modifs
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
    parser = argparse.ArgumentParser(description="Uses VARNA to plot the fast-paths predicted by RAFFT. !! It creates a temporary directory in the current folder!!")
    parser.add_argument('rafft_out', help="rafft_output")
    parser.add_argument('--out', '-o', help="output file")
<<<<<<< HEAD
    parser.add_argument('--width', '-wi', help="figure width", type=int, default=500)
    parser.add_argument('--height', '-he', help="figure height", type=int, default=300)
    parser.add_argument('--res_varna', '-rv', help="change varna resolution", type=float, default=1.0)
    parser.add_argument('--line_thick', '-lt', help="line thickness", type=int, default=2)
    parser.add_argument('--font_size', '-fs', help="font size for the colors", type=int, default=3)
    parser.add_argument('--varna_jar', help="varna jar (please download it from VARNA website)")
    parser.add_argument('--no_col', action="store_true", help="don't use the color gradient for the edges")
    parser.add_argument('--no_fig', action="store_true", help="you already computed the structures previously?")
=======
    parser.add_argument('--width', '-wi', help="figure width", type=int, default=8)
    parser.add_argument('--height', '-he', help="figure height", type=int, default=5)
    parser.add_argument('--n_steps', '-ns', help="integration steps", type=int, default=50)
    parser.add_argument('--show_thres', '-st', help="threshold population to show", type=float, default=0.1)
    parser.add_argument('--font_size', '-fs', help="font size for the colors", type=int, default=3)
>>>>>>> origin/modifs
    return parser.parse_args()


def main():
    args = parse_arguments()
<<<<<<< HEAD
=======
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.size"] = args.font_size
    plt.rcParams["figure.figsize"] = args.width, args.height

>>>>>>> origin/modifs

    fast_paths, seq = parse_rafft_output(args.rafft_out)

    # to draw the paths
    nb_steps = len(fast_paths)
    nb_saved = len(fast_paths[-1])

    # transition matrix
<<<<<<< HEAD
    struct_list = [st for el in fast_paths for st, _ in el]
=======
    # struct_list = [st for el in fast_paths for st, _ in el]
    struct_list = []
    for el in fast_paths:
        for st, _ in el:
            if st not in struct_list:
                struct_list += [st]
>>>>>>> origin/modifs
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

    for step_i, fold_step in enumerate(fast_paths):
        if len(fold_step) > 1:
            for str_i, (struct, nrj) in enumerate(fold_step):
                lprev_co = get_connected_prev(struct, fast_paths[step_i - 1])
                nrj_changes[(step_i, str_i)] = {}
                map2_struct[struct_map[struct]] = (step_i, str_i)

<<<<<<< HEAD

=======
>>>>>>> origin/modifs
                for si in lprev_co:
                    prev_st, prev_nrj = fast_paths[step_i-1][si]
                    delta_nrj = nrj - prev_nrj
                    map_cur, map_prev = struct_map[struct], struct_map[prev_st]
                    transition_mat[map_cur, map_prev] = exp(delta_nrj/KT)
                    transition_mat[map_prev, map_cur] = exp(-delta_nrj/KT)

<<<<<<< HEAD
    trajectory = []
    init_pop = array([1.0] + [0.0 for _ in range(nb_struct-1)])
    for _ in range(100):
=======

    norm_h = transition_mat.sum(axis=1)
    transition_mat = (transition_mat.T/norm_h).T
    norm_v = transition_mat.sum(axis=0)
    transition_mat /= norm_h

    trajectory = []
    init_pop = array([1.0] + [0.0 for _ in range(nb_struct-1)])
    for _ in range(args.n_steps):
>>>>>>> origin/modifs
        init_pop = matmul(init_pop, transition_mat)
        init_pop /= sum(init_pop)
        trajectory += [init_pop]

    res = []
    for si, final_p in enumerate(init_pop):
        step_i, str_i = map2_struct[struct_map[struct_list[si]]]
        struct, nrj = fast_paths[step_i][str_i]
        res += [(struct_list[si], final_p, nrj)]
<<<<<<< HEAD
    res.sort(key=lambda el: el[1])
    for st, fp, nrj in res:
        print("{} {:5.3f} {:5.1f}".format(st, fp, nrj))

    trajectory = array(trajectory)
    for si, st in enumerate(struct_list):
        plt.plot(trajectory[:, si])

    plt.show()
=======

    res.sort(key=lambda el: el[1])
    for st, fp, nrj in res:
        print("{} {:5.3f} {:5.1f} {:d}".format(st, fp, nrj, struct_map[st]))

    trajectory = array(trajectory)

    left, width = 0.10, 0.85
    bottom, height = 0.10, 0.85
    spacing = 0.000
    rect_scatter = [left, bottom, width, height]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    fig = plt.figure(1)
    kin_f = fig.add_axes(rect_scatter)
    kin_f.grid(True, color="grey",linestyle="--", linewidth=0.2)
    kin_f.set_xlim([0, args.n_steps])
    kin_f.set_ylim([-0.01, 1.01])
    for si, st in enumerate(struct_list):
        if any([el > args.show_thres for el in trajectory[:, si]]):
            kin_f.plot(trajectory[:, si], alpha=0.8, label=si)

    kin_f.legend()
    if args.out is not None:
        plt.savefig(args.out, dpi=300, transparent=True)
    else:
        plt.show()
>>>>>>> origin/modifs


if __name__ == '__main__':
    main()

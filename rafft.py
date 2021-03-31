"""Fold an RNA sequence by searching for complementary segments.

Take one RNA sequence and produce two strands: X, X'. X' is a complementary
strand of X so the alignment of both strands is good if it contains canonical
pairs.

Using the auto-correlation, taking advantage of the FFT, one can find quickly
the regions with many canonical pairs.

Next, we take the largest consecutive pairs, and fix them into the list of base-pairs (BP)

Then the sequence is split into inner loop and outer loop.

The inner and outer loops are treated recursively until no BP can be formed
"""

from numpy import concatenate
from numpy import sum as npsum
from os.path import realpath, dirname
from utils import plot_bp_matrix, auto_cor, dot_bracket
from utils import prep_sequence, benchmark_vrna, eval_dynamic
from utils import get_inner_loop, get_outer_loop, eval_one_struct
from utils import merge_pair_list
from itertools import product
import argparse
from RNA import fold_compound

# window size when searching for pairs
PK_MODE = False
# Min number of BP
MIN_BP = 1
# Min number of internal unpaired positions
MIN_HP = 3


def window_slide(seq, cseq, pos, pos_list):
    """Slide a window along the align pair of sequence to find the consecutive paired positions
    """
    len_seq = seq.shape[1]
    # the position of the spike gives the relative position between both
    # strands
    if pos < len_seq:
        seq_ = seq[:, :pos+1]
        cseq_ = cseq[:, len_seq-pos-1:]
    else:
        seq_ = seq[:, pos-len_seq+1:]
        cseq_ = cseq[:, :2*len_seq-pos-1]

    # test if it represents the average bp
    # tmp = mean(npsum(seq_*cseq_, axis=0))
    # print(tmp == cor, pos, len_seq)
    len_2 = int(seq_.shape[1]/2) + seq_.shape[1]%2

    # When the strands are appropriatly aligned, we have to search for base
    # pairs with a sliding window
    tot = npsum(seq_[:, :len_2]*cseq_[:, :len_2], axis=0)

    # search for consecutive BPs
    max_nb, tmp_max, max_score, max_i, max_j = 0, 0, 0, 0, 0
    for i in range(len_2):
        if pos < len_seq:
            ip, jp = i, pos-i
        else:
            ip, jp = pos-len_seq+1+i, len_seq-i-1

        # check if positions are contiguous
        if i > 0 and pos_list[ip] - pos_list[ip-1] == 1 and \
           pos_list[jp+1] - pos_list[jp] == 1:
            tot[i] = (tot[i-1]+tot[i])*tot[i]

        if tot[i] == 0:
            tmp_max = 0
        else:
            tmp_max += 1

        # search for the highest number of consecutive BPs
        # and test if at least MIN_HP unpaired positions in between
        if tot[i] >= max_score and pos_list[jp] - pos_list[ip] > MIN_HP:
            max_score = tot[i]
            max_nb = tmp_max
            max_i, max_j = ip, jp

    return max_nb, max_i, max_j, max_score


def find_best_consecutives(seq, cseq, pos_list, pair_list, cor_l):
    # find largest bp region
    best_sol = []
    seen = set()
    max_bp, max_i, max_j, max_s, tmp_nrj = 0, 0, 0, 0, MIN_NRJ
    best_nrj = MIN_NRJ
    for pos, c in cor_l[::-1][:NB_MODE]:
        mx_i, mip, mjp, ms = window_slide(seq, cseq, pos, pos_list)

        if mx_i > 0:
            tmp_pair = [(pos_list[mip-i], pos_list[mjp+i]) for i in range(mx_i)]
            tmp_nrj = eval_dynamic(SEQ_COMP, pair_list, tmp_pair, LEN_SEQ, SEQ)
        else:
            tmp_nrj = MIN_NRJ

        if tmp_nrj < MIN_NRJ:
            max_bp, max_s, max_i, max_j = mx_i, ms, mip, mjp
            best_tmp = tmp_pair
            best_nrj = tmp_nrj
            best_sol += [(max_bp, max_s, max_i, max_j, best_nrj, best_tmp)]

    best_sol.sort(key=lambda el: el[4])
    return best_sol


def create_childs(seq, cseq, pair_list, pos_list):
    """Recursive scheme
    """

    len_seq = seq.shape[1]
    cor_l = auto_cor(seq, cseq, PAD)
    cor_l.sort(key=lambda el: el[1])

    best_solutions = find_best_consecutives(seq, cseq, pos_list, pair_list, cor_l)

    cur_list_sol = []
    for solution in best_solutions:
        # save the largest number of consecutive BPs
        max_bp, max_s, max_i, max_j, best_nrj, best_tmp = solution
        for el in pair_list:
            best_tmp += [el]

        if max_j - max_i > 1:
            # Inner loop case
            iseq, icseq, ipos_list_2 = get_inner_loop(seq, cseq, max_i, max_j, max_bp, pos_list, len_seq)
            in_side = (iseq, icseq, ipos_list_2)
        else:
            in_side = None

        if max_i - (max_bp - 1) > 0 or max_j + max_bp < len_seq:
            # Outer loop case
            oseq, ocseq, opos_list_2 = get_outer_loop(seq, cseq, max_i, max_j, max_bp, pos_list, len_seq)
            out_side = (oseq, ocseq, opos_list_2)
        else:
            out_side = None

        cur_list_sol += [(in_side, out_side, best_tmp, best_nrj)]
    return cur_list_sol


def bfs_pairs(glob_tree, step=0):

    tmp_glob_tree = []
    tmp_tree = []
    new_glob_tree = []

    if VERBOSE:
        print("# {:-^20}".format(step))
        print(SEQ)
        for tree in glob_tree:
            if len(tree) > 0:
                tmp_pair_l = tree[0][2]
                tmp_str = dot_bracket(tmp_pair_l, LEN_SEQ)
                print(tmp_str, "{:.2f}".format(eval_one_struct(SEQ_COMP, tmp_pair_l, LEN_SEQ, SEQ)))

    new_sol = False
    # split current nodes
    for tree in glob_tree:
        tmp_tree = []
        for in_side, out_side, pair_list, cur_nrj in tree:
            icur_list, ocur_list = [], []

            if in_side is not None:
                iseq, icseq, ipos_list = in_side
                icur_list = create_childs(iseq, icseq, pair_list, ipos_list)
                if len(icur_list) > 0:
                    tmp_tree += [icur_list]

            if out_side is not None:
                oseq, ocseq, opos_list = out_side
                ocur_list = create_childs(oseq, ocseq, pair_list, opos_list)
                if len(ocur_list) > 0:
                    tmp_tree += [ocur_list]

            if len(icur_list) == 0 and len(ocur_list) == 0:
                new_glob_tree += [[(in_side, out_side, pair_list, cur_nrj)]]
            else:
                new_sol = True

        if len(tmp_tree) > 0:
            tmp_glob_tree += [tmp_tree]


    # if no new tree formed, print out the results
    if not new_sol:
        all_pairs = []
        for tree in glob_tree:
            all_pairs += [tree[0][2]]
        return all_pairs

    # Combine stems formed in independent sub segments
    nb_branch = 0
    for tree in tmp_glob_tree:
        if len(tree) > 1:
            for comp in product(*tree):
                tmp_tree = []
                cur_pair_list = []
                for one_side in list(comp):
                    in_side, out_side, pair_list, cur_nrj = one_side
                    merge_pair_list(cur_pair_list, pair_list)
                    tmp_tree += [(in_side, out_side, cur_pair_list, cur_nrj)]
                new_glob_tree += [tmp_tree]
                if nb_branch > MAX_BRANCH:
                    break
                nb_branch += 1
        else:
            one_side_pos = tree[0]
            for one_side in one_side_pos:
                tmp_tree = []
                cur_pair_list = []
                in_side, out_side, pair_list, cur_nrj = one_side
                merge_pair_list(cur_pair_list, pair_list)
                tmp_tree += [(in_side, out_side, cur_pair_list, cur_nrj)]
                new_glob_tree += [tmp_tree]
                if nb_branch > MAX_BRANCH:
                    break
                nb_branch += 1

    # XXX: Put the energy evaluation once above
    new_glob_tree.sort(key=lambda el: eval_one_struct(SEQ_COMP, el[0][2],LEN_SEQ, SEQ))

    # XXX: This makes sure that no duplication are in the saved stack of
    # structures. Try to remove this workaround
    seen = set()
    unique = []
    for pi, tree in enumerate(new_glob_tree):
        tmp_str = dot_bracket(tree[0][2], LEN_SEQ)
        if tmp_str not in seen:
            seen.add(tmp_str)
            unique += [pi]
    new_glob_tree = [new_glob_tree[pi] for pi in unique]
 
    # Save the best trajectories from the saved ones
    new_glob_tree = new_glob_tree[:MAX_STACK]

    return bfs_pairs(new_glob_tree, step+1)


def parse_arguments():
    """Parsing command line
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--sequence', '-s', help="sequence")
    parser.add_argument('--seq_file', '-sf', help="sequence file")
    parser.add_argument('--n_mode', '-n', help="number of mode to test during the search", type=int, default=100)
    parser.add_argument('--pad', '-p', help="padding, a normalization constant for the autocorrelation", type=float, default=1.0)
    parser.add_argument('--min_bp', '-mb', help="minimum bp to be detectable", type=int, default=1)
    parser.add_argument('--min_hp', '-mh', help="minimum unpaired positions in hairpins", type=int, default=3)
    parser.add_argument('--min_nrj', '-mn', help="minimum nrj loop", type=float, default=0)
    parser.add_argument('--max_stack', '-ms', help="maximum stored stacks", type=int, default=1)
    parser.add_argument('--max_stack_branch', '-msb', help="maximum branches to explor", type=int, default=1000)
    parser.add_argument('--bp_only', action="store_true", help="don't use the NRJ")
    parser.add_argument('--fasta', action="store_true", help="fasta output")
    parser.add_argument('--one', action="store_true", help="output only the structure with lowest energy")
    parser.add_argument('--verbose', action="store_true", help="verbose output")
    parser.add_argument('--GC', type=float, help="GC weight", default=3.00)
    parser.add_argument('--AU', type=float, help="GC weight", default=2.00)
    parser.add_argument('--GU', type=float, help="GU weight", default=1.00)
    return parser.parse_args()


def main():
    args = parse_arguments()
    # HANDLE INPUTS -----------------------------------------------------------
    assert args.sequence is not None or args.seq_file is not None, "error, the sequence is missing!"

    if args.sequence is not None:
        sequence = args.sequence
    else:
        sequence = "".join([l.strip() for l in open(args.seq_file) if not l.startswith(">")]).replace("T", "U")

    len_seq = len(sequence)
    global MIN_BP, MIN_HP, LEN_SEQ, SEQ_COMP, BP_ONLY, SEQ, MIN_NRJ, NB_MODE, PAD
    global MAX_STACK, MAX_BRANCH, VERBOSE
    BP_ONLY = args.bp_only
    MIN_BP = args.min_bp
    MIN_HP = args.min_hp
    MIN_NRJ = args.min_nrj
    LEN_SEQ = len_seq
    SEQ_COMP = fold_compound(sequence)
    SEQ = sequence
    NB_MODE = args.n_mode
    PAD = args.pad
    MAX_STACK = args.max_stack
    MAX_BRANCH = args.max_stack_branch
    VERBOSE = args.verbose

    # FOLDING -----------------------------------------------------------------
    pos_list = list(range(len_seq))
    # encode the sequence into 2 mirror strands
    eseq, cseq = prep_sequence(sequence, args.GC, args.AU, args.GU)
    all_struct = bfs_pairs([[((eseq, cseq, pos_list), None, [], 0.0)]])
    all_struct.sort(key=lambda el: eval_one_struct(SEQ_COMP, el, LEN_SEQ, SEQ))

    if args.one:
        all_struct = [all_struct[0]]

    if not VERBOSE:
        for pair_list in all_struct[::-1]:
            str_struct = dot_bracket(pair_list, len_seq)
            nrj_pred = SEQ_COMP.eval_structure(str_struct)

            if args.fasta:
                print(f"{str_struct} {nrj_pred:10.2f}")
            else:
                print(sequence, len_seq, str_struct, f"{nrj_pred:4.2f}", str_struct.count("("))


if __name__ == '__main__':
    main()

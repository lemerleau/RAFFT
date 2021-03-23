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
from utils import plot_bp_matrix, auto_cor, dot_bracket
from utils import prep_sequence, benchmark_vrna, eval_dynamic
from utils import get_inner_loop, get_outer_loop, eval_one_struct
import argparse
from RNA import fold_compound, PS_rna_plot

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
    list_sol = []
    for i in range(len_2):
        if pos < len_seq:
            ip, jp = i, pos-i
        else:
            ip, jp = pos-len_seq+1+i, len_seq-i-1

        # check if positions are contiguous
        # if i > 0:
        #     tot[i] = (tot[i-1]+tot[i])*tot[i]
        if i > 0 and pos_list[ip] - pos_list[ip-1] == 1 and \
           pos_list[jp+1] - pos_list[jp] == 1:
            tot[i] = (tot[i-1]+tot[i])*tot[i]

        if tot[i] == 0:
            tmp_max = 0
        else:
            tmp_max += 1

        # search for the highest number of consecutive BPs
        # and test if at least MIN_HP unpaired positions in between
        # if pos_list[jp] - pos_list[ip] > MIN_HP:
        if tot[i] >= max_score and pos_list[jp] - pos_list[ip] > MIN_HP:
            max_score = tot[i]
            max_nb = tmp_max
            max_i, max_j = ip, jp
            list_sol += [(max_nb, max_i, max_j, max_score)]

    # list_sol.sort(key=lambda el: el[0])
    # return list_sol[::-1]
    return max_nb, max_i, max_j, max_score


def find_best_consecutives(seq, cseq, pos_list, pair_list, cor_l):
    # find largest bp region
    max_bp, max_i, max_j, max_s, tmp_nrj = 0, 0, 0, 0, 0
    best_nrj = MIN_NRJ
    best_tmp = []
    for pos, c in cor_l[::-1][:NB_MODE]:
        mx_i, mip, mjp, ms = window_slide(seq, cseq, pos, pos_list)
        # for mx_i, mip, mjp, ms in window_slide(seq, cseq, pos, pos_list):

        if mx_i > 0:
            tmp_pair = [(pos_list[mip-i], pos_list[mjp+i]) for i in range(mx_i)]
            tmp_nrj = eval_dynamic(SEQ_COMP, pair_list, tmp_pair, LEN_SEQ, SEQ)
        else:
            tmp_pair = []
            tmp_nrj = 0

        # if ms > max_s:
        if best_nrj > tmp_nrj:
            max_bp, max_s, max_i, max_j = mx_i, ms, mip, mjp
            best_nrj = tmp_nrj
            best_tmp = tmp_pair
    return max_bp, max_s, max_i, max_j, best_nrj, best_tmp
        


def create_childs(seq, cseq, pair_list, pos_list, glob_list):
    """Recursive scheme
    """

    len_seq = seq.shape[1]
    cor_l = auto_cor(seq, cseq, PAD)
    cor_l.sort(key=lambda el: el[1])

    max_bp, max_s, max_i, max_j, best_nrj, tmp_pair = find_best_consecutives(seq, cseq, pos_list, pair_list, cor_l)

    # If no BP found, end the recursion
    if best_nrj >= MIN_NRJ:
        return

    # save the largest number of consecutive BPs
    for i in range(max_bp):
        pair_list += [(pos_list[max_i-i], pos_list[max_j+i])]

    # for el in pair_list:
    #     tmp_pair += [el]

    # print(" ", dot_bracket(pair_list, LEN_SEQ), eval_one_struct(SEQ_COMP, pair_list, LEN_SEQ, SEQ))
    # global FOLD_S
    # PS_rna_plot(SEQ, dot_bracket(pair_list, LEN_SEQ), f"scratch/img_fold/fold_{FOLD_S}.ps")
    # FOLD_S += 1


    # print("".join(SEQ[i] if i in pos_list else "-" for i in POS_LIST))
    # print(dot_bracket(pair_list, LEN_SEQ))
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

    cur_nrj = eval_one_struct(SEQ_COMP, pair_list, LEN_SEQ, SEQ)
    glob_list += [(in_side, out_side, pair_list, cur_nrj)]
    return glob_list


def bfs_pairs(glob_list):
    
    nb_el = len(glob_list)
    tmp_list = []
    for in_side, out_side, pair_list, cur_nrj in glob_list:

        if in_side is not None:
            iseq, icseq, ipos_list = in_side
            create_childs(iseq, icseq, pair_list, ipos_list, tmp_list)
        if out_side is not None:
            oseq, ocseq, opos_list = out_side
            create_childs(oseq, ocseq, pair_list, opos_list, tmp_list)
        # print(dot_bracket(pair_list, LEN_SEQ))

    if len(tmp_list) == 0:
        return dot_bracket(pair_list, LEN_SEQ)

    return bfs_pairs(tmp_list)


def parse_arguments():
    """Parsing command line
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--sequence', '-s', help="sequence")
    parser.add_argument('--seq_file', '-sf', help="sequence file")
    parser.add_argument('--struct', '-st', help="target structure to compare")
    parser.add_argument('--struct_file', '-stf', help="target structure file")
    parser.add_argument('--n_mode', '-n', help="number of mode to test during the search", type=int, default=200)
    parser.add_argument('--pad', '-p', help="padding, a normalization constant for the autocorrelation", type=float, default=1.0)
    parser.add_argument('--min_bp', '-mb', help="minimum bp to be detectable", type=int, default=1)
    parser.add_argument('--min_hp', '-mh', help="minimum unpaired positions in internal loops", type=int, default=3)
    parser.add_argument('--min_nrj', '-mn', help="minimum nrj loop", type=float, default=0)
    parser.add_argument('--bp_only', action="store_true", help="don't use the NRJ")
    parser.add_argument('--plot', action="store_true", help="plot bp matrix")
    parser.add_argument('--vrna', action="store_true", help="compare VRNA")
    parser.add_argument('--fasta', action="store_true", help="fasta output")
    parser.add_argument('--GC', type=float, help="GC weight", default=3.0)
    parser.add_argument('--AU', type=float, help="GC weight", default=2.0)
    parser.add_argument('--GU', type=float, help="GU weight", default=1.0)
    return parser.parse_args()


def main():
    args = parse_arguments()
    # HANDLE INPUTS -----------------------------------------------------------
    assert args.sequence is not None or args.seq_file is not None, "error, the sequence is missing!"
    init_struct = None
    if args.struct:
        init_struct = args.struct

    if args.struct_file:
        init_struct = "".join([l.strip() for l in open(args.struct) if not l.startswith(">")]).replace("T", "U")

    if args.sequence is not None:
        sequence = args.sequence
    else:
        sequence = "".join([l.strip() for l in open(args.seq_file) if not l.startswith(">")]).replace("T", "U")

    sequence = sequence.replace("N", "")
    len_seq = len(sequence)
    global MIN_BP, MIN_HP, LEN_SEQ, SEQ_FOLD, SEQ_COMP, BP_ONLY, SEQ, MIN_NRJ, OUT_DONE, GLOB_PAIRS, NB_MODE, PAD, POS_LIST, FOLD_S
    BP_ONLY = args.bp_only
    MIN_BP = args.min_bp
    MIN_HP = args.min_hp
    MIN_NRJ = args.min_nrj
    LEN_SEQ = len_seq
    SEQ_COMP = fold_compound(sequence)
    SEQ = sequence
    OUT_DONE = False
    NB_MODE = args.n_mode
    PAD = args.pad
    POS_LIST = list(range(len_seq))
    FOLD_S = 1

    # FOLDING -----------------------------------------------------------------
    pos_list = list(range(len_seq))
    # encode the sequence into 2 mirror strands
    eseq, cseq = prep_sequence(sequence, args.GC, args.AU, args.GU)
    str_struct = bfs_pairs([((eseq, cseq, pos_list), None, [], 0.0)])
    # pair_list = recursive_struct(eseq, cseq, [], pos_list, args.pad, args.n_mode)
    # str_struct = dot_bracket(pair_list, len_seq)
    nrj_pred = SEQ_COMP.eval_structure(str_struct)

    # FOR BENCHMARKS
    vrna_struct = None
    if args.vrna:
        vrna_struct, vrna_mfe, bp_dist = benchmark_vrna(sequence, str_struct)
        print(len_seq, vrna_mfe, nrj_pred, bp_dist, sequence, str_struct, vrna_struct)
    elif args.fasta:
        nb_bp = str_struct.count("(")
        # print(f">fft {nrj_pred:.2f} {nb_bp}")
        # print(f"{sequence}")
        # print(f"{str_struct}")
    else:
        print(sequence, len_seq, str_struct, nrj_pred, str_struct.count("("))

    if args.plot:
        if args.vrna:
            plot_bp_matrix(sequence, pair_list, vrna_struct)
        else:
            plot_bp_matrix(sequence, pair_list, init_struct)


if __name__ == '__main__':
    main()

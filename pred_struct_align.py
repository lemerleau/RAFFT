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
from align_utils import align_seq, backtracking
import argparse
from RNA import fold_compound

# window size when searching for pairs
PK_MODE = False
# Min number of BP
MIN_BP = 1
# Min number of internal unpaired positions
MIN_HP = 3
# Min nrj loop
MIN_NRJ = 0

CAN_PAIR = [('A' ,'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G')]

def recursive_struct(seq, cseq, pair_list, pos_list, pad=1, nb_mode=3):
    """Recursive scheme
    """
    len_seq = seq.shape[1]
    cor_l = [(i, 0) for i in range(len_seq*2 - 1)]
    cor_l = auto_cor(seq, cseq, pad)
    cor_l.sort(key=lambda el: el[1])

    max_bp, max_i, max_j, max_s, best_nrj, tmp_nrj = 0, 0, 0, 0, MIN_NRJ, MIN_NRJ
    best_pair_list = []
    for id_c, (pos, c) in enumerate(cor_l[::-1][:nb_mode]):

        if pos < len_seq:
            seq_ = seq[:, :pos+1]
            cseq_ = cseq[:, len_seq-pos-1:]
            pos_list_ = [el for el in pos_list[:pos+1]]
        else:
            seq_ = seq[:, pos-len_seq+1:]
            cseq_ = cseq[:, :2*len_seq-pos-1]
            pos_list_ = [el for el in pos_list[pos-len_seq+1:]]

        # find largest bp region
        dp_el = align_seq(seq_, cseq_, BULGE, SEQ, MIN_HP, pos_list_, MAX_BULGE)
        if dp_el is not None:
            if BP_ONLY:
                # use the number of BPs only
                tmp_nrj = -dp_el.score
            else:
                tmp_pair = [(pos_list[pi], pos_list[pj]) for pi, pj in backtracking(dp_el, len_seq, pos)]
                tmp_nrj = eval_dynamic(SEQ_COMP, pair_list, tmp_pair, LEN_SEQ)
        else:
            tmp_nrj = 0

        if dp_el is not None and best_nrj > tmp_nrj:
            best_nrj = tmp_nrj
            best_pair_list = backtracking(dp_el, len_seq, pos)
            in_i, in_j = best_pair_list[0]
            out_i, out_j = best_pair_list[-1]
            id_c_m = id_c

    if len(best_pair_list) <= MIN_BP:
        return pair_list

    # save the largest number of consecutive BPs
    for ip, jp in best_pair_list:
        assert (SEQ[pos_list[ip]], SEQ[pos_list[jp]]) in CAN_PAIR, "error {}".format((SEQ[pos_list[ip]], SEQ[pos_list[jp]]))
        pair_list += [(pos_list[ip], pos_list[jp])]

    if out_i > 1 or out_j < len_seq-1:
        # Outer loop case
        oseq = concatenate((seq[:, :out_i], seq[:, out_j+1:]), axis=1)
        ocseq = concatenate((cseq[:, :(len_seq-out_j-1)], cseq[:, (len_seq-(out_i)):]), axis=1)
        pos_list_2 = pos_list[:out_i] + pos_list[out_j+1:]
        recursive_struct(oseq, ocseq, pair_list, pos_list_2, pad, nb_mode)

    if in_j - in_i > 2:
        # Inner loop case
        oseq = seq[:, in_i+1:in_j]
        ocseq = cseq[:, (len_seq-in_j):(len_seq-in_i-1)]
        pos_list_2 = pos_list[in_i+1:in_j]
        recursive_struct(oseq, ocseq, pair_list, pos_list_2, pad, nb_mode)

    return pair_list


def parse_arguments():
    """Parsing command line
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--sequence', '-s', help="sequence")
    parser.add_argument('--seq_file', '-sf', help="sequence file")
    parser.add_argument('--struct', '-st', help="target structure to compare")
    parser.add_argument('--struct_file', '-stf', help="target structure file")
    parser.add_argument('--n_mode', '-n', help="number of mode to test during the search", type=int, default=20)
    parser.add_argument('--pad', '-p', help="padding, a normalization constant for the autocorrelation", type=float, default=1.0)
    parser.add_argument('--min_bp', '-mb', help="minimum bp to be detectable", type=int, default=3)
    parser.add_argument('--min_hp', '-mh', help="minimum unpaired positions in internal loops", type=int, default=3)
    parser.add_argument('--min_nrj', '-mn', help="minimum nrj loop", type=float, default=0)
    parser.add_argument('--fasta', action="store_true", help="fasta output")
    parser.add_argument('--bp_only', action="store_true", help="don't use the NRJ")
    parser.add_argument('--pk', action="store_true", help="pseudoknot")
    parser.add_argument('--plot', action="store_true", help="plot bp matrix")
    parser.add_argument('--vrna', action="store_true", help="compare VRNA")
    parser.add_argument('--GC', type=float, help="GC weight", default=3.0)
    parser.add_argument('--AU', type=float, help="GC weight", default=2.0)
    parser.add_argument('--GU', type=float, help="GU weight", default=1.0)
    parser.add_argument('--bulge', type=float, help="GU weight", default=-4.0)
    parser.add_argument('--max_b', '-mbu', help="max bulge", type=int, default=1)
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
    global PK_MODE, MIN_BP, MIN_HP, LEN_SEQ, SEQ_FOLD, SEQ_COMP, BP_ONLY, SEQ, BULGE, MAX_BULGE
    PK_MODE, BP_ONLY = args.pk, args.bp_only
    MIN_BP = args.min_bp
    MIN_HP = args.min_hp
    LEN_SEQ = len_seq
    SEQ_COMP = fold_compound(sequence)
    SEQ = sequence
    BULGE = args.bulge
    MAX_BULGE = args.max_b

    # FOLDING -----------------------------------------------------------------
    pos_list = list(range(len_seq))
    # encode the sequence into 2 mirror strands
    eseq, cseq = prep_sequence(sequence, args.GC, args.AU, args.GU)
    pair_list = recursive_struct(eseq, cseq, [], pos_list, args.pad, args.n_mode)
    str_struct = dot_bracket(pair_list, len_seq)
    nrj_pred = SEQ_COMP.eval_structure(str_struct)

    # FOR BENCHMARKS
    vrna_struct = None
    if args.vrna:
        vrna_struct, vrna_mfe, bp_dist = benchmark_vrna(sequence, str_struct)
        print(len_seq, vrna_mfe, nrj_pred, bp_dist, sequence, str_struct, vrna_struct)
    elif args.fasta:
        print(f">fft {nrj_pred}")
        print(f"{sequence}")
        print(f"{str_struct}")
    else:
        print(sequence, len_seq, str_struct, nrj_pred, str_struct.count("("))
        # print(sequence)
        # print(str_struct)
        # print("SCORE:", len(pair_list))
        # print("LEN:", len_seq)
        # print("VNRA_NRJ:", nrj_pred)

    if args.plot:
        if args.vrna:
            plot_bp_matrix(sequence, pair_list, vrna_struct)
        else:
            plot_bp_matrix(sequence, pair_list, init_struct)


if __name__ == '__main__':
    main()

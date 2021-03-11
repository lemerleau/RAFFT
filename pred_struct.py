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
from utils import prep_sequence, benchmark_vrna
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

    max_nb, tmp_max, max_score, max_i, max_j = 0, 0, 0, 0, 0
    for i in range(len_2):
        # print(i, tot.shape)
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


def recursive_struct(seq, cseq, pair_list, pos_list, pad=1, nb_mode=3,mle=0):
    """Recursive scheme
    """
    len_seq = seq.shape[1]
    cor_l = [(i, 0) for i in range(len_seq*2 - 1)]
    cor_l = auto_cor(seq, cseq, pad)
    cor_l.sort(key=lambda el: el[1])

    # find largest bp region with lowest free energy
    max_bp, max_i, max_j, max_s, min_loop_energy = 0, 0, 0, 0, mle
    for pos, c in cor_l[::-1][:nb_mode]:
        mx_i, mip, mjp, ms = window_slide(seq, cseq, pos, pos_list)

        #Update made by Nono to compute the energy of each region of bp
        if mx_i > 0 :
            tmp_bp = []
            for i in range(mx_i):
                tmp_bp += [(pos_list[mip-i], pos_list[mjp+i])]
            tmp_dot_bracket = dot_bracket(tmp_bp,len(SEQ))

            if (mjp+mx_i)> len(pos_list)-1 :
                tmp_seq = SEQ[pos_list[mip-mx_i+1]:]
                tmp_strc = tmp_dot_bracket[pos_list[mip-mx_i+1]:]
            else :
                tmp_seq = SEQ[pos_list[mip-mx_i+1]:pos_list[mjp+mx_i]]
                tmp_strc = tmp_dot_bracket[pos_list[mip-mx_i+1]:pos_list[mjp+mx_i]]

            tmp_fc = fold_compound(tmp_seq)
            tmp_mfe = tmp_fc.eval_structure(tmp_strc)

            #if ms > max_s: previous if statement.
            if tmp_mfe < min_loop_energy:
                max_bp, max_s, max_i, max_j = mx_i, ms, mip, mjp
                min_loop_energy = tmp_mfe
        ##########################End modif################################
    # If no BP found, end the recursion
    if max_bp < MIN_BP:
        return pair_list

    # save the largest number of consecutive BPs
    for i in range(max_bp):
        pair_list += [(pos_list[max_i-i], pos_list[max_j+i])]

    if PK_MODE:
        # If pseudoknot, merge both unpaired segments
        oseq = concatenate((seq[:, :max_i-max_bp+1], seq[:, max_i+1:max_j], seq[:, max_j+max_bp:]), axis=1)
        ocseq = concatenate((cseq[:, :len_seq - (max_j+max_bp)], cseq[:, len_seq-max_j:len_seq-max_i-1], cseq[:, len_seq-(max_i-max_bp+1):]), axis=1)
        pos_list_2 = pos_list[:max_i-max_bp+1] + pos_list[max_i+1:max_j] + pos_list[max_j+max_bp:]
        recursive_struct(oseq, ocseq, pair_list, pos_list_2, pad, nb_mode,mle)
    else:
        if max_i - (max_bp - 1) > 0 or max_j + max_bp < len_seq:
            # Outer loop case
            oseq = concatenate((seq[:, :max_i-max_bp+1], seq[:, max_j+max_bp:]), axis=1)
            ocseq = concatenate((cseq[:, :len_seq - (max_j+max_bp)], cseq[:, len_seq-(max_i-max_bp+1):]), axis=1)
            pos_list_2 = pos_list[:max_i-max_bp+1] + pos_list[max_j+max_bp:]
            recursive_struct(oseq, ocseq, pair_list, pos_list_2, pad, nb_mode,mle)

        if max_j - max_i > 1:
            # Inner loop case
            oseq = seq[:, max_i+1:max_j]
            ocseq = cseq[:, len_seq-max_j:len_seq-max_i-1]
            pos_list_2 = pos_list[max_i+1:max_j]
            recursive_struct(oseq, ocseq, pair_list, pos_list_2, pad, nb_mode,mle)

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
    parser.add_argument('--pk', action="store_true", help="pseudoknot")
    parser.add_argument('--plot', action="store_true", help="plot bp matrix")
    parser.add_argument('--vrna', action="store_true", help="compare VRNA")
    parser.add_argument('--GC', type=float, help="GC weight", default=1.0)
    parser.add_argument('--AU', type=float, help="GC weight", default=1.0)
    parser.add_argument('--GU', type=float, help="GU weight", default=1.0)
    parser.add_argument('--min_loop_energy', '-mle', help="minimum loop free energy to be detectable", type=int, default=0)
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
    global PK_MODE, MIN_BP, MIN_HP, LEN_SEQ, SEQ_FOLD, SEQ_COMP,SEQ #Update by Nono: Added SEQ as global var
    PK_MODE = args.pk
    MIN_BP = args.min_bp
    MIN_HP = args.min_hp
    LEN_SEQ = len_seq
    SEQ_COMP = fold_compound(sequence)
    SEQ = sequence #Set SEQ to be the sequence to fold.

    # FOLDING -----------------------------------------------------------------
    pos_list = list(range(len_seq))
    # encode the sequence into 2 mirror strands
    eseq, cseq = prep_sequence(sequence, args.GC, args.AU, args.GU)
    pair_list = recursive_struct(eseq, cseq, [], pos_list, args.pad, args.n_mode,args.min_loop_energy)
    str_struct = dot_bracket(pair_list, len_seq)
    nrj_pred = SEQ_COMP.eval_structure(str_struct)

    # FOR BENCHMARKS
    vrna_struct = None
    if args.vrna:
        vrna_struct, vrna_mfe, bp_dist = benchmark_vrna(sequence, str_struct)
        print(len_seq, vrna_mfe, nrj_pred, bp_dist, sequence, str_struct, vrna_struct)
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

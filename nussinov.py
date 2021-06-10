"""Nussinov secondary structure prediction

Maximize the correct pairs number for a given sequence.
"""

import argparse
#from utils import paired_positions #, benchmark_vrna
from RNA import fold_compound


# all the correct pairs
CORRECT_PAIRS = {
    ('A', 'U'): -1.0, ('U', 'A'): -1.0,
    ('G', 'C'): -1.0, ('C', 'G'): -1.0,
    ('G', 'U'): -1.0, ('U', 'G'): -1.0,
    }

def paired_positions(structure):
    "return a list of pairs (paired positions)"
    # save open bracket in piles
    pile_reg, pile_pk = [], []
    pairs = []
    for i, sstruc in enumerate(structure):
        if sstruc in ["<", "("]:
            pile_reg += [i]
        elif sstruc == "[":
            pile_pk += [i]
        elif sstruc in [">", ")"]:
            pairs += [(pile_reg.pop(), i)]
        elif sstruc == "]":
            pairs += [(pile_pk.pop(), i)]
    return pairs

def pair_nrj(nuc_a, nuc_b):
    try:
        return CORRECT_PAIRS[(nuc_a, nuc_b)]
    except KeyError:
        return 0.0


def fill_dp_matrix(rna_seq, theta=1):
    "fill the dynamic programming matrix"
    len_seq = len(rna_seq)
    rna_pos = list(range(len_seq))

    # create the empty matrix
    dp_matrix = [[0] * len_seq for _ in rna_pos]

    # 1) case where i is not paired
    case_no_pair = lambda ip, jp: dp_matrix[ip+1][jp]
    # 2) case where i and j are paired
    case_pair_ij = lambda ip, jp: dp_matrix[ip+1][jp-1] + pair_nrj(rna_seq[ip], rna_seq[jp])
    # 3) case where i and k are paired (i < k < j)
    case_pair_ik = lambda ip, jp, kp: dp_matrix[ip+1][kp-1] + dp_matrix[kp+1][jp] + pair_nrj(rna_seq[ip], rna_seq[kp])

    # fill the matrix
    for i in rna_pos[::-1]:
        for j in rna_pos[i+theta+1:]:
            options = [case_no_pair(i, j), case_pair_ij(i, j)]
            for k in range(i+theta+1, j):
                options += [case_pair_ik(i, j, k)]
            # choose the best option
            dp_matrix[i][j] = min(options)

    return dp_matrix


def backtrace(dp_matrix, ip, jp, theta, rna_seq):
    "bracktrace through the dynamic programming matrix"

    if (jp - ip) <= theta:
        return "." * (jp - ip + 1)
    else:
        # case where i is not paired
        if dp_matrix[ip][jp] == dp_matrix[ip+1][jp]:
            seq_1 = backtrace(dp_matrix, ip+1, jp, theta, rna_seq)
            return "." + seq_1

        if dp_matrix[ip][jp] == dp_matrix[ip+1][jp-1] + pair_nrj(rna_seq[ip], rna_seq[jp]):
            seq_1 = backtrace(dp_matrix, ip+1, jp-1, theta, rna_seq)
            return "(" + seq_1 + ")"

        for kp in range(ip + theta + 1, jp):
            if dp_matrix[ip][jp] == dp_matrix[ip+1][kp-1] + \
               dp_matrix[kp+1][jp] +\
               pair_nrj(rna_seq[ip], rna_seq[kp]):
                seq_1 = backtrace(dp_matrix, ip+1, kp-1, theta, rna_seq)
                seq_2 = backtrace(dp_matrix, kp+1, jp, theta, rna_seq)
                return "(" + seq_1 + ")" + seq_2


def parse_arguments():
    """Parsing command line
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--sequence', '-s', help="sequence")
    parser.add_argument('--seq_file', '-sf', help="sequence file")
    parser.add_argument('-t', '--theta', type=int, help="theta", default=3)
    parser.add_argument('--vrna', action="store_true", help="compare VRNA")
    return parser.parse_args()


def main():
    "testings"
    args = parse_arguments()
    if args.sequence is not None:
        sequence = args.sequence
    else:
        sequence = "".join([l.strip() for l in open(args.seq_file) if not l.startswith(">")]).replace("T", "U")

    len_seq = len(sequence)
    dp_mat = fill_dp_matrix(sequence, args.theta)

    second_struct = backtrace(dp_mat, 0, len(dp_mat)-1, args.theta, sequence)
    seq_comp = fold_compound(sequence)
    nrj_pred = seq_comp.eval_structure(second_struct)

    for pi, pj in paired_positions(second_struct):
        assert (sequence[pi], sequence[pj]) in CORRECT_PAIRS.keys(), "ERROR in prediction!"

    if args.vrna:
        vrna_struct, vrna_mfe, bp_dist = benchmark_vrna(sequence, second_struct)
        print(len_seq, vrna_mfe, nrj_pred, bp_dist, sequence, second_struct, vrna_struct)
    else:
        print(sequence, len_seq, second_struct, nrj_pred, second_struct.count("("))
        # print(sequence)
        # print(second_struct)
        # print("SCORE:", second_struct.count("("))
        # print("LEN:", len_seq)
        # print("VNRA_NRJ:", nrj_pred)


if __name__ == '__main__':
    main()

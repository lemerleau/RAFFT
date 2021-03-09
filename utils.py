"""Utils functions for the structure prediction
"""

from numpy import array, zeros, flip
from numpy import sum as npsum
from scipy.signal import convolve
import matplotlib.pyplot as plt
from RNA import fold, fold_compound, bp_distance


ENCODING = {"A": [1, 0, 0, 0], "G": [0, 1, 0, 0], "C": [0, 0, 1, 0], "U": [0, 0, 0, 1]}
CENCODING = {"A": [0, 0, 0, 1], "G": [0, 0, 1, 1], "C": [0, 1, 0, 0], "U": [1, 1, 0, 0]}
CAN_PAIR = [('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G')]

# the foward strand use the normal encoding
ENCODE = lambda s: array([ENCODING[n] for n in s])
# take the complementary nucleotides
C_ENCODE = lambda s: array([CENCODING[n] for n in s])


def dot_bracket(pair_list, len_seq):
    """convert the list of BPs into a dot bracket notation
    """
    str_struct = list("."*len_seq)
    for pi, pj in pair_list:
        str_struct[pi], str_struct[pj] = "(", ")"
    return "".join(str_struct)


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


def prep_sequence(sequence):
    """Encode the sequence into two mirror strands
    """
    e_seq = ENCODE(sequence).T
    c_seq = flip(C_ENCODE(sequence).T, axis=1)
    return e_seq, c_seq


def seq_conv(seq, cseq):
    "Compute the autocorrelation for the 4 components then sum per position"
    cseq = flip(cseq, axis=1)
    cor_ = []
    for i in range(4):
        # the convolution routine will use the fft if faster
        cor_ += [convolve(seq[i, ], cseq[i, ])]
    return npsum(array(cor_), axis=0)


def auto_cor(seq, cseq, pad):
    """Compute the auto correlation between the two strands
    """
    len_seq = seq.shape[1]
    cor = seq_conv(seq, cseq)
    norm = [el+pad for el in list(range(len_seq)) + list(range(len_seq-1))[::-1]]
    cor_l = [(i, c) for i, c in enumerate(cor/norm)]
    return cor_l


def plot_bp_matrix(sequence, pair_list, str_struct):
    """Plot the BP matrix
    """
    len_seq = len(sequence)
    mat = zeros((len_seq, len_seq))

    for pi, pj in pair_list:
        mat[pi, pj] = 1.0

    if str_struct is not None:
        for pi, pj in paired_positions(str_struct):
            mat[pj, pi] = 1.0

    fig = plt.figure(1)
    mat_f = fig.add_subplot(111)
    mat_f.set_xlim([-1, len_seq+1])
    mat_f.set_ylim([len_seq+1, -1])
    mat_f.imshow(mat, cmap="Greys")
    mat_f.plot([len_seq+1, 0], [len_seq+1, 0], linestyle="--", color="grey", linewidth=0.3)
    mat_f.grid(True, color="grey", linestyle="--", linewidth=0.2)
    plt.show()


def benchmark_vrna(sequence, pred_struct):
    "compute MFE structure and energy"
    vrna_struct, vrna_mfe = fold(sequence)
    bp_dist = bp_distance(pred_struct, vrna_struct)
    nb_bps_fft_st = pred_struct.count("(")
    nb_bps_vrn_st = vrna_struct.count("(")
    tot_nb = nb_bps_fft_st + nb_bps_vrn_st
    bp_dist = bp_distance(pred_struct, vrna_struct)
    if tot_nb > 0:
       norm_bp_dist = (tot_nb - bp_dist) / float(tot_nb)
    else:
        norm_bp_dist = 1.0
    return vrna_struct, vrna_mfe, norm_bp_dist

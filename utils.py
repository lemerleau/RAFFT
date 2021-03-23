"""Utils functions for the structure prediction
"""

from numpy import array, zeros, flip, concatenate
from numpy import sum as npsum
from numpy.fft import fft, ifft
from scipy.signal import convolve
import matplotlib.pyplot as plt
from RNA import fold, fold_compound, bp_distance



def dot_bracket(pair_list, len_seq, SEQ=None):
    """convert the list of BPs into a dot bracket notation
    """
    str_struct = list("."*len_seq)
    for pi, pj in pair_list:
        if SEQ is not None:
            print(SEQ[pi], SEQ[pj])
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


def prep_sequence(sequence, gc_wei=1.0, au_wei=1.0, gu_wei=1.0):
    """Encode the sequence into two mirror strands
    """
    ENCODING = {"A": [1., 0, 0, 0], "G": [0, 1., 0, 0], "C": [0, 0, 1., 0], "U": [0, 0, 0, 1.], ".": [0, 0, 0, 0]}
    CENCODING = {"A": [0, 0, 0, au_wei], "G": [0, 0, gc_wei, gu_wei], "C": [0, gc_wei, 0, 0], "U": [au_wei, gu_wei, 0, 0], ".": [0, 0, 0, 0]}
    CAN_PAIR = [('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G')]

    # the foward strand use the normal encoding
    ENCODE = lambda s: array([ENCODING[n] for n in s])
    # take the complementary nucleotides
    C_ENCODE = lambda s: array([CENCODING[n] for n in s])

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

def auto_cor_xy(seq, cseq, pad=1.0):
    """Compute the auto correlation between the two strands
    """
    len_seq = seq.shape[1]
    len_cseq = cseq.shape[1]
    cor = seq_conv(seq, cseq)
    norm = [(el+pad) for el in list(range(len_seq)) + list(range(len_cseq-1))[::-1]]
    cor_l = [(i, c) for i, c in enumerate(cor/norm)]
    return cor_l


def auto_cor(seq, cseq, pad=1.0):
    """Compute the auto correlation between the two strands
    """
    len_seq = seq.shape[1]
    cor = seq_conv(seq, cseq)
    norm = [(el+pad) for el in list(range(len_seq)) + list(range(len_seq-1))[::-1]]
    cor_l = [(i, c) for i, c in enumerate(cor/norm)]
    # cor_l = [(i, c) for i, c in enumerate(cor)]
    return cor_l


def auto_cor_test(seq, cseq, pad=1.0):
    """Compute the auto correlation between the two strands
    """
    len_seq = seq.shape[1]
    cor = seq_conv(seq, cseq)
    norm = [(el+pad) for el in list(range(len_seq)) + list(range(len_seq-1))[::-1]]
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


def MCC_bench(pred_struct, target_struct):
    pred_pl = set(paired_positions(pred_struct))
    target_pl = set(paired_positions(target_struct))
    pred_up = set([i for i, el in enumerate(pred_struct) if el == "."])
    target_up = set([i for i, el in enumerate(target_struct) if el == "."])
    true_pos = len(pred_pl & target_pl)  #  in both
    false_neg = len(target_pl-pred_pl)  # in true but not in pred struct
    false_pos = len(pred_pl-target_pl)  # in pred but not in true struct

    # put zero if not BP formed
    sensitivity = true_pos / (false_neg + true_pos) if (false_neg + true_pos) > 0 else 0.0
    pvv = true_pos / (true_pos + false_pos)
    if sensitivity  + pvv > 0:
        F1 = (2 * sensitivity * pvv)/(sensitivity  + pvv)
    else:
        F1 = 0.
    return sensitivity * 100.0, pvv * 100.0, F1 * 100


def eval_dynamic(seq_comp, pair_list, moves, len_seq, SEQ):
    "eval individual loop moves"
    dot_struct = dot_bracket(pair_list, len_seq)
    tmp_struct = dot_bracket(pair_list+moves, len_seq)
    return seq_comp.eval_structure(tmp_struct) - seq_comp.eval_structure(dot_struct)


def eval_one_struct(seq_comp, pair_list, len_seq, SEQ):
    "eval individual loop moves"
    dot_struct = dot_bracket(pair_list, len_seq)
    return seq_comp.eval_structure(dot_struct)


def get_outer_loop(seq, cseq, max_i, max_j, max_bp, pos_list, len_seq):
    oseq = concatenate((seq[:, :max_i-max_bp+1], seq[:, max_j+max_bp:]), axis=1)
    ocseq = concatenate((cseq[:, :len_seq - (max_j+max_bp)], cseq[:, len_seq-(max_i-max_bp+1):]), axis=1)
    pos_list_2 = pos_list[:max_i-max_bp+1] + pos_list[max_j+max_bp:]
    return oseq, ocseq, [el for el in pos_list_2]


def get_inner_loop(seq, cseq, max_i, max_j, max_bp, pos_list, len_seq):
    oseq = seq[:, max_i+1:max_j]
    ocseq = cseq[:, len_seq-max_j:len_seq-max_i-1]
    pos_list_2 = pos_list[max_i+1:max_j]
    return oseq, ocseq, [el for el in pos_list_2]


def merge_pair_list(pair_1, pair_2):
    pair_3 = set(pair_1) | set(pair_2)
    for el in pair_2:
        if el not in pair_1:
            pair_1 += [el]
    

def read_fasta(infile):
    results = {}
    for l in open(infile):
        if l.startswith(">"):
            name = l.strip()[1:]
            results[name] = ""
        else:
            results[name] += l.strip()
    return results

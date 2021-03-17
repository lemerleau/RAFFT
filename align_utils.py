"""Alignment functions
"""
from numpy import sum as npsum
from numpy import array, zeros, flip
from numpy import dot
from collections import namedtuple


Aligned_el = namedtuple("Aligned_el", ["score", "prev", "px", "py", "opt"])
CAN_PAIR = [('A' ,'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G')]


def align_seq(seq_x, cseq_x, penalty, SEQ, theta, pos_list, MAX_BULGE):
    "optimize the BP by aligning correlation"
    len_x = seq_x.shape[1]
    # len_x = int(seq_x.shape[1]/2) + seq_x.shape[1]%2

    dp_mat = [[Aligned_el(0, None, i, j, 0) for j in range(len_x+1)] for i in range(len_x+1)]

    # dp_mat = [[None for j in range(len_x+1)] for i in range(len_x+1)]
    # print("len, max", len_x, MAX_BULGE)
    # for i in range(0, len_x+1-MAX_BULGE+1):
    #     for j in range(max(0, i-MAX_BULGE), min(i+MAX_BULGE+2, len_x+1-i-theta+1)):
    #         dp_mat[i][j] = Aligned_el(0, None, i, j, 0)

    def comp_score(ss, cs, prev, pi, pj):
        sco = dot(ss, cs)
        # if (SEQ[pos_list[pi]], SEQ[pos_list[len_x-pj-1]]) not in CAN_PAIR and sco > 0.0:
        #     print("error")
        if sco <= 0:
            sco = -100000
        if abs(pos_list[pi] - pos_list[pi-1]) == 1 and abs(pos_list[pj+1] - pos_list[pj]) == 1:
            sco += prev
        return sco

    saved_max, saved_el = 0, None

    # for i in range(1, len_x+1):
    for i in range(1, len_x+1-MAX_BULGE):
        # for j in range(1, len_x+1-i-theta):
        for j in range(max(1, i-MAX_BULGE), min(i+MAX_BULGE+1, (len_x+1)-i-theta+1)):
            # if j - i > 5:
            #     pass
            options = [
                (1, dp_mat[i-1][j-1], comp_score(seq_x[:, i-1], cseq_x[:, j-1], dp_mat[i-1][j-1].score, i-1, j-1)),
                (2, dp_mat[i][j-1], dp_mat[i][j-1].score + penalty),
                (3, dp_mat[i-1][j], dp_mat[i-1][j].score + penalty),
                (4, None, 0)]
            options += []

            opt, prev, m_score = max(options, key=lambda el: el[2])

            dp_mat[i][j] = Aligned_el(m_score, prev, i, j, opt)

            if saved_el is None or saved_max <= m_score:
                saved_el = dp_mat[i][j]
                saved_max = m_score

    return saved_el


def backtracking(dp_el, len_seq, pos):
    "from the DP max value, backtrack the path"
    # Bactracking
    if dp_el is None:
        return []
    tmp = dp_el
    pair_list = []

    while True:
        if tmp.opt == 1:
            i, j = tmp.px-1, tmp.py - 1
            if pos < len_seq:
                ip, jp = i, pos-j
            else:
                ip, jp = pos-len_seq+1+i, len_seq-j-1
            pair_list += [(ip, jp)]
        if tmp.prev is None or tmp.prev == 0:
            break
        tmp = tmp.prev
    return pair_list

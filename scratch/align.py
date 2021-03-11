from numpy import sum as npsum
from collections import namedtuple


Aligned_el = namedtuple("Aligned_el", ["score", "prev", "px", "py", "opt"])


def align_seq(seq_x, cseq_x, seq_y, cseq_y, penalty):
    "optimize the BP by aligning correlation"
    len_x = int(seq_x.shape[1])
    len_y = int(seq_y.shape[1])
    cor_x = npsum(seq_x * cseq_x[:,::-1], axis=0)
    cor_y = npsum(seq_y * cseq_y[:,::-1], axis=0)
    print(cor_x)
    print(cor_y)

    dp_mat = [[None for _ in range(len_y+1)] for __ in range(len_x+1)]
    dp_mat[0][0] = Aligned_el(0, None, -1, -1, 0)

    # initialize the matrix
    for i in range(1, len_x+1):
        dp_mat[i][0] = Aligned_el(0, dp_mat[i-1][0], i, 0, 0)
    for j in range(1, len_y+1):
        dp_mat[0][j] = Aligned_el(0, dp_mat[0][j-1], 0, j, 0)

    saved_max, saved_el = 0, None
    for i in range(1, len_x+1):
        for j in range(1, len_y+1):
            options = [
                (1, dp_mat[i-1][j-1], dp_mat[i-1][j-1].score + cor_x[i-1] * cor_y[j-1]),
                (2, dp_mat[i][j-1], dp_mat[i][j-1].score + penalty),
                (3, dp_mat[i-1][j], dp_mat[i-1][j].score + penalty),
                (4, None, 0)
                ]

            opt, prev, m_score = max(options, key=lambda el: el[2])

            dp_mat[i][j] = Aligned_el(m_score, prev, i, j, opt)
            if saved_el is None or saved_max < m_score:
                saved_el = dp_mat[i][j]
                saved_max = m_score

    return saved_el


def backtracking(dp_el):
    "from the DP max value, backtrack the path"
    # Bactracking
    results = []
    tmp = dp_el
    while tmp is not None and tmp.score > 0:
        if tmp.opt in [1, 2]:
            px = tmp.px - 1
        else:
            px is None

        if tmp.opt in [1, 3]:
            py = tmp.py - 1
        else:
            py is None
        tmp = tmp.prev
        yield (px, py)

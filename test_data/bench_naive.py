"""Simple wrapper for benchmarks
"""

from os import popen

cmd_line = "python ../naive_pred_struct.py -s {} -mb 1 -mh 3"

print(f"seq,len_seq,struct,nrj,nbp")
for i, l in enumerate(open("test.seq")):
    seq, _ = l.strip().split()
    res = popen(cmd_line.format(seq))
    sequence, len_seq, pred_struct, nrj_pred, nb_bp = res.read().strip().split()
    print(f"{sequence},{len_seq},{pred_struct},{nrj_pred},{nb_bp}")

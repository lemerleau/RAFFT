"""Simple wrapper for benchmarks
bench.py np_procs out_file
"""

from multiprocessing import Pool
from os import popen
from sys import argv


cmd_line = "python ../pred_struct_al_stack.py -n 100 -ms 10 -mn 0 --one -s {}"
# cmd_line = "python ../pred_struct.py -s {} -n 50 -mb 2 -mh 4 --GC 1 --AU 1 --GU 1"


def run_bench(seq):
    res = popen(cmd_line.format(seq))
    sequence, len_seq, pred_struct, nrj_pred, nb_bp = res.read().strip().split()
    return f"{sequence},{len_seq},{pred_struct},{nrj_pred},{nb_bp}\n"


pool = Pool(int(argv[1]))
out_file = argv[2]
target_file = argv[3]
sequences = [l.strip().split(",")[0] for l in open(target_file) if len(l.strip().split(",")[0]) > 100 and len(l.strip().split(",")[0]) < 300]
# sequences = [l.strip().split(",")[0] for l in open(target_file)]
results = pool.map(run_bench, sequences)
pool.close()


with open(f"{out_file}", "w") as out:
    out.write(f"seq,len_seq,struct,nrj,nbp\n")
    for el in results:
        out.write(el)

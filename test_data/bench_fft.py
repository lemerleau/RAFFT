"""Simple wrapper for benchmarks
bench.py np_procs out_file
"""

from multiprocessing import Pool
from os import popen
from sys import argv


cmd_line = "python ../pred_struct.py -s {} -n 200 -mb 5 -mh 3 --GC 3 --AU 3 --GU 1"
# cmd_line = "python ../pred_struct.py -s {} -n 50 -mb 2 -mh 4 --GC 1 --AU 1 --GU 1"


def run_bench(arg):
    seq, _ = arg.strip().split(",")
    res = popen(cmd_line.format(seq))
    sequence, len_seq, pred_struct, nrj_pred, nb_bp = res.read().strip().split()
    return f"{sequence},{len_seq},{pred_struct},{nrj_pred},{nb_bp}\n"


pool = Pool(int(argv[1]))
out_file = argv[2]
target_file = argv[3]
results = pool.map(run_bench, (l for l in open(target_file)))
pool.close()


with open(f"{out_file}", "w") as out:
    out.write(f"seq,len_seq,struct,nrj,nbp\n")
    for el in results:
        out.write(el)

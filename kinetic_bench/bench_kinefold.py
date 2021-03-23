"""Benchmark with multiple structure in the output
"""

from multiprocessing import Pool
from os import popen
from sys import argv

cmd_line = "python ../kinfold.py -t 10000 -ns 50 -s {}"

def run_bench(args):
    seq, struct, name = args
    res = popen(cmd_line.format(seq))
    pred_struct = res.read().strip().split()
    return ",".join([seq, name, ",".join(pred_struct)])


pool = Pool(int(argv[1]))
out_file = argv[2]
target_file = argv[3]
sequences = [l.strip().split(",") for l in open(target_file)]
results = pool.map(run_bench, sequences)
pool.close()

with open(f"{out_file}", "w") as out:
    for el in results:
        out.write(el+"\n")

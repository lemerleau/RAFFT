"""score multiple structures for one input
"""

from multiprocessing import Pool
import subprocess
from math import isnan
import argparse
from os import system
from RNA import energy_of_struct


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


def read_true_struct(infile="./benchmark_cleaned_red_all_length.csv"):
    results = {}
    for l in open(infile):
        seq, struct, name = l.strip().split(",")
        results[seq] = (struct, name)
    return results


def read_csv(infile):
    results = []
    for l in open(infile):
        results += [l.strip().split(",")]
    return results


def create_ct_file(struct, sequence, out_file, name):
    nb_base = len(sequence)
    pair_list = paired_positions(struct)
    pair_co = {}
    for pi, pj in pair_list:
        pair_co[pi] = pj
        pair_co[pj] = pi

    with open(out_file, "w") as out:
        out.write(f"{nb_base} {name}\n")
        for i, nuc in enumerate(sequence):
            base_id = i+1
            id_m1, id_p1 = i, i+2
            bp_id = pair_co[i]+1 if i in pair_co else 0
            nat_id = base_id
            out.write(f"{base_id} {nuc} {id_m1} {id_p1} {bp_id} {nat_id}\n")

            
def read_log_file(infile):
    for l in open(infile):
        if l.startswith("PPV"):
            pvv = float(l.strip().split()[-1][:-1])
        if l.startswith("Sensitivity"):
            sensitivity = float(l.strip().split()[-1][:-1])
            # nothing predicted
            if isnan(sensitivity):
                sensitivity = 0.0
    return pvv, sensitivity


def test_one_seq(record):
    # cmd_line = "/home/vaitea/programs/RNAstructure/exe/scorer {} {} {}"
    cmd_line = "../RNAstructure/exe/scorer {} {} {}"
    raw_file = "../raw_data/archiveII/{}.ct"
    val = record
    seq, name, conf = val[0], val[1], val[2:]
    max_pvv, max_sens = 0, 0
    max_struct = "."*len(seq)
    for istruct, iscore in zip(range(len(conf))[::2], range(len(conf))[1::2]):
        struct, score = conf[istruct], conf[iscore]
        create_ct_file(struct, seq, f"./log/{name}_pred.ct", name+"_pred")
        pred_cmd_line = cmd_line.format(f"./log/{name}_pred.ct", raw_file.format(name), f"./log/{name}_pred.log").split()
        subprocess.Popen(pred_cmd_line, stdout=subprocess.PIPE, env={'DATAPATH': '/home/vaitea/programs/RNAstructure/data_tables/'}).communicate()
        pred_pvv, pred_sens = read_log_file(f"./log/{name}_pred.log")
        if pred_pvv >= max_pvv:
            max_pvv, max_sens, max_struct = pred_pvv, pred_sens, struct
    return max_pvv, max_sens, max_struct, seq, name



def parse_arguments():
    """Parsing command line
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('input_file', help="input")
    parser.add_argument('output_file', help="input")
    return parser.parse_args()


def main():
    args = parse_arguments()
    prediction = read_csv(args.input_file)
    true_str = read_true_struct()
    system("rm -r log")
    system("mkdir -p log")
    pool = Pool(4)
    # results = pool.map(test_one_seq, prediction[:2])
    results = pool.map(test_one_seq, prediction)

    with open(args.output_file, "w") as out:
        out.write("seq,len_seq,struct,nrj,nbp,pvv,sens,name\n")
        # for seq, (struct, name) in true_str.items():
        for pred_pvv, pred_sens, pred_struct, seq, name in results:
            # if seq in prediction:
            pred_nrj = energy_of_struct(seq, pred_struct)
            len_seq = len(seq)
            nbbp = pred_struct.count("(")
            out.write(f"{seq},{len_seq},{pred_struct},{pred_nrj},{nbbp},{pred_pvv},{pred_sens},{name}\n")

if __name__ == '__main__':
    main()

"""Simple wrapper for Kinfold software
"""


import argparse
from os import popen
from RNA import energy_of_struct


def density_struct(trajectory):
    results = {}
    for el in trajectory:
        # print(el)
        try:
            str_struct, nrj, time = el.strip().split()[:3]
            if str_struct in results:
                results[str_struct][0] += 1
            else:
                results[str_struct] = [1, float(nrj)]
        except:
            print(el)
    return results



def parse_arguments():
    """Parsing command line
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--sequence', '-s', help="sequence")
    parser.add_argument('--seq_file', '-sf', help="sequence file")
    parser.add_argument('-t', '--time', help="simulation time", type=int, default=500)
    parser.add_argument('-ns', '--nb_sim', help="number of trajectories", type=int, default=1)
    parser.add_argument('-np', '--nb_print', help="number of conformations to print", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.sequence is not None:
        sequence = args.sequence
    else:
        sequence = "".join([l.strip() for l in open(args.seq_file) if not l.startswith(">")]).replace("T", "U")
    len_seq = len(sequence)
    sim_time = args.time
    sim_nb = args.nb_sim
    cmd_line = f'echo "{sequence}" | Kinfold --time={sim_time} --num={sim_nb} --cut 0'
    results = popen(cmd_line)
    dens_str_ = density_struct(results.read().split("\n"))
    dens_str = [(struct, count) for struct, count in dens_str_.items()]
    dens_str.sort(key=lambda el: el[1][1])
    for struct, (count, nrj) in dens_str[:args.nb_print]:
        # print(struct, count, nrj)
        print(f"{struct} {count:10}")

    # kin_struct = results.read().split("\n")[-2].strip().split()[0]
    # kin_nrj = energy_of_struct(sequence, kin_struct)
    # print(sequence, len_seq, kin_struct, kin_nrj, kin_struct.count("("))


if __name__ == '__main__':
    main()

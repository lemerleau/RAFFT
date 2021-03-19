"""Simple wrapper for linear software
"""


import argparse
from os import popen
# from RNA import energy_of_struct
from uuid import uuid4


def parse_arguments():
    """Parsing command line
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--sequence', '-s', help="sequence")
    parser.add_argument('--seq_file', '-sf', help="sequence file")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.sequence is not None:
        sequence = args.sequence
    else:
        sequence = "".join([l.strip() for l in open(args.seq_file) if not l.startswith(">")]).replace("T", "U")
    len_seq = len(sequence)
    cmd_line = "echo {} | /home/vaitea/programs/LinearFold/linearfold".format(sequence)
    results = popen(cmd_line)
    try:
        linear_struct = results.read().split("\n")[1].split()[0]
        # linear_nrj = energy_of_struct(sequence, linear_struct)
        linear_nrj = 0.0
        linear_nbp = linear_struct.count("(")
    except:
        linear_nrj, linear_struct, linear_nbp = "Na", "Na", "Na"
    print sequence, len_seq, linear_struct, linear_nrj, linear_nbp


if __name__ == '__main__':
    main()

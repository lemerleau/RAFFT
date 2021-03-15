"""Simple wrapper for Kinfold software
"""


import argparse
from os import popen
from RNA import energy_of_struct


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
    cmd_line = f'echo "{sequence}" | Kinfold'
    results = popen(cmd_line)
    kin_struct = results.read().split("\n")[-2].strip().split()[0]
    kin_nrj = energy_of_struct(sequence, kin_struct)
    print(sequence, len_seq, kin_struct, kin_nrj, kin_struct.count("("))


if __name__ == '__main__':
    main()

"""Simple wrapper for mxfold software
"""


import argparse
from os import popen
from RNA import energy_of_struct
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
    seq_id = uuid4()
    with open(f"log/{seq_id}.seq", "w") as out:
        out.write(f">{seq_id}\n{sequence}")
    cmd_line = f'mxfold2 predict log/{seq_id}.seq'
    results = popen(cmd_line)
    try:
        mxfold_struct = results.read().split("\n")[-2].split()[0]
        mxfold_nrj = energy_of_struct(sequence, mxfold_struct)
        mxfold_nbp = mxfold_struct.count("(")
    except:
        mxfold_nrj, mxfold_struct, mxfold_nbp = "Na", "Na", "Na"
    print(sequence, len_seq, mxfold_struct, mxfold_nrj, mxfold_nbp)


if __name__ == '__main__':
    main()

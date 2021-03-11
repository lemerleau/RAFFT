import argparse
from RNA import fold

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
    vrna_struct, vrna_mfe = fold(sequence)
    print(sequence, len_seq, vrna_struct, vrna_mfe, vrna_struct.count("("))


if __name__ == '__main__':
    main()

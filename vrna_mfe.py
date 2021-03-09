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
    len_seq = len(args.sequence)
    vrna_struct, vrna_mfe = fold(args.sequence)
    print(args.sequence, len_seq, vrna_struct, vrna_mfe, vrna_struct.count("("))


if __name__ == '__main__':
    main()

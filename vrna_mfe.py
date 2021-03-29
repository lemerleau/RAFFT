import argparse
from RNA import fold, md, read_parameter_file


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
    
    # read_parameter_file("/home/vaitea/programs/ViennaRNA-2.4.17/misc/dna_mathews1999.par")
    # read_parameter_file("/home/vaitea/programs/ViennaRNA-2.4.17/misc/rna_langdon2018.par")
    # read_parameter_file("/home/vaitea/programs/ViennaRNA-2.4.17/misc/rna_andronescu2007.par")
    len_seq = len(sequence)
    vrna_struct, vrna_mfe = fold(sequence)
    print(sequence, len_seq, vrna_struct, vrna_mfe, vrna_struct.count("("))


if __name__ == '__main__':
    main()

from collections import namedtuple
from glob import glob
from random import sample
import re
import subprocess
from os import popen
from RNA import b2Shapiro, db_from_plist, PS_rna_plot
from re import findall
from math import isnan

Structp = namedtuple("Structp", ["base_id", "nuc", "id_m1", "id_p1", "bp_id", "nat_id"])


def read_true_struct(infile="./benchmark_results/benchmark_cleaned_all_length.csv"):
    results = {}
    for l in open(infile):
        seq, struct, name = l.strip().split(",")
        results[seq] = (struct, name)
    return results


def read_csv(infile):
    # results = []
    results = {}
    first = True
    for l in open(infile):
        if not first:
            seq, len_seq, struct, nrj, nb_bp, pvv, sens, name = l.strip().split(",")
            if struct != "Na":
                if isnan(float(pvv)):
                    pvv = 0.0
                results[seq] = (int(len_seq), struct, float(nrj), int(nb_bp), float(pvv), float(sens))
        else:
            first = False
    return results


def dot_bracket(pair_list, len_seq):
    """convert the list of BPs into a dot bracket notation
    """
    str_struct = list("."*len_seq)
    for pi, pj in pair_list:
        str_struct[pi], str_struct[pj] = "(", ")"
    return "".join(str_struct)


def read_seq(infile):
    elems = open(infile).readlines()
    name_, seq_ = elems[1].strip(), elems[2].strip()
    # return name_, seq_
    return seq_[:-1].upper()


def get_bp_list(struct):
    bp_list = []
    for el in struct:
        if el.bp_id is not "0":
            pi, pj = int(el.base_id)-1, int(el.bp_id)-1
            if (pi, pj) not in bp_list and (pj, pi) not in bp_list:
                bp_list += [(pi, pj)]
    return bp_list


def read_ct(infile):
    elems = open(infile).readlines()
    nb_base, name = elems[0].split()[0], "_".join(elems[0].split()[1:])
    struct = []
    for el in elems[1:]:
        base_id, nuc, id_m1, id_p1, bp_id, nat_id = el.split()
        struct += [Structp(base_id, nuc, id_m1, id_p1, bp_id, nat_id)]

    assert int(nb_base) == len(struct), "error nb base"
    return struct


def dotb_from_ct(infile):
    cmd_line = f"/home/vaitea/programs/ViennaRNA-2.4.17/src/Utils/ct2db < {infile}"
    res = popen(cmd_line)
    struct = res.read().strip().split()[1]
    return struct


def get_loop_content(struct):
    shap = b2Shapiro(struct)
    interior = findall(r"I\d+", shap)
    stack = findall(r"S\d+", shap)
    multi = findall(r"M\d+", shap)
    hairpin = findall(r"H\d+", shap)
    ext_loop = findall(r"E\d+", shap)
    bulge = findall(r"B\d+", shap)
    interior_nb = sum(int(el[1:]) for el in interior)
    stack_nb    = sum(int(el[1:]) for el in stack)
    multi_nb    = sum(int(el[1:]) for el in multi)
    hairpin_nb  = sum(int(el[1:]) for el in hairpin)
    ext_loop_nb = sum(int(el[1:]) for el in ext_loop)
    bulge_nb  = sum(int(el[1:]) for el in bulge)
    tot_ = interior_nb +stack_nb +multi_nb +hairpin_nb +ext_loop_nb+bulge_nb
    try:
        return float(interior_nb)/tot_, float(stack_nb)/tot_, float(multi_nb)/tot_, float(hairpin_nb)/tot_, float(ext_loop_nb)/tot_, float(bulge_nb)/tot_
    except:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

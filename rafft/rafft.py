#!/usr/bin/env python3
"""
RAFFT is a folding tool that builds fast-folding paths for a given sequence.
Starting from the completely unfolded structure, it quickly identifies stems
with an FFT-based technique. Then, forms them if they improve the overall
stability in a sequential fashion. Multiple folding paths can be explored and
displayed.

Usage:
To display only the k=10 structures found at the end
python rafft.py [-s <SEQ> | -sf <SEQ_FILE>] -ms 10

To display the k=10 visited at each folding steps
python rafft.py [-s <SEQ> | -sf <SEQ_FILE>] -ms 10 --verbose

Inputs:
-s <SEQ> is the sequence given in the standard input
-sf <SEQ_FILE> is a fasta formatted file (or a simple text file with only the
 sequence in it)

The algorithm has two critical parameters:
-ms <INT> is the number of saved structures at each folding steps (default=1)
-n <INT> is the number of positional lag to search for stems (default=100)

"""

import argparse
from numpy import sum as npsum
from rafft.utils import auto_cor, dot_bracket
from rafft.utils import prep_sequence
from rafft.utils import get_inner_loop, get_outer_loop, eval_one_struct
from rafft.utils import merge_pair_list
from itertools import product
from RNA import fold_compound, md


class Glob_parms:
    "Store all non redundant information"

    def __init__(self, sequence, nb_mode, max_stack, max_branch, min_hp,
                 min_nrj, traj, temp, gc_wei, au_wei, gu_wei):
        self.sequence, self.temp, self.nb_mode, = sequence, temp, nb_mode,
        self.max_stack, self.min_hp, self.min_nrj = max_stack, min_hp, min_nrj
        self.traj, self.temp, self.max_branch = traj, temp, max_branch
        self.gc_wei, self.au_wei, self.gu_wei = gc_wei, au_wei, gu_wei
        self.model = md()
        self.model.temperature = temp
        self.len_seq = len(sequence)
        self.seq_comp = fold_compound(sequence, self.model)


class Node:
    "unpaired regions"

    def __init__(self, forward, backward, unpaired_pos):
        self.forward, self.backward = forward, backward
        self.pos_list = unpaired_pos


class Structure:
    "A structure is modeled as a tree; in bfs, the tree is a list of nodes"

    def __init__(self, node_list, pair_list):
        self.node_list = node_list
        self.energy = 0.0
        self.pair_list = pair_list
        self.str_struct = ""


def window_slide(seq, cseq, pos, pos_list, min_hp):
    """Slide a window along the align pair of sequence to find the consecutive paired positions
    """
    len_seq = seq.shape[1]
    # the position of the spike gives the relative position between both
    # strands
    if pos < len_seq:
        seq_ = seq[:, :pos+1]
        cseq_ = cseq[:, len_seq-pos-1:]
    else:
        seq_ = seq[:, pos-len_seq+1:]
        cseq_ = cseq[:, :2*len_seq-pos-1]

    # test if it represents the average bp
    # tmp = mean(npsum(seq_*cseq_, axis=0))
    # print(tmp == cor, pos, len_seq)
    len_2 = int(seq_.shape[1]/2) + seq_.shape[1] % 2

    # When the strands are appropriatly aligned, we have to search for base
    # pairs with a sliding window
    tot = npsum(seq_[:, :len_2]*cseq_[:, :len_2], axis=0)

    # search for consecutive BPs
    max_nb, tmp_max, max_score, max_i, max_j = 0, 0, 0, 0, 0
    for i in range(len_2):
        if pos < len_seq:
            ip, jp = i, pos-i
        else:
            ip, jp = pos-len_seq+1+i, len_seq-i-1

        # check if positions are contiguous
        if i > 0 and pos_list[ip] - pos_list[ip-1] == 1 and \
           pos_list[jp+1] - pos_list[jp] == 1:
            tot[i] = (tot[i-1]+tot[i])*tot[i]

        if tot[i] == 0:
            tmp_max = 0
        else:
            tmp_max += 1

        # search for the highest number of consecutive BPs
        # and test if at least MIN_HP unpaired positions in between
        if tot[i] >= max_score and pos_list[jp] - pos_list[ip] > min_hp:
            max_score = tot[i]
            max_nb = tmp_max
            max_i, max_j = ip, jp

    return max_nb, max_i, max_j, max_score


def find_best_consecutives(cor_l, upair, cur_str, glob_parms):
    # find largest bp region
    best_sol = []
    max_bp, max_i, max_j, max_s, tmp_nrj = 0, 0, 0, 0, glob_parms.min_nrj
    best_nrj = glob_parms.min_nrj

    for pos, c in cor_l[::-1][:glob_parms.nb_mode]:
        mx_i, mip, mjp, ms = window_slide(upair.forward, upair.backward, pos,
                                          upair.pos_list, glob_parms.min_hp)

        if mx_i > 0:
            tmp_pair = [(upair.pos_list[mip-i], upair.pos_list[mjp+i]) for i in range(mx_i)]
            tmp_nrj = eval_one_struct(cur_str.pair_list+tmp_pair, glob_parms) - cur_str.energy
        else:
            tmp_nrj = glob_parms.min_nrj

        if tmp_nrj < glob_parms.min_nrj:
            max_bp, max_s, max_i, max_j = mx_i, ms, mip, mjp
            best_tmp = tmp_pair
            best_nrj = tmp_nrj
            best_sol += [(max_bp, max_s, max_i, max_j, best_nrj, best_tmp)]

    best_sol.sort(key=lambda el: el[4])
    return best_sol


def create_childs(upair, cur_str, glob_parms):
    """Recursive scheme
    """

    len_seq = upair.forward.shape[1]
    cor_l = auto_cor(upair.forward, upair.backward)
    cor_l.sort(key=lambda el: el[1])

    best_solutions = find_best_consecutives(cor_l, upair, cur_str, glob_parms)

    cur_list_sol = []
    for solution in best_solutions:
        # save the largest number of consecutive BPs
        max_bp, max_s, max_i, max_j, best_nrj, best_tmp = solution
        best_nrj += cur_str.energy
        for el in cur_str.pair_list:
            best_tmp += [el]

        if max_j - max_i > 1:
            # Inner loop case
            iseq, icseq, ipos_list_2 = get_inner_loop(upair.forward,
                                                      upair.backward, max_i,
                                                      max_j, max_bp,
                                                      upair.pos_list,
                                                      len_seq)
            in_side = Node(iseq, icseq, ipos_list_2)
        else:
            in_side = None

        if max_i - (max_bp - 1) > 0 or max_j + max_bp < len_seq:
            # Outer loop case
            oseq, ocseq, opos_list_2 = get_outer_loop(upair.forward,
                                                      upair.backward, max_i,
                                                      max_j, max_bp,
                                                      upair.pos_list,
                                                      len_seq)
            # print("out", oseq.shape, ocseq.shape)
            out_side = Node(oseq, ocseq, opos_list_2)
        else:
            out_side = None
        cur_list_sol += [(in_side, out_side, best_tmp, best_nrj)]
    return cur_list_sol


def bfs_pairs(glob_tree, glob_parms, step=0, glob_traj=[], seen=set()):
    """Bread-first procedure to create helices.
    """
    tmp_glob_tree = []
    new_glob_tree = []
    glob_traj += [glob_tree]

    # split current nodes
    for struct in glob_tree:
        tmp_tree = []
        for un_paired in struct.node_list:
            # create possible helices from the unpaired region
            cur_list = create_childs(un_paired, struct, glob_parms)

            if len(cur_list) > 0:
                tmp_tree += [cur_list]

        if len(tmp_tree) > 0:
            tmp_glob_tree += [tmp_tree]

    # Combine stems formed in independent sub segments
    nb_branch = 0
    for helices in tmp_glob_tree:
        # a comp is a combination of helices
        for helix in product(*helices):
            tmp_tree = Structure(node_list=[], pair_list=[])
            # one helix split the unpaired region in two part in_side/out_side
            for in_side, out_side, tmp_pairs, tmp_nrj in helix:
                # merge the formed helix (all of them are independent)
                merge_pair_list(tmp_tree.pair_list, tmp_pairs)

                if in_side is not None:
                    tmp_tree.node_list += [in_side]
                if out_side is not None:
                    tmp_tree.node_list += [out_side]

            new_nrj = eval_one_struct(tmp_tree.pair_list, glob_parms)
            tmp_tree.energy = new_nrj
            tmp_str = dot_bracket(tmp_tree.pair_list, glob_parms.len_seq)

            if tmp_str not in seen:
                tmp_tree.str_struct = tmp_str
                new_glob_tree += [tmp_tree]
                nb_branch += 1
                seen.add(tmp_str)

            if nb_branch >= glob_parms.max_branch:
                break

    # sort by energy
    new_glob_tree += glob_tree
    new_glob_tree.sort(key=lambda el: el.energy)

    # Save the best trajectories among all the combinations of helices
    new_glob_tree = new_glob_tree[:glob_parms.max_stack]

    # test if the same structures are found
    if [st.str_struct for st in glob_tree] == [st.str_struct for st in new_glob_tree]:
        return glob_tree, glob_traj

    return bfs_pairs(new_glob_tree, glob_parms, step+1, glob_traj, seen)


def fold(sequence, nb_mode=100, max_stack=1, max_branch=100, min_hp=3,
         min_nrj=0.0, traj=False, temp=37.0, gc_wei=3.0, au_wei=2.0,
         gu_wei=1.0):
    "fold a given sequence"
    glob_parms = Glob_parms(sequence, nb_mode, max_stack, max_branch, min_hp,
                            min_nrj, traj, temp, gc_wei, au_wei, gu_wei)
    
    pos_list = list(range(glob_parms.len_seq))

    eseq, cseq = prep_sequence(sequence, gc_wei, au_wei, gu_wei)
    init_node = Node(eseq, cseq, pos_list)
    unfold_struct = Structure(node_list=[init_node], pair_list=[])
    unfold_struct.str_struct = "."*glob_parms.len_seq

    structures, trajectory = bfs_pairs([unfold_struct], glob_parms,
                                       step=0, glob_traj=[], seen=set())

    if traj:
        return structures, trajectory
    else:
        return structures

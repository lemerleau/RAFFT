import argparse
from numpy import sum as npsum
from rafft.utils import auto_cor, dot_bracket
from rafft.utils import prep_sequence
from rafft.utils import get_inner_loop, get_outer_loop, eval_one_struct
from rafft.utils import merge_pair_list
from itertools import product
from RNA import fold_compound, md
from rafft.rafft import window_slide


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


class Structure :

    def __init__(self, bpList=[], node_list=[]):

        self.energy = 0.0
        self.bpList = bpList
        self.str_struct =""
        self.children = []
        self.node_list = node_list

    def __str__(self, level=0):
        ret = "\t"*level+repr(self.str_struct)+" level:"+str(level)+" \n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret

    def __repr__(self) :
        return '<Tree Node representation>'

    def buildStructure(self,added_pairs, seq_comp, len_seq) :
        pair_list = added_pairs + self.bpList
        str_struct = list("."*len_seq)
        for pi, pj in pair_list:
            str_struct[pi], str_struct[pj] = "(", ")"
        str_struct = "".join(str_struct)
        return str_struct, seq_comp.eval_structure(str_struct)


    def create_nodes(self,node, glob_parms) :

        nodes = []
        (X, X_bar, pos_list) = node.forward, node.backward, node.pos_list
        len_seq = X.shape[1]
        cor_l = auto_cor(X, X_bar)
        cor_l.sort(key=lambda el: el[1])

        max_bp, max_i, max_j, max_s, tmp_nrj = 0, 0, 0, 0, glob_parms.min_nrj

        for pos, c in cor_l[::-1][:glob_parms.nb_mode]:
            mx_i, mip, mjp, ms = window_slide(X, X_bar, pos,
                                              pos_list, glob_parms.min_hp)
            if mx_i > 0 :
                tmp_pairs = [(pos_list[mip-i], pos_list[mjp+i]) for i in range(mx_i)]
                tmp_strc, tmp_energy = self.buildStructure(tmp_pairs, glob_parms.seq_comp, glob_parms.len_seq)

                if tmp_energy - self.energy < glob_parms.min_nrj :
                    max_bp, max_i, max_j, max_s = mx_i, mip, mjp, ms
                    tmp_pairs = self.bpList + tmp_pairs
                    if max_j - max_i > 1 :
                        X_child, X_child_bar, child_pos_list = get_inner_loop(X,
                                                                  X_bar, max_i,
                                                                  max_j, max_bp,
                                                                  pos_list,
                                                                  len_seq)
                        in_ = Node(X_child, X_child_bar, child_pos_list)
                    else :
                        in_ = None

                    if max_i - (max_bp - 1) > 0 or max_j + max_bp < len_seq:
                        # Outer loop case
                        X_Ochild, X_Ochild_bar, O_child_pos_list = get_outer_loop(X,
                                                                  X_bar, max_i,
                                                                  max_j, max_bp,
                                                                  pos_list,
                                                                  len_seq)
                        out_ =Node(X_Ochild, X_Ochild_bar, O_child_pos_list)
                    else:
                        out_ = None
                    nodes += [(in_, out_, tmp_pairs, tmp_energy)]


        return nodes


def bfs_pairs(root,structures, glob_parms, step=0, seen=set()):

    # split current nodes
    all_children = []
    new_structures = []
    for structure in structures:
        tmp_children= []
        for node in structure.node_list :

            if node is not None :
                curr_children = structure.create_nodes(node, glob_parms)

                if len(curr_children) > 0:
                    tmp_children += [curr_children]

        if len(tmp_children) > 0 :
            all_children += [(structure,tmp_children)]

    # Combine stems formed in independent sub segments
    nb_branch = 0
    for structure,children in all_children:
        # a comp is a combination of helices
        new_children = []
        for children_pair in product(*children):
            # one helix split the unpaired region in two part in_side/out_side
            new_structure = Structure(node_list=[], bpList=[])
            for child in children_pair:
                in_side, out_side, tmp_pairs, tmp_nrj = child
                # merge the formed helix (all of them are independent)
                merge_pair_list(new_structure.bpList, tmp_pairs)
                new_structure.node_list +=[in_side, out_side]


            sigma, new_nrj = new_structure.buildStructure(added_pairs=[],
                                                    seq_comp=glob_parms.seq_comp,
                                                    len_seq=glob_parms.len_seq
                                                    )

            if sigma not in seen:
                new_structure.str_struct = sigma
                new_structure.energy = new_nrj
                new_structures += [new_structure]
                new_children +=[new_structure]
                nb_branch += 1
                seen.add(sigma)

            if nb_branch >= glob_parms.max_branch:
                break
        if len(new_children) > 0 :
            new_children.sort(key=lambda el: el.energy)
            structure.children = new_children[:glob_parms.max_stack]

    # sort by energy
    new_structures += structures
    new_structures.sort(key=lambda el: el.energy)

    # Save the best trajectories among all the combinations of helices
    new_structures = new_structures[:glob_parms.max_stack]

    #remove all the children that did not survive
    for structure in structures :
        for child in structure.children :
            if child not in new_structures :
                structure.children.remove(child)

    # test if the same structures are found
    if [node.str_struct for node in structures] == [node.str_struct for node in new_structures]:
        return structures, root

    return bfs_pairs(root,new_structures, glob_parms, step+1, seen)


def fold(sequence, nb_mode=100, max_stack=1, max_branch=100, min_hp=3,
         min_nrj=0.0, traj=False, temp=37.0, gc_wei=3.0, au_wei=2.0,
         gu_wei=1.0):
    "fold a given sequence"
    glob_parms = Glob_parms(sequence, nb_mode, max_stack, max_branch, min_hp,
                            min_nrj, traj, temp, gc_wei, au_wei, gu_wei)

    pos_list = list(range(glob_parms.len_seq))

    eseq, cseq = prep_sequence(sequence, gc_wei, au_wei, gu_wei)
    init_node = Node(eseq, cseq, pos_list)
    unfold_struct = Structure(node_list=[init_node], bpList=[])
    unfold_struct.str_struct = "."*glob_parms.len_seq

    structures = bfs_pairs(unfold_struct,[unfold_struct], glob_parms)

    return structures

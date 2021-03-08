"""Simple wrapper for benchmarks
"""

from os import popen

print(f"len_seq,vrna_mfe,nrj_pred,hamm_dist,sequence,pred_struct,mfe_struct,nb_bp_pred,nb_bp_mfe")
for i, l in enumerate(open("test.seq")):
    seq, struct = l.strip().split()
    res = popen(f"python ../pred_struct.py -s {seq}  -n 10 -mb 1 -mh 3 --vrna")
    len_seq_, vrna_mfe_, nrj_pred_, hamm_dist_, sequence, pred_struct, mfe_struct = res.read().strip().split()
    len_seq, vrna_mfe, nrj_pred, hamm_dist = int(len_seq_), float(vrna_mfe_), float(nrj_pred_), float(hamm_dist_)
    hamm_dist = 100.0*((len_seq - hamm_dist)/float(len_seq))
    nb_bp_pred, nb_bp_mfe = pred_struct.count("("), mfe_struct.count("(")
    print(f"{len_seq},{vrna_mfe:.1f},{nrj_pred:.1f},{hamm_dist:.1f},{sequence},{pred_struct},{mfe_struct},{nb_bp_pred},{nb_bp_mfe}")

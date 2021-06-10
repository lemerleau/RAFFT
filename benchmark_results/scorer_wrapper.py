"""
    RUN this in the RNAstructure/exe/
"""


from os import popen, remove, environ, path, mkdir
from pandas import DataFrame, read_csv
from uuid import uuid4
from argparse import ArgumentParser,RawTextHelpFormatter
import json

def getScores(predicted, accepted, sequence) :

    RNAstructure_ROOT = environ.get("RNAstructure_ROOT")
    if not path.exists(RNAstructure_ROOT+"exe/tmp") :
        mkdir(RNAstructure_ROOT+"exe/tmp")


    dot2ct = RNAstructure_ROOT+"exe/./dot2ct "+RNAstructure_ROOT+"exe/tmp/{}.fasta "+RNAstructure_ROOT+"exe/tmp/{}.ct"
    id_ = str(uuid4())

    with open(RNAstructure_ROOT+"exe/tmp/"+id_+".pred.fasta", 'w') as fle :
        fle.write('>'+str(id_)+'\n')
        fle.write(sequence+"\n")
        fle.write(predicted)
    fle.close()

    rst = popen(dot2ct.format(id_+".pred",id_+".pred"))
    rst.close()
    with open(RNAstructure_ROOT+"exe/tmp/"+id_+".acc.fasta", 'w') as fle :
        fle.write('>'+str(id_)+'\n')
        fle.write(sequence+"\n")
        fle.write(accepted)
    fle.close()
    rst = popen(dot2ct.format(id_+".acc",id_+".acc"))
    rst.close()

    scorer_cmd = RNAstructure_ROOT+"exe/./scorer "+RNAstructure_ROOT+"exe/tmp/{}.pred.ct "+RNAstructure_ROOT+"exe/tmp/{}.acc.ct  "+RNAstructure_ROOT+"exe/tmp/{}.score"

    rst = popen(scorer_cmd.format(id_, id_, id_))
    rst.close()
    with open(RNAstructure_ROOT+"exe/tmp/"+id_+".score",'r') as fle :
        score_out = fle.read().split()
        scores = list(filter(lambda el: '%' in el, score_out))

    fle.close()
    remove(RNAstructure_ROOT+"exe/tmp/"+id_+".pred.fasta")
    remove(RNAstructure_ROOT+"exe/tmp/"+id_+".acc.fasta")
    remove(RNAstructure_ROOT+"exe/tmp/"+id_+".pred.ct")
    remove(RNAstructure_ROOT+"exe/tmp/"+id_+".acc.ct")
    remove(RNAstructure_ROOT+"exe/tmp/"+id_+".score")

    print ("Sensitivity = ", float(scores[0][:-1]), " PPV = ", float(scores[1][:-1]))
    return float(scores[0][:-1]), float(scores[1][:-1])


def main() :
    #Best nrj RAFFT output with ms=50 n = 100
    """
    mfe_rafft_data = read_csv('rafft_mfe.csv')
    finale_data = []
    for i,line in enumerate(mfe_rafft_data.values) :
        print (i)
        sensitivity, ppv = getScores(line[3], line[4], line[2])
        list_ = list(line[1:]) + [sensitivity, ppv ]
        finale_data.append(list_)


    df = DataFrame(finale_data, columns=['name','sequence', 'mfe_prediction', 'native', 'energy', 'length', 'nbp', 'sensitivity', 'ppv'])
    df.to_csv('rafftMFE_data.csv',index=False)

    mfe_rafft_data = read_csv('rafft_mfe.csv')
    with open("full_100n_50ms.json") as fe :
        json_data = json.load(fe)

    #Best ppv RAFFT output with ms=50 n = 100
    finale_data = []
    for i,line in enumerate(mfe_rafft_data.values) :
        print (i)

        strcs = json_data[line[2]]
        sensitivity, ppv_best = getScores(strcs[0].split()[0], line[4], line[2])
        best = list(line[1:]) + [sensitivity, ppv_best ]
        for strc in strcs[1:] :
            sensitivity, ppv = getScores(strc.split()[0], line[4], line[2])
            if ppv > ppv_best :
                ppv_best = ppv
                best = [line[1], line[2], strc.split()[0] ,line[4],float(strc.split()[1]), line[6], strc.split()[0].count('(') ] + [sensitivity, ppv_best ]

        finale_data.append(best)


    df = DataFrame(finale_data, columns=['name','sequence', 'mfe_prediction', 'native', 'energy', 'length', 'nbp', 'sensitivity', 'ppv'])
    df.to_csv('rafft_best_ppv.csv',index=False)
    """
    p = "(((((((((......(((((....((..((((((.......))..))))..)).....))))).....(((((.((((.(((((...))))).)))).)).)))..)))))))))."
    a = "((((((((.....(.(((((......((((((.............))))..)).....)))))..).((..(....((.(((((...))))).))....)..))...))))))))."
    s = "UUAAGUGACGAUAGCCUAGGAGAUACACCUGUUCCCAUGCCGAACACAGAAGUUAAGCCCUAGUACGCCUGAUGUAGUUGGGGGUUGCCCCCUGUUAGAUACGGUAGUCGCUUAGC"
    print (getScores(p, a,s))






if __name__ =="__main__" :
    main()

"""
    @authors : nono&vaitea

"""

#Import the necessary labaries
from numpy import array
import sys
from pandas import read_csv,DataFrame
from os import environ
from RNA import fold_compound
from subprocess import Popen, PIPE
from multiprocess import Pool,cpu_count


def fold_with_contextfold(seq):

    p = Popen(["java",'-cp',str(environ.get('CONTEXT_FOLD'))+"/bin","contextFold.app.Predict", "in:"+''.join(seq),"model:"+str(environ.get('CONTEXT_FOLD'))+"/trained/StHighCoHigh.model"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    rst, err = p.communicate()

    if err :
        print(err,"ERROR during the execution of contextFold.app.Predict program: please make sure the evironment variable CONTEXT_FOLD is set correctly")
        return (None, 0.0);
    else :
        tmp_fc = fold_compound(seq)
        strc = rst.split()[1].decode()
        mfe = tmp_fc.eval_structure(str(strc))
        assert len(strc)==len(seq)

        return (strc, mfe)

def ppcontextFold(listOfSeqs) :

    pool = Pool(cpu_count())
    result = array(pool.map(fold_with_contextfold, listOfSeqs))
    pool.close()

    return list(result[:,0]),list(result[:,1])



def main():

    df = read_csv("benchmark_cleaned_all_length.csv")
    sequences = df.values[:,0]

    strcs, energies = ppcontextFold(sequences)

    df_saver = DataFrame(array([sequences,strcs,energies]).T, columns=["Sequence","Structure","Energy"])
    df_saver.to_csv('contextFold_out.csv')





if __name__=="__main__":
    main()

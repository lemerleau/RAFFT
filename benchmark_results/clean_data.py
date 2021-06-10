import pandas as pd
import numpy as np
import json






def main() :
    #Load the benchmark data
    bench_data = pd.read_csv('benchmark_cleaned_all_length.csv')

    #Load RAFFT output
    with open('rafft_nono.out') as fle :
        fft_data = fle.read().split('\n')
        print(len(fft_data))
    fft_data.remove('')

    # Getting the total sequences folded
    total_seq = []
    for i in range(len(fft_data)) :
        if 'A' in list(fft_data[i]) :
            total_seq.append([i,fft_data[i]])
    try:
        assert len(total_seq) == len(bench_data.values)
    except Exception as e:
        print(len(total_seq),len(bench_data.values))
    data = []
    for i,lst in enumerate(total_seq) :
        #print(i,total_seq[i+1][0]-lst[0])
        try :
            data.append(fft_data[lst[0]:total_seq[i+1][0]])
        except IndexError :
            data.append(fft_data[lst[0]:])


    full_benchDATA = {}

    for lst in data :
        print (lst[0])
        full_benchDATA[lst[0]] = lst[1:]
    with open("full_100n_50ms.json", 'w') as fle :
        json.dump(full_benchDATA,fle)

    mfe_fft_structures  = []

    for key in full_benchDATA.keys():
        list_ = full_benchDATA[key][0].split()
        filted_df = bench_data[bench_data['sequence']==key].values
        if len(filted_df)> 1 :
            print('ERROOOOORRR')
            break

        mfe_fft_structures.append([filted_df[0][-1], key, list_[0], filted_df[0][1], float(list_[1]), len(key), list_[0].count('(')])

    pd.DataFrame(mfe_fft_structures, columns=['name','sequence', 'mfe_prediction', 'native', 'energy', 'length', 'nbp']).to_csv('rafft_mfe.csv')



if __name__=="__main__" :
    main()

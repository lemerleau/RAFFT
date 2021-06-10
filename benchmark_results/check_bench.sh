#!/bin/bash 

for seq in `cat benchmark_cleaned_all_length.csv | cut -f 1 -d ','`; 

do  
	echo $seq;
       `python ../rafft.py -n 100 -ms 50 -s $seq >> rafft_nono.out`;
	       
done ; 


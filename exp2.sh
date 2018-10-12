#!/usr/local/bin/bash
for i in 1000 2000 4000 8000 16000 32000; do
    printf "iteration $i cpu\n" >> exp2.txt
    ./heatdist 1000 $i 0  >> exp2.txt
    printf "iteration $i gpu\n" >> exp2.txt
    ./heatdist 1000 $i 1  >> exp2.txt
done

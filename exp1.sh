#!/usr/local/bin/bash
for i in 1000 2000 4000 8000 16000; do
    ./heatdist $i 1000 0  >> exp1.txt
    ./heatdist $i 1000 1  >> exp1.txt
done

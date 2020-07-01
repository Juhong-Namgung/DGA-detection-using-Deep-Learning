#!/bin/bash

wget https://osint.bambenekconsulting.com/feeds/dga-feed.txt

sed -n "16,\$p" dga-feed.txt > dga.txt

rm dga-feed.txt

:<<'END'
echo "domain,label" > dga.csv

sed -n "16,\$p" dga-feed.txt | while read line

do

        dga=$(echo $line | awk -F ',' '{print $1}')
        label=$(echo $line | awk -F ' ' '{print $4}')

        echo "$dga,$label" >> ./dga.csv

done
END


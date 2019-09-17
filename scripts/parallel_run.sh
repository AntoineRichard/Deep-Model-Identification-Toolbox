#!/bin/bash
file=$1

cat /cs-share/dream/RSS-Kingfisher/training_script/$file | shuf | while read line 
do
    nice $line
done


#!/bin/bash
file=$1

cat $file | shuf | while read line 
do
    nice $line
done






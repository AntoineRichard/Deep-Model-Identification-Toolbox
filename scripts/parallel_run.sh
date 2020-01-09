#!/bin/bash

source /home/dream_user/.bashrc
set -e

file=$1
rm -f /tmp/parallel_run.log
cat $file | shuf | while read line 
do
    nice $line 2>&1 | tee -a /tmp/parallel_run.log
done






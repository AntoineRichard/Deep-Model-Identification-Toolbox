# How to use DeepID

## Training locally one network at a time

We provide 2 scripts to show you how to train networks using DeepID:
 - `sample\_local\_run.sh`
 - `sample\_local\_run\_per.sh`

They contain a command line to train a basic MLP with or without priorization.
To change the model, the priorization and other, please have a look at the README.md located 
in the network folder.

## Training a whole gridsearch

We provide all the scripts we used to generate our grid-searches under the script folder. Select 
one you like edit it and execute it.

Once it is done you have 2 options:
 - Running the grid-search locally: Use the `parallel_run.sh` and give the path to the previously generated text file.
 - Running on a cluster: (TODO see sample_parallel-ssh.sh)

We usually run everything on a computer room which gives us about 25 slaves.

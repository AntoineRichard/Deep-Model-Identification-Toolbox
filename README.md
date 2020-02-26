# Deep ID : a simple deep-learning based identification toolbox

Deep ID features all of the most common, and state of the art architectures for deep system identification.
It comes with a command line model generator, for MLPs, 1D CNNs, RNNs, LSTMs, GRUs and Attention based models.
Additionnaly this toolbox provide support for advanced data handling such as: continuous-time seq2seq, prioritized
 experience replay, and gradient upper-bound priorization schemes.
Finally, Deep ID implements the most common tools for proper model evaluation such has k-fold Cross Validation,
singlestep and multistep error evaluation along with a TensorBoard backend for visualization. 
With over 35 parameters available from command line Deep ID is extremely flexible making grid searches for optimal
parameters easy and efficient. More about Deep ID below.

## Using DeepID

This section covers how to use the DeepID toolbox.

### Prerequisite

To run DeepID you need:
- TensorFlow v1.14.0 (and the associated CUDA/CUDNN requirements)
- Scikit-Learn
- Numpy
- Python3 **(REQUIRED FOR TRAINING ONLY)**
- Compatibility with 

This Framework was sucessfully tested on the
following system architectures PPC64, AARCH64 and x64. 

Please note that for the simplest model it might be worth to train on a CPU instead of a GPU.

### Running DeepID
To run DeepID issue the following command:

 - ``python3 run_training.py ''A list of arguments''``
 - ``python3 run_training.py --help`` gives you the exhaustive list of arguments.

To learn more about the available arguments please refere to the ReadMe located under networks.
A small example of how to train your own models is available in the ReadMe located under example.

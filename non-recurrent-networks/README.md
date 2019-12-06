# Deep ID : a simple deep-learning based identification toolbox

Deep ID features all of the most common, and state of the art architectures for deep system identification.
It comes with a command line model generator, for MLPs, 1D CNNs, RNNs, LSTMs, GRUs and Attention based models.
Additionnaly this toolbox provide support for advanced data handling such as continuous-time seq2seq.
Finally, Deep ID implements the most common tools for proper model evaluation such has k-fold Cross Validation,
singlestep and multistep error evaluation along with a TensorBoard backend for visualization. 
With over 40 parameters available from command line Deep ID is extremely flexible making grid searches for optimal
parameters easy and efficient. More about Deep ID below.

## Using DeepID

This section covers how to use the framework. The framework itself is detailed in section Framework, the models
are defined in the section Models.

###

## The Framework

This section covers the different element of the framework and how they interact. As of now the framework
is articulated around 5 main blocks:
- An argument Parser: settings.py
- Datasets Readers: reader.py
- Samplers: sampler.py
- Training Loops: train.py
- A network generator: network\_generator.py models.py

## Models

This section covers the available models and how to invoke them from the command line argument.

### Available Models
- MLPs
- Complex valued MLPs
- 1D CNNs
- Attention Networks
- RNNs
- LSTMs
- GRUs
- Custom

> Future plans include wavenet like network and adding atrous convolution support to 1D CNN. And possibly
 attention based LSTMs/GRUs.

### Available Activation Function
- ReLu
- Leaky ReLu
- Tanh
- Sigmoid

> Future plans include parametric ReLus and Exponential ReLus.

### MLP Models
For the MLP the command line generator support 2 layers:
- Dense layer : 'd'
- Dropout layer : 'r'

To generate a model the user has to use the --model argument and add a string after it.
Each word that are going to compose this string have to be separated by an underscore.
The first word defines the model type, in our case: MLP. The second defines the type of
activation funtion that are going to be used in the model. Then the user must specify the 
layers he wants to use in a sequential order.


In the end you can specify a MLP model as follows:

- --model MLP\_RELU\_d32\_d32
- --model MLP\_TANH\_r\_d16
- --model MLP\_LRELU\_d64\_r\_d128\_r\_d16

> Future plans include batch normalization.


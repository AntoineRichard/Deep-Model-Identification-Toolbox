# The Framework

## Overview

This section covers the different element of the framework and how they interact. As of now the framework
is articulated around 6 blocks:
- An argument Parser: settings.py
- Datasets Readers: reader.py
- Samplers: sampler.py
- Models: models.py
- Training Loops: train.py
- A network generator: network\_generator.py

## Settings

This section covers the different parameters that can be adjusted through command line arguments.
To ease the use of this tool, we rely on argparse to parse the arguments provided by the user. 
Please find thereafter the list of available arguments.


### Data-Loading arguments

The framework offers 3 options to load data. However all those options require the data to be 
in the HDF5 format with the following structure:
- Each line is a new data point
- The data must be chronologically ordered
- On a line, the data must have the following format: TimeStamp (optional), States, Commands
- Additionally the data can contain a continuity bit: TimeStamp (optional), Continuity\_bit (optional), States, Commands.

Theoritically the timestamp and the continuity bit can be placed anywhere in the array, yet it is better to place them as shown above.

No matter how many datafile you want to use, they have to be placed inside a folder (even if there is only one)

There are 3 data options:
- **3 Folders**: for train, test and validation. To use this option simply add the following arguments:
   - **--train\_data** */Path/2/Train/Folder* ``str``
   - **--test\_data** */Path/2/Test/Folder* ``str``
   - **--val\_data** */Path/2/Val/Folder* ``str``
- **Ratio**: where the test in a ratio of the original data, and the val a ratio of the remaining data.
to use this option add the following arguments. Where TestRatio and ValRatio are values between 0 and 1. Setting the TestRatio to 0.2
would mean that 20% of the data is going to be allocated to the test set. Setting the ValRatio to 0.1 would 
mean that out of the reaming 80% of the data 10% are going to be allocated to the validation set.
   - **--train\_data** */Path/2/Train/Folder* ``str``
   - **--test\_ratio** *TestRatio* ``float``
   - **--val\_ratio** *ValRatio* ``float``
- **Cross Validation**: this option applies the well known cross validation onto the data. To use this mode add
the following arguments. The number of folds corresponds to the number of slices your data
is going to be cutted into, and the Test and Validation indexes corresponds to which slices are going to be used
for test and validation. Please not that the validation index can be equal to the test index, and both data will be
different. More about cross validation on Google.
   - **--train\_data** *Path/2/Train/Folder* ``str``
   - **--use\_cross\_validation** *True* ``bool``
   - **--folds** *NumberOfFolds* ``int``
   - **--test\_idx** *TestIndex* ``int``
   - **--val\_idx** *ValidationIndex* ``int``

### Data-Settings arguments

The formating of the data is made whithin thr reader. it can be modified through the following parameters:

- **--max\_sequence\_size** *SizeOfTheSequence* ``int``. The SizeOfTheSequence corresponds to the size of the
 history of State/Commands that are going to be fed to the network. Typical value 12.
- **--forecast** *SizeForecast* ``int``. The amount of forecast step that the network should be trained for. 
Typical value 1.
- **--trajectory\_length** *TrajectoryLength* ``int``. The number of iterative forecasting step that will be
performed in multistep evaluation. Typical value 20.
- **--input\_dim** *InputDim* ``int``. The size of the input vector (states dimensions + commands dimensions).
- **--output\_dim** *OutputDim* ``int``. The size of the output vector (states dimensions)
- **--timestamp\_idx** *TimestampIndex* ``int`` **OPTIONAL**. The index of the TimeStamp; if the dataset does not contain
any do not set this argument. 
- **--continuity\_idx** *ContinuityIndex* ``int`` **OPTIONAL**. The index of the continuity bit; if the dataset does not
contain any do not set this argument.

### Training-Settings arguments

- **--batch\_size** *BatchSize* ``int``. The size of the training batch. Typical value 64.
- **--val\_batch\_size** *ValidationBatchSize* ``int`` **OPTIONAL**. The size of the batch when validating. Typical value 2000.
 If not set it will use the whole of the validation set.
- **--test\_batch\_size** *TestBatchSize* ``int`` **OPTIONAL**. The size of the batch when testing. Typical value 2000. If not set
it will use the whole of the test set.
- **--max\_iterations** *MaxIteration* ``int``. The number of iteration to run for. Typical value 30000.
- **--log\_frequency** *LogFrequency* ``int``. The Loging frequency (how often the network performances are going
 to be evaluated). Typical value 25.
- **--learning\_rate** *LearningRate* ``float``. The learning rate value: Typical value 0.001.
- **--dropout** *Dropout* ``float`` **OPTIONAL**. The dropout value if it is not used then don't set it.



> Please note that as of today forecasting farther than 1 time step is not well supported.
Please leave this setting to one.

##Models

This section covers the available models and how to invoke them from the command line argument.

### Available Models
- MLPs (and variation around state systems)
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
- ReLU
- Leaky ReLU 
- Parametric ReLU (P-ReLU)
- Exponential Linear Unit (ELU)
- Scaled ELU (SELU)
- SWISH
- CSWISH (constant version of SWISH)
- MISH
- Tanh
- Sigmoid

### MLP Models
For the MLP the command line generator support 2 layers:
- Dense layer : 'd' followed by a number: d256, d48, d12
                    where the number indicates the number of
                    neurons in that layer.
- Dropout layer : 'r', the keeprate, is defined when starting
                      the training.

To generate a model the user has to use the --model argument and add a string after it.
Each word that are going to compose this string have to be separated by an underscore.
The first word defines the model type, in our case: MLP. The second defines the type of
activation funtion that are going to be used in the model. Then the user must specify the 
layers he wants to use in a sequential order.


In the end you can specify a MLP model as follows:

- --model MLP\_RELU\_d32\_d32
- --model MLP\_TANH\_r\_d16
- --model MLP\_LRELU\_d64\_r\_d128\_r\_d16

> The model **automatically casts to the right number of output** using the --output\_dim argument provided by the user.
> Future plans include batch normalization.

### PHYSICAL MLP Models
These models optimise a linear model and a non-linear function such that: 
X_{t+1} = A*X_t + B*U_t + F(X_{t..t-n}, U_{t..t-n})
Where A and B or optimized to minimize the error between the prediction
 and the real value, and F is optimized to minimize the prediction error
 of the linear model.

For the Physical MLP the command line generator support 2 layers:
- Dense layer : 'd' followed by a number: d256, d48, d12
                    where the number indicates the number of
                    neurons in that layer.
- Dropout layer : 'r', the keeprate, is defined when starting
                      the training.

To generate a model the user has to use the --model argument and add a string after it.
Each word that are going to compose this string have to be separated by an underscore.
The first word defines the model type, in our case: MLP. The second defines the type of
activation funtion that are going to be used in the model. Then the user must specify the 
layers he wants to use in a sequential order.


In the end you can specify a Physical MLP model as follows:

- --model MLPPHY\_RELU\_d32\_d32
- --model MLPPHY\_TANH\_r\_d16
- --model MLPPHY\_LRELU\_d64\_r\_d128\_r\_d16

> The model **automatically casts to the right number of output** using the --output\_dim argument provided by the user.
> Future plans include batch normalization.

### Complex MLP Models
For the Complex value MLP the command line generator support 2 layers:
- Dense layer : 'd' followed by a number: d256, d48, d12
                    where the number indicates the number of
                    neurons in that layer.
- Dropout layer : 'r', the keeprate, is defined when starting
                      the training.

To generate a model the user has to use the --model argument and add a string after it.
Each word that are going to compose this string have to be separated by an underscore.
The first word defines the model type, in our case: a CPLXMLP. The second defines the type of
activation funtion that are going to be used in the model. Then the user must specify the 
layers he wants to use in a sequential order.


In the end you can specify a Complex valued MLP model as follows:

- --model CPLXMLP\_RELU\_d32\_d32
- --model CPLXMLP\_TANH\_r\_d16
- --model CPLXMLP\_LRELU\_d64\_r\_d128\_r\_d16

> The model **automatically casts to the right number of output** using the --output\_dim argument provided by the user.
> Future plans include batch normalization.

### 1D CNN Models
For the 1D CNNs the command line generator support 4 layers:
- Convolution layers: 'k' followed by a number, followed by 'c',
                          followed by another number: k3c64 is a 
                          convolution of kernel size 3 with a depth
                          of 64. 
- Pooling layers: 'p' followed by a number: p2 is a pooling layer with
                      a kernel size of 2, with stride 2.
- Dense layers: 'd' followed by a number: d256, d48, d12
                    where the number indicates the number of
                    neurons in that layer.
- Dropout layers: 'r', the keeprate, is defined when starting
                      the training.

To generate a model the user has to use the --model argument and add a string after it.
Each word that are going to compose this string have to be separated by an underscore.
The first word defines the model type, in our case: CNN. The second defines the type of
activation funtion that are going to be used in the model. Then the user must specify the 
layers he wants to use in a sequential order.


In the end you can specify a CNN model as follows:

- --model CNN\_RELU\_k3c64\_p2\_d32
- --model CNN\_TANH\_k5c32\_p2\_k3c128\_r\_d16
- --model CNN\_LRELU\_k5c64\_r\_d128\_r\_d16

> Please note that when using a dense layer for the first time the output of the feature extractor is automically flatenned. Hence one cannot do: --model CNN\_TANH\_k3c32\_d16\_k3c32. Also one must specify at least one dense layer after the feature extractor. Hence one cannot do: --model CNN\_TANH\_k3c32.
> The model **automatically casts to the right number of output** using the --output\_dim argument provided by the user.
> Future plans include batch normalization, and Fully Convolutional support.

### RNN Models
For the RNNs the command line generator support 2 layers:
- Dense layers: 'd' followed by a number: d256, d48, d12
                    where the number indicates the number of
                    neurons in that layer.
- Dropout layers: 'r', the keeprate, is defined when starting
                      the training.
Additionnaly the command line generator must be provided with 2 parameters:
- Hidden State Size: 'hs' followed by a number: hs32 means a
                         hidden state of size 32, use this parameter
                         only once. If set multipletime then the
                         parser will select the lastest.
- Recurrent layers: 'l' followed by a number: l3 means that 3 
                        recurrent layers are being stacked ontop
                        of each other.
    
To generate a model the user has to use the --model argument and add a string after it.
Each word that are going to compose this string have to be separated by an underscore.
The first word defines the model type, in our case: RNN. The second defines the type of
activation funtion that are going to be used in the model. Then the user must specify the 
layers he wants to use in a sequential order.

RNN being a reccurent architecture, the framework allows for specific options. More specifically, the RNN has been ``rolled'' in two steps. A first step rolls the first element of the sequence when the second step rolls the rest. This is practical if one uses continuous-time RNN has it allows to save the future next hidden-state.
  
In the end you can specify a RNN model as follows:
- --model RNN\_LRELU\_hs32\_l1\_d16\_d16
- --model RNN\_TANH\_hs128\_l3\_d32 
- --model RNN\_RELU\_hs64\_l2\_d64\_r\_d32

> The model **automatically casts to the right number of output** using the --output\_dim argument provided by the user.

### GRU Models
For the GRUs the command line generator support 2 layers:
- Dense layers: 'd' followed by a number: d256, d48, d12
                    where the number indicates the number of
                    neurons in that layer.
- Dropout layers: 'r', the keeprate, is defined when starting
                      the training.
Additionnaly the command line generator must be provided with 2 parameters:
- Hidden State Size: 'hs' followed by a number: hs32 means a
                         hidden state of size 32, use this parameter
                         only once. If set multipletime then the
                         parser will select the lastest.
- Recurrent layers: 'l' followed by a number: l3 means that 3 
                        recurrent layers are being stacked ontop
                        of each other.
    
To generate a model the user has to use the --model argument and add a string after it.
Each word that are going to compose this string have to be separated by an underscore.
The first word defines the model type, in our case: GRU. The second defines the type of
activation funtion that are going to be used in the model. Then the user must specify the 
layers he wants to use in a sequential order.

GRU being a reccurent architecture, the framework allows for specific options. More specifically, the GRU has been ``rolled'' in two steps. A first step rolls the first element of the sequence when the second step rolls the rest. This is practical if one uses continuous-time GRU has it allows to save the future next hidden-state.
  
In the end you can specify a GRU model as follows:
- --model GRU\_LRELU\_hs32\_l1\_d16\_d16
- --model GRU\_TANH\_hs128\_l3\_d32 
- --model GRU\_RELU\_hs64\_l2\_d64\_r\_d32

> The model **automatically casts to the right number of output** using the --output\_dim argument provided by the user.

### LSTM Models
For the LSTMs the command line generator support 2 layers:
- Dense layers: 'd' followed by a number: d256, d48, d12
                    where the number indicates the number of
                    neurons in that layer.
- Dropout layers: 'r', the keeprate, is defined when starting
                      the training.
Additionnaly the command line generator must be provided with 2 parameters:
- Hidden State Size: 'hs' followed by a number: hs32 means a
                         hidden state of size 32, use this parameter
                         only once. If set multipletime then the
                         parser will select the lastest.
- Recurrent layers: 'l' followed by a number: l3 means that 3 
                        recurrent layers are being stacked ontop
                        of each other.
    
To generate a model the user has to use the --model argument and add a string after it.
Each word that are going to compose this string have to be separated by an underscore.
The first word defines the model type, in our case: LSTM. The second defines the type of
activation funtion that are going to be used in the model. Then the user must specify the 
layers he wants to use in a sequential order.

LSTM being a reccurent architecture, the framework allows for specific options. More specifically, the LSTM has been ``rolled'' in two steps. A first step rolls the first element of the sequence when the second step rolls the rest. This is practical if one uses continuous-time LSTM has it allows to save the future next hidden-state.
  
In the end you can specify a LSTM model as follows:
- --model LSTM_LRELU_hs32_l1_d16_d16
- --model LSTM_TANH_hs128_l3_d32 
- --model LSTM_RELU_hs64_l2_d64_r_d32

> The model **automatically casts to the right number of output** using the --output\_dim argument provided by the user.

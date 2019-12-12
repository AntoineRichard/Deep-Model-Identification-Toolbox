# Deep ID : a simple deep-learning based identification toolbox

Deep ID features all of the most common, and state of the art architectures for deep system identification.
It comes with a command line model generator, for MLPs, 1D CNNs, RNNs, LSTMs, GRUs and Attention based models.
Additionnaly this toolbox provide support for advanced data handling such as continuous-time seq2seq.
Finally, Deep ID implements the most common tools for proper model evaluation such has k-fold Cross Validation,
singlestep and multistep error evaluation along with a TensorBoard backend for visualization. 
With over 35 parameters available from command line Deep ID is extremely flexible making grid searches for optimal
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
- Models: models.py
- Training Loops: train.py
- A network generator: network\_generator.py

## Settings

This section covers the different parameters that can be adjusted through command line arguments.

### General principle

To ease the use of this tool we rely on argparse to parse the arguments provided by the user.
To feed the argument and start training use the following command:

python3 rules.py ''The list of arguments''

### Data-Loading arguments

The framework offers 3 options to load data. However all those options requires the data to be 
in the HDF5 format with the following structure:
- Each line is a new data point
- The data must be chronologically ordered
- On a line the data must have the following format: TimeStamp (optional), States, Commands
- Additionally the data can contain a continuity idx: TimeStamp (optional), Continuity\_idx (optional), States, Commands.

Theoritically the timestamp and the continuity idx can be placed anywhere in the array, yet it is better to place them as shown above.

No matter how many datafile you want to train one they have to be placed inside a folder (even if there is only one)

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


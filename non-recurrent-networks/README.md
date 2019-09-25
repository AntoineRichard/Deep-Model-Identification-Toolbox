# Non-Recurrent-networks models

## Model list:
### MLP Models
To define and train a MLP the name of the model has to start by "MLP" and be followed by a underscore ("\_").
Then you have to define the activation function that you want to use among the following functions:

- "RELU" : defined as tf.nn.relu
- "TANH": defined as tf.nn.tanh
- "SIGMOID" : defined as tf.nn.sigmoid
- "LRELU" : defined as tf.nn.leaky\_relu

You then have to specify the size of each of the dense layer composing the MLP and separate them with an underscore.

In the end you can specify a MLP model as follows:

- --model MLP\_RELU\_32\_32
- --model MLP\_TANH\_16
- --model MLP\_LRELU\_64\_128\_16
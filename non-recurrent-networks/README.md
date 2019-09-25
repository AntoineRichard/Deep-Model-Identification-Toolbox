# Non-Recurrent-networks models

## Model list:
### MLP\_ACT\_X\_X\_...
To define and train a MLP the name of the model has to start by "MLP" and be followed by a underscore ("\_").
Then you have to define the activation function that you want to use among the following functions:

- "RELU" : defined as tf.nn.relu
- "TANH": defined as tf.nn.tanh
- "SIGMOID" : defined as tf.nn.sigmoid
- "LRELU" : defined as tf.nn.leaky\_relu


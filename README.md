# Linear Neural Networks

If github in unable to render a Jupyter notebook, copy the link of the notebook and enter into the nbviewer: https://nbviewer.jupyter.org/


These notebooks provide an introduction to Linear Neural Networks (LNNs). a LNN can learn only linear functions using a single layer of neuron(s). We create LNNs for solving binary and multi-class classification problems.

- Binary Clasification: the logistic regression model is casted as a LNN model with a single neuron

- Multi-class Classification: the softmax regression model is casted as a LNN model with multiple neurons. The number of neurons is equal to the number of classes.

The following gives a brief description of the notebooks.

- Notebook 1: Logistic regression as LNN for binary classification (linearly separable data)

- Notebook 2: Logistic regression as LNN for binary classification (linearly non-separable data)

- Notebook 3: Softmax regression as LNN for multi-class classification 




For creating the LNN, we will use the Keras API with TensorFlow 2.0 as the backend.


## Create Artificial Neural Networks (ANNs) using Keras

Keras is a high-level Deep Learning API that allows to easily build, train, evaluate, and execute all sorts of ANNs. TensorFlow 2 has adopted Keras as its official high-level API: tf.keras. It only supports TensorFlow as the backend, but it has the advantage of offering some very useful extra features.

We will use Keras to build the LNN. It involves two steps.
- Choose a suitable Keras model class & create the model using it
- Add layers of different types based on the need

### Step 1: Instantiate a Model using the Sequential API
The Sequential API is a straightforward and simple list of layers. It is limited to single-input, single-output stacks of layers.
https://keras.io/api/models/


### Step 2: Add Suitable Layers

Layers are the basic building blocks of neural networks in Keras: https://keras.io/api/layers/ 

For building a LNN, we only need to add **Dense layers**, which is a core layer is Keras: https://keras.io/api/layers/core_layers/dense/

Dense layer implements the operation: 

        output = activation(dot(input, kernel) + bias) 
      
where activation is the element-wise activation function passed as the activation argument, kernel is a weight matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True).

The tf.keras.layers.Dense class has the following parameters:

- units: Number of output neurons.

- activation (default: **"linear" activation**): Activation function to use.

- use_bias (default: **True**): Boolean, whether the layer uses a bias vector.

- kernel_initializer (default: **"glorot_uniform"**): Initializer for the kernel weights matrix.

- bias_initializer (default: **"zeros"**): Initializer for the bias vector.

- kernel_regularizer (default: None): Regularizer function applied to the kernel weights matrix.

- bias_regularizer (default: None): Regularizer function applied to the bias vector.

- activity_regularizer (default: None): Regularizer function applied to the output of the layer (its "activation").

- kernel_constraint (default: None): Constraint function applied to the kernel weights matrix.

- bias_constraint (default: None): Constraint function applied to the bias vector.


Note that by deault Glorot Uniform weight initialization technique is used. Generally the **He initialization** works well with the ReLU activation function (kernel_initializer="he_normal") in deep ANNs. 

### Note: Constructing the LNN Model

For constructing the LNN model, we will:
- Initialize the network weights with zero values
- Use the sigmoid activation function for binary classification and the softmax activation function for multi-class classification

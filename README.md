# Linear Neural Networks

If GitHub is unable to render a Jupyter notebook, copy the link of the notebook and enter it into the nbviewer: https://nbviewer.jupyter.org/


These notebooks provide an introduction to Linear Neural Networks (LNNs). An LNN can learn only linear functions using a single layer of neuron(s). We create LNNs for solving binary and multi-class classification problems.

- Binary Classification: the logistic regression is formulated as an LNN model with a single neuron

- Multi-class Classification: the softmax regression is formulated as an LNN model with multiple neurons. The number of neurons is equal to the number of classes.

The following gives a brief description of the notebooks.

- Notebook 1: Logistic regression as LNN for binary classification (linearly separable data)

- Notebook 2: Logistic regression as LNN for binary classification (linearly non-separable data)

- Notebook 3: Softmax regression as LNN for multi-class classification

        -- We use learning curves to monitor the model’s performance (accuracy and loss) on both the training and validation data over iterations (epochs).

- Notebook 4: Softmax regression as LNN for multi-class classification (MNIST dataset)


        -- We introduce two types of regularization in the LNN model: Weight regularization (L2 or L1), and Early stopping




To create the LNN, we utilize the **Keras API with TensorFlow 2.0 as the backend**.


## Create Artificial Neural Networks (ANNs) using Keras

Keras is a high-level deep learning API that allows us to easily build, train, evaluate, and execute various types of ANNs. TensorFlow 2 has adopted Keras as its official high-level API: tf.keras.

We will use Keras to build the LNN, which involves two steps:

- Choose a suitable Keras model class and create the model using it.
- Add layers of different types based on the requirements.


### Step 1: Instantiate a Model using the Sequential API

The Sequential API provides a straightforward way to stack layers in a linear fashion. It is limited to single-input and single-output configurations. For more details, visit the Keras Sequential API documentation: https://keras.io/api/models/


### Step 2: Add Suitable Layers

Layers are the fundamental building blocks of neural networks in Keras. For more information, refer to the Keras Layers documentation: https://keras.io/api/layers/ 

To build an LNN, we primarily need to add Dense layers, which are the core layers in Keras. You can find more about Dense layers in the Keras **Dense Layer** documentation: https://keras.io/api/layers/core_layers/dense/

The Dense layer implements the operation: 

        output = activation(dot(input, kernel) + bias)

where the activation function is applied element-wise, the kernel is a weight matrix created by the layer, and the bias is a bias vector created by the layer (only applicable if use_bias is set to True).



The tf.keras.layers.Dense class includes the following parameters:

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


Note that by default Glorot Uniform weight initialization technique is used. Generally the **He initialization** works well with the ReLU activation function (kernel_initializer="he_normal") in deep ANNs. 

        For the LNN model, we will initialize the network weights with zero values and use the sigmoid activation function for binary classification and the softmax activation function for multi-class classification.
### Note: Constructing the LNN Model

For constructing the LNN model, we will:
- Initialize the network weights with zero values
- Use the sigmoid activation function for binary classification and the softmax activation function for multi-class classification

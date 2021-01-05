# Only one activation function instantiation
# We use SoftMax activation function, only
import numpy as np
import nnfs
import Layers
import Activation
import Optimizers
import Loss

from nnfs.datasets import spiral_data

nnfs.init()

# Create dataset
X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 64 output values
dense1 = Layers.Layer_Dense(2, 64, weight_regularizer_l2=5e-4,
                     bias_regularizer_l2=5e-4)
dense2 = Layers.Layer_Dense(64, 64, weight_regularizer_l2=5e-4,
                     bias_regularizer_l2=5e-4)# A hidden layer
    
# Create ReLU activation (to be used with Dense layer):
#activation = Activation.Activation_Softmax()
activation = Activation.Activation_ReLU()
# Create second Dense layer with 64 input features (as we take output
# of previous layer here) and 3 output values (output values)
denseOut = Layers.Layer_Dense(64, 3)
    
# Create Softmax classifier's combined loss and activation
loss_activation = Activation.Activation_Softmax_Loss_CategoricalCrossentropy()
    
# Create optimizer
optimizer = Optimizers.Optimizer_Adam(learning_rate=0.02, decay=5e-7)
    
# Train in loop
for epoch in range(11001):
    
    
    # Forward pass
    dense1.forward(X)
    activation.forward(dense1.output)
    dense2.forward(activation.output) # A hidden layer - begin
    activation.forward(dense2.output) # A hidden layer - END
    denseOut.forward(activation.output)
    loss = loss_activation.forward(denseOut.output, y)
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'lr: {optimizer.current_learning_rate}')
    
    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    denseOut.backward(loss_activation.dinputs)
    activation.backward(denseOut.dinputs)
    dense2.backward(activation.dinputs)# A hidden layer
    activation.backward(dense2.dinputs)# A hidden layer
    dense1.backward(activation.dinputs)
    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(denseOut)
    optimizer.post_update_params()


# Validate the model

# Create test dataset
X_test, y_test = spiral_data(samples=100, classes=3)
dense1.forward(X_test)
activation.forward(dense1.output)
dense2.forward(activation.output)# A hidden layer
activation.forward(dense2.output)# A hidden layer
denseOut.forward(activation.output)
loss = loss_activation.forward(denseOut.output, y_test)
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == y_test)

print(f'validation: acc: {accuracy:.3f}, loss: {loss:.3f}')

import numpy as np
from layers import Dense, ReLU, Sigmoid, Softmax, binary_cross_entropy_gradient, binary_cross_entropy_loss, cross_entropy_gradient, cross_entropy_loss, mse_grad, mse_loss


network = [
    Dense(28*28,20),
    ReLU(),
    Dense(20,10),
    Softmax(),
]

def normalize(input,reference):
    return (input - reference.mean()) / (reference.std())
def denormalize(input,reference):
    return input * reference.std() + reference.mean()



from keras.datasets import mnist
from keras.utils import to_categorical


# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten the images

# One-hot encode the labels
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

normalized_x_data = (X_train / 255.0).reshape(-1,28*28,1)
normalized_y_data = y_train_one_hot.reshape(-1,10,1)


normalized_x_test = (X_test / 255.0).reshape(-1,28*28,1)
normalized_y_test = y_test_one_hot.reshape(-1,10,1)


LEARNING_RATE = 1e-2
EPOCHS = 30

for e in range(1,EPOCHS+1):
    EpochLoss = 0
    perm = np.random.permutation(len(normalized_x_data))
    # Shuffle both arrays using the same permutation index
    shuffled_normalized_x_data = normalized_x_data[perm]
    shuffled_normalized_y_data = normalized_y_data[perm]

    for x,y_true in zip(shuffled_normalized_x_data,shuffled_normalized_y_data):
        y_pred = x
        for layer in network:
            y_pred = layer.forward(y_pred)
        loss = cross_entropy_loss(y_pred,y_true)
        gradient = cross_entropy_gradient(y_pred,y_true)
        EpochLoss+=loss
        for layer in reversed(network):
            gradient = layer.backward(gradient,LEARNING_RATE)
            
    accuracy =0
    for x,y_true in zip(normalized_x_test,normalized_y_test):
        y_pred = x
        for layer in network:
            y_pred = layer.forward(y_pred)
        ind = np.argmax(y_pred)
        actual_ind = np.argmax(y_true)
        accuracy += 1 if actual_ind == ind else 0

    thisAccuracy = (accuracy*100)/len(normalized_x_test)
        
    print(f"Epoch : {e}/{EPOCHS} | Loss : {EpochLoss} | Accuracy : {thisAccuracy}%")



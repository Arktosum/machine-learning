import numpy as np

class Dense:
    def __init__(self,nInputs,nOutputs):
        self.weights = np.random.randn(nOutputs,nInputs)
        self.biases = np.random.randn(nOutputs,1)
    
    def forward(self,input):
        self.input = input        
        return self.weights @ self.input + self.biases

    def backward(self,gradient,learning_rate):
        dw = gradient @ self.input.T
        db = gradient
        self.weights -= dw * learning_rate
        self.biases -= db * learning_rate
        dx = self.weights.T @ gradient
        return dx

def mse_loss(y_pred,y_true) -> float:
    return (0.5*((y_pred-y_true)**2)).mean()

def mse_grad(y_pred,y_true):
    return y_pred - y_true

import numpy as np

class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, d_out,learning_rate):
        d_out[self.x <= 0] = 0
        return d_out

class Sigmoid:
    def forward(self, x):
        self.sig = 1 / (1 + np.exp(-x))
        return self.sig

    def backward(self, d_out,learning_rate):
        return d_out * self.sig * (1 - self.sig)
class Tanh:
    def forward(self, x):
        self.x = x
        return np.tanh(x)

    def backward(self, d_out, learning_rate):
        return d_out * (1 - np.tanh(self.x) ** 2) * learning_rate
    
class Softmax:
    def forward(self, x):
        max_x = np.max(x)
        exps = np.exp(x - max_x)
        self.softmax = exps / np.sum(exps)
        return self.softmax

    def backward(self, d_out, learning_rate):
        softmax_col = self.softmax.reshape(-1, 1)
        jacobian = np.diagflat(self.softmax) - np.dot(softmax_col, softmax_col.T)
        gradient = np.dot(jacobian.T, d_out)
        return gradient.reshape(-1, 1)
    
def cross_entropy_loss(y_pred, y_true):
    # Clip y_pred to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / len(y_true)
    return loss

def cross_entropy_gradient(y_pred, y_true):
    # Clip y_pred to avoid division by zero
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    grad = - (y_true / y_pred - (1 - y_true) / (1 - y_pred)) / len(y_true)
    return grad

def binary_cross_entropy_loss(y_pred, y_true):
    # Clip y_pred to avoid log(0) and log(1)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)

def binary_cross_entropy_gradient(y_pred, y_true):
    # Calculate gradient of binary cross-entropy loss
    grad = (y_pred - y_true) / (y_pred * (1 - y_pred) + 1e-15)
    return grad / len(y_true)
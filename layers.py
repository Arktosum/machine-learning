import math
from matrix import Matrix
import random
class Dense:
    def __init__(self,nInputs,nOutputs):
        self.weights = Matrix(nOutputs,nInputs,True)
        self.biases = Matrix(nOutputs,1,True)
    def forward(self,input : Matrix):
        self.inputs = input
        return self.weights @ self.inputs + self.biases
    def backward(self,gradient,learning_rate):
        dw = gradient @ self.inputs.transpose()
        db = gradient
        self.weights -= dw * learning_rate
        self.biases -= db * learning_rate
        return self.weights.transpose() @ gradient

class ReLU:
    def __init__(self):
        self.input = None
    def relu(self,x : float):
        return x if x > 0 else 0
    def relu_derivative(self,x : float):
        return 1 if x > 0 else 0
    def apply(self,input:Matrix):
        output = input.clone()
        output.apply(lambda x,i,j : self.relu(x))
        return output
    def apply_prime(self,input:Matrix):
        output = input.clone()
        output.apply(lambda x,i,j : self.relu_derivative(x))
        return output
    
    def forward(self,input:Matrix):
        self.input = input
        return self.apply(self.input)
    def backward(self,gradient:Matrix,learning_rate):
        return gradient * self.apply_prime(self.input)
        
    
class Sigmoid:
    def __init__(self):
        self.input = None
    def sigmoid(self,x : float):
        return 1 / (1 + math.exp(-x))
    def sigmoid_derivative(self,x : float):
        sigmoid = self.sigmoid(x)
        return sigmoid*(1-sigmoid)
    def apply(self,input:Matrix):
        output = input.clone()
        output.apply(lambda x,i,j : self.sigmoid(x))
        return output
    def apply_prime(self,input:Matrix):
        output = input.clone()
        output.apply(lambda x,i,j : self.sigmoid_derivative(x))
        return output
    
    def forward(self,input:Matrix):
        self.input = input
        return self.apply(self.input)
    def backward(self,gradient:Matrix,learning_rate):
        return gradient * self.apply_prime(self.input)
    
    
class Tanh:
    def __init__(self):
        self.input = None
    def tanh(self,x : float):
        pos_exp = math.exp(x)
        neg_exp = math.exp(-x)
        return (pos_exp - neg_exp)/(pos_exp + neg_exp)
    def tanh_derivative(self,x : float):
        tanh = self.tanh(x)
        return 1 - tanh**2
    def apply(self,input:Matrix):
        output = input.clone()
        output.apply(lambda x,i,j : self.tanh(x))
        return output
    def apply_prime(self,input:Matrix):
        output = input.clone()
        output.apply(lambda x,i,j : self.tanh_derivative(x))
        return output
    
    def forward(self,input:Matrix):
        self.input = input
        return self.apply(self.input)
    def backward(self,gradient:Matrix,learning_rate):
        return gradient * self.apply_prime(self.input)
    
    
    

def mse_loss(y_pred : Matrix,y_true : Matrix):
    return (0.5*((y_pred - y_true)**2)).mean()

def mse_gradient(y_pred : Matrix,y_true: Matrix):
    return (y_pred - y_true)

def binary_cross_entropy_loss(y_pred : Matrix,y_true : Matrix) -> float:
    epsilon = 1e-9 # a very small number to overcome math domain error for log x when x = 0
    logP = y_pred.clone()
    logP.apply(lambda x,i,j : math.log(x+epsilon))
    loginvP = y_pred.clone()
    loginvP.apply(lambda x,i,j : math.log(1-x + epsilon))
    return (-(y_true*logP + (1-y_true)*loginvP)).mean()

def binary_cross_entropy_gradient(y_pred : Matrix,y_true : Matrix) -> Matrix:
    epsilon = 1e-9
    a = y_true/(y_pred+epsilon)
    b = (1-y_true) / (1-y_pred + epsilon)
    return -(a-b)

class Softmax:
    def __init__(self):
        pass
    def softmax(self,input : Matrix) -> Matrix:
        new = input.clone()
        max_element = new.max()
        # subtract with maximum value to prevent overflow math range error. does not affect the result.
        new.apply(lambda x,i,j : math.exp(x-max_element)) 
        sumexp = new.sum()
        soft = new.clone()
        soft.apply(lambda x,i,j : x / sumexp)
        soft_sum = soft.sum()
        if soft_sum <= 0.9:
            raise ValueError("Sum of softmax is not adding up to 1! : ",soft_sum)
        else:
            pass
        return soft
    def softmax_derivative(self):
        # self.output is outputs from softmax. (n,1)
        n = self.output.shape[0]
        tiled = self.output.tile()
        return (tiled*(Matrix.identity(n) - tiled.transpose()))
    def forward(self,input : Matrix) -> Matrix:
        self.output = self.softmax(input)
        return self.output
    def backward(self,output_gradient : Matrix, learning_rate : float) -> Matrix:
        return self.softmax_derivative() @ output_gradient
        
        
        
def cross_entropy_loss(y_pred : Matrix,y_true:Matrix) -> float:
    # y_pred is vector predictions from softmax (n , 1)
    # y_true is vector one hot encoding in binary (0,1)  ( n,1 )
    log = y_pred.clone()
    epsilon = 1e-10
    log.apply(lambda x,i,j : math.log(x + epsilon))
    prod = y_true * log
    return -prod.sum()

def cross_entropy_gradient(y_pred : Matrix , y_true : Matrix) -> Matrix:
    # y_pred is vector predictions from softmax (n , 1)
    # y_true is vector one hot encoding in binary (0,1)  ( n,1 )
    epsilon = 1e-10
    pred = y_pred.clone() + epsilon
    return -(y_true / pred)
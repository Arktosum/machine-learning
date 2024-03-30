import math
from layers import Dense, ReLU, Sigmoid, Softmax, Tanh, binary_cross_entropy_gradient, binary_cross_entropy_loss, cross_entropy_gradient, cross_entropy_loss
from matrix import Matrix
from network import NeuralNetwork
from utils import AB_split, StandardNormalizer, shuffle_two_arrays
import numpy as np
import matplotlib.pyplot as plt

# import tensorflow as tf

# mnist = tf.keras.datasets.mnist
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# print("Train images shape:", train_images.shape)
# print("Train labels shape:", train_labels.shape)
# print("Test images shape:", test_images.shape)
# print("Test labels shape:", test_labels.shape)

nn = NeuralNetwork()
nn.network = [
    Dense(1,10),
    ReLU(),
    Dense(10,4),
    Softmax()
]
nn.loss_function = cross_entropy_loss
nn.loss_function_gradient = cross_entropy_gradient

x_data = []
y_data = []

for i in range(-2000,2001):
    # 3 classes. 
    #  [- 200,-1]  [1,0,0]
    #  [0 , 100]   [0,1,0]
    #  [101, 200]  [0,0,1]
    if -2000 <= i <= -1001:
        x_data.append(Matrix.fromVector([i]).reshape(-1,1))
        y_data.append(Matrix.fromVector([1,0,0,0]).reshape(-1,1))
    elif -1000 <= i <= -1:
        x_data.append(Matrix.fromVector([i]).reshape(-1,1))
        y_data.append(Matrix.fromVector([0,1,0,0]).reshape(-1,1))
    elif 0 <= i <= 1000:
        x_data.append(Matrix.fromVector([i]).reshape(-1,1))
        y_data.append(Matrix.fromVector([0,0,1,0]).reshape(-1,1))
    elif 1001 <= i <= 2000:
        x_data.append(Matrix.fromVector([i]).reshape(-1,1))
        y_data.append(Matrix.fromVector([0,0,0,1]).reshape(-1,1))

# NORMALIZING WHOLE DATASET
sn = StandardNormalizer(x_data)
norm_x = sn.apply(x_data)
# sno = StandardNormalizer(y_data)
# norm_y = sno.apply(y_data)
norm_y = y_data
# # SPLITTING INTO DATASETS
norm_x_train,norm_y_train,norm_x_test,norm_y_test = AB_split(norm_x,norm_y,0.2)

lossHistory =  nn.train(norm_x_train,norm_y_train,250,1e-3,1)
norm_predictions =  nn.test(norm_x_test,norm_y_test)

# denorm_y_test = sno.remove(norm_y_test)
# denorm_predictions = sno.remove(norm_predictions)
denorm_x_test = sn.remove(norm_x_test)
denorm_y_test = norm_y_test
denorm_predictions = norm_predictions

accuracy = 0
for x,y,y_pred in zip(denorm_x_test,denorm_y_test,denorm_predictions):
    y_index = y.transpose()[0].index(max(y.transpose()[0]))
    y_pred_index = y_pred.transpose()[0].index(max(y_pred.transpose()[0]))
    accuracy +=  1 if y_index == y_pred_index else 0
    print(y_index,y_pred_index)

print(f"Accuracy : {(accuracy/len(denorm_x_test))*100}%",)

# for x,y in zip(train_images, train_labels):
#     flat_x = (x/255.0).reshape(-1,1)
#     one_hot_y = np.zeros((10,1))
#     one_hot_y[y] = 1.0
    
#     flat_x_vector = Matrix.fromVector(flat_x).reshape(-1,1)
#     ony_hot_y_vector = Matrix.fromVector(one_hot_y).reshape(-1,1)
#     x_train.append(flat_x_vector)
#     y_train.append(ony_hot_y_vector)

# plt.grid(True)
# plt.plot([i for i in range(len(lossHistory))],lossHistory)
# plt.show()

# plt.grid(True)
# plt.scatter([x[0][0] for x in denorm_x_test],[y[0][0] for y in denorm_y_test])
# plt.scatter([x[0][0] for x in denorm_x_test],[y[0][0] for y in denorm_predictions])
# plt.show()

# plt.grid(True)
# plt.scatter([y[0][0] for y in denorm_y_test],[y[0][0] for y in denorm_predictions])
# plt.show()
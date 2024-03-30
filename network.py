from matrix import Matrix

from utils import shuffle_two_arrays, split_into_batches

class NeuralNetwork:
    def __init__(self):
        self.network = []
        self.loss_function = None
        self.loss_function_gradient = None
    def forward(self,input):
        # input : (m,1)
        output = input
        for layer in self.network:
            output = layer.forward(output)
        return output
    def backward(self,gradient,learning_rate):
        for layer in reversed(self.network):
            gradient = layer.backward(gradient,learning_rate)
    
    def predictOne(self,input:Matrix) -> Matrix:
        '''
        Returns one output for one vector of input (m,1)
        '''
        return self.forward(input)
    
    def test(self,x_test,y_test):
        '''
            S = number of Samples.
            x_test = (S,m,1) , m is number of features
            y_test = (S,n,1) , n is number of targets
        '''
        batch_loss = []
        outputs = []
        for x,y_true in zip(x_test,y_test):
            y_pred = self.predictOne(x)
            loss = self.loss_function(y_pred,y_true)
            batch_loss.append(loss)
            outputs.append(y_pred)
        total_loss = sum(batch_loss) / len(x_test)
        print(f"Test loss : {total_loss}")
        return outputs
    
    def train(self,x_train,y_train,epochs=1,learning_rate=1e-3,batch_size=1):
        # S = number of Samples.
        # x_train = (S,m,1) , m is number of features
        # y_train = (S,n,1) , n is number of targets
        S = len(x_train)
        m = x_train[0].shape[0]
        n = y_train[0].shape[0]
        batch_count = S//batch_size
        print(f"Total Samples : {S} | Batch Count : {batch_count}")
        lossHistory = []
        for e in range(1,epochs+1):
            epochLoss = 0           
            x_shuffled,y_shuffled = shuffle_two_arrays(x_train,y_train)
            x_shuffled,y_shuffled = x_train,y_train
            x_batches,y_batches = split_into_batches(x_shuffled,y_shuffled,batch_size)
            b=1
            for x_batch,y_batch in zip(x_batches,y_batches):
                batch_loss = []
                batch_gradient = Matrix(n,1)
                for x,y_true in zip(x_batch,y_batch): 
                    y_pred = self.forward(x)
                    loss = self.loss_function(y_pred,y_true)
                    loss_gradient = self.loss_function_gradient(y_pred,y_true)
                    batch_loss.append(loss)
                    batch_gradient += loss_gradient
                batch_loss = sum(batch_loss)/len(x_batch)
                epochLoss+=batch_loss
                self.backward(batch_gradient,learning_rate)
                b+=1
                # print(f"Batch loss : {b}/{len(x_batches)} | {batch_loss}")
            epochLoss = epochLoss/len(x_batches)
            lossHistory.append(epochLoss)
            print(f"Epoch : {e} | Loss {epochLoss}")
        return lossHistory


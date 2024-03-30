from matrix import Matrix
import random
class StandardNormalizer:
    def __init__(self,input : list[Matrix]):
        # array[vector(m,1)]
        m = input[0].shape[0]
        meanVector = Matrix(m,1)
        for x in input:
            meanVector += x
        meanVector = meanVector / len(input)
        SD2 = Matrix(m,1)
        for x in input:
            SD2 += (x - meanVector)**2
        SD2 = SD2 / len(input)
        SD = SD2**0.5
        self.meanVector = meanVector
        self.SDVector = SD
    def apply(self,input : list[Matrix]) -> list[Matrix]:
        '''
            X_new = (X - mu )/sd
        '''
        output = []
        for x in input:
            res = (x - self.meanVector)/self.SDVector
            output.append(res)
        return output
    def remove(self,input : list[Matrix]) -> list[Matrix]:
        '''
            X_new = X*sd + mu
        '''
        output = []
        for x in input:
            res = x*self.SDVector + self.meanVector
            output.append(res)
        return output
        
        
def split_into_batches(array1, array2, batch_size):
    batches1 = [array1[i:i+batch_size] for i in range(0, len(array1), batch_size)]
    batches2 = [array2[i:i+batch_size] for i in range(0, len(array2), batch_size)]
    return batches1, batches2

def AB_split(X,Y,split=0.2):
    '''
    Shuffles and splits the set into training and testing.
    assumption, X and Y are just list[any] and length are same.
    '''
    newX,newY = shuffle_two_arrays(X,Y)
    TOTAL_LENGTH = len(newX)
    B_LENGTH = int(TOTAL_LENGTH*split) # floors result.
    A_LENGTH = TOTAL_LENGTH - B_LENGTH
    AX = newX[:A_LENGTH]
    BX = newX[A_LENGTH:]
    AY = newY[:A_LENGTH]
    BY = newY[A_LENGTH:]
    return AX,AY,BX,BY

def shuffle_two_arrays(array1, array2):
    array1_shuffled = array1.copy()
    array2_shuffled = array2.copy()
    combined = list(zip(array1_shuffled, array2_shuffled))
    random.shuffle(combined)
    array1_shuffled, array2_shuffled = zip(*combined)
    return list(array1_shuffled), list(array2_shuffled)
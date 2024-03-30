import math
import random 
class Matrix:
    def __init__(self,nRows,nCols,isRandom=False):
        '''
        Random uses standard normal distribution with mean = 0 and SD = 1
        '''
        self.matrix = []
        self.shape = (nRows,nCols)
        self.nRows = nRows
        self.nCols = nCols
        for i in range(nRows):
            row = []
            for j in range(nCols):
                row.append(random.gauss(0,1) if isRandom else 0)
            self.matrix.append(row)
    
    def matMulDimensions(self,other):
        if self.nCols == other.nRows:
            return True
        raise ValueError(f"Invalid matrix dimensions for matrix multiplication! ({self.shape}) and ({other.shape}) !")
    def __matmul__(self,other):
        assert self.matMulDimensions(other)
        newMatrix = Matrix(self.nRows,other.nCols)
        
        for i in range(self.nRows):
            for j in range(other.nCols):
                SUM = 0
                for k in range(other.nRows):
                    SUM += self[i][k] * other[k][j]
                newMatrix[i][j] = SUM
                
        return newMatrix
                    
        
    def apply(self,func):
        for i in range(self.nRows):
            for j in range(self.nCols):
                self[i][j] = func(self[i][j],i,j)
                
    def __getitem__(self, index):
        return self.matrix[index]
    def __setitem__(self, index, value):
        self.matrix[index] = value
    def clone(self):
        newMatrix = Matrix(self.nRows,self.nCols)
        for i in range(self.nRows):
            for j in range(self.nCols):
                newMatrix[i][j] = self[i][j]
        return newMatrix
    def __repr__(self):
        # stringify = f"Shape : {self.shape}\n"
        stringify = f""
        for i in range(self.nRows):
            stringify += str(self[i]) +'\n'
        return stringify
    
    def __add__(self, other):
        newMatrix = self.clone()
        if(isinstance(other,Matrix)):
            newMatrix.apply(lambda x,i,j : self[i][j] + other[i][j])
            return newMatrix
        elif isinstance(other,int) or isinstance(other,float):
            other = float(other) # in case it's an integer
            newMatrix.apply(lambda x,i,j : self[i][j] + other)
            return newMatrix
        else:
            raise ValueError("Invalid input type")
    def sum(self):
        SUM = 0.0
        for row in self.matrix:
            for ele in row:
                SUM+=ele
        return SUM
    def mean(self):
        SUM = self.sum()
        return SUM / (self.nRows * self.nCols)
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        newMatrix = self.clone()
        if(isinstance(other,Matrix)):
            newMatrix.apply(lambda x,i,j : self[i][j] * other[i][j])
            return newMatrix
        elif isinstance(other,int) or isinstance(other,float):
            other = float(other) # in case it's an integer
            newMatrix.apply(lambda x,i,j : self[i][j] * other)
            return newMatrix
        else:
            raise ValueError("Invalid input type")
    
    def __pow__(self, other):
        newMatrix = self.clone()
        if isinstance(other,int) or isinstance(other,float):
            other = float(other) # in case it's an integer
            newMatrix.apply(lambda x,i,j : self[i][j] ** other)
            return newMatrix
        else:
            raise ValueError("Invalid input type")
    def exp(self):
        newMatrix = self.clone()
        newMatrix.apply(lambda x,i,j : math.exp(self[i][j]) )
        return newMatrix
    def reshape(self,newRow,newCol):
        # or*oc = nr*nc
        if newRow == -1 and newCol == -1:
            raise ValueError("Both rows and columns cannot be -1")
        if newRow == -1:
            newRow = self.nRows*self.nCols // newCol
        if newCol == -1:
            newCol = self.nRows*self.nCols // newRow
        
        newMatrix = Matrix(newRow,newCol)
        # flatten
        arr = []
        for row in self.matrix:
            arr.extend(row)
        
        newMatrix.apply(lambda x,i,j : arr[newCol*i+j])
        return newMatrix

    @staticmethod
    def fromVector(array):
        newMatrix = Matrix(1,len(array))   
        newMatrix.matrix = [array[:]]
        return newMatrix
            
            
            
    def transpose(self):
        newMatrix = Matrix(self.nCols,self.nRows)
        newMatrix.apply(lambda x,i,j : self[j][i])
        return newMatrix
    def __neg__(self):
        # s = -s
        return self * -1
    def __sub__(self, other):
        # s = s - o = s + -o
        return self + -other
    def __rsub__(self, other):
        # s = o - s = -s + o
        return -self + other
    def __rmul__(self, other):
        # s = o * s
        return self * other # commutative?
    def __truediv__(self, other):
        # s = s / o = s * o^-1
        return self * (other ** -1)
    def max(self) -> float:
        max_ = -float('inf')
        for row in self.matrix:
            for ele in row:
                max_ = max(max_,ele)
        return max_
    def tile(self):
        if self.nCols != 1:
            raise ValueError("Only works with column vectors!")

        n = self.nRows
        # tiles a vector (n,1) by  n times creating a (n,n) matrix.
        # used in softmax derivative so far.
        newMatrix = Matrix(n,n)
        newMatrix.apply(lambda x,i,j : self.matrix[i][0])
        
        return newMatrix
    
    @staticmethod
    def identity(n : int):
        '''
        Returns a nxn identity matrix
        '''
        newMatrix = Matrix(n,n)
        newMatrix.apply(lambda x,i,j : 1.0 if(i==j) else 0.0)
        return newMatrix
        
        
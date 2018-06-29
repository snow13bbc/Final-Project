import numpy as np
import random
import progressbar

#I learned the basic idea of neural net and adapted and edited the skeleton
#for the code from https://github.com/stephencwelch/Neural-Networks-Demystified.


class neuralnet(object):

    def __init__(self,inputLayerSize,hiddenLayerSize,outputLayerSize):
        #Adjust the hyperparameter as needed
        self.inputLayerSize = inputLayerSize
        self.hiddenLayerSize = hiddenLayerSize
        self.outputLayerSize = outputLayerSize

        #For weights parameters
        #You need weigh matrix between input and hidden layer, and also between
        #hidden and output layer. For now, they are just random values now.
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        self.bias1 = np.random.randn(1,self.hiddenLayerSize)
        self.bias2 = np.random.randn(1,self.outputLayerSize)

    def forward(self,X):
    	#This is a feedforward function, where you estimate the output
    	#using the weigh matrix and activation function in each layer.

        self.z2 = np.dot(X,self.W1) + self.bias1
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.W2) + self.bias2
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
    	#This is one type of activation function that we are using.
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self,z):
    	#This is derivative of sigmoid function. This is needed to calculate
    	#gradient descent.
        return np.exp(-z)/((1+np.exp(-z))**2)

    def costFunction(self, X, y, lam):
        '''
        J tells you what the difference is between the predicted and actual result.
        Compute cost for given X,y, use weights already stored in class.
        To avoid overfitting, I add a regularization paramter, lam to our cost function.
        The higher the lam value, the bigger the penalties impose.

        '''
        m = len(X[0])
        self.yHat = self.forward(X)

        J = (1/m) * np.sum((y - self.yHat)**2) + ((lam/2) * (np.sum(self.W2**2) + np.sum(self.W1**2)))

        return J

    def costFunctionPrime(self, X, y):
        #To get a batch gradient descent, we need to do more derivatives.
        self.yHat = self.forward(X)
        #We need to add lam to make sure we are not overfitting
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)
        dJdb1 = delta2
        dJdb2 = delta3

        return dJdW1, dJdW2, dJdb1, dJdb2

    def train(self,X,y,iterations,alpha,lam):
        self.X = X
        self.y = y
        m = X.shape[0]
        bar = progressbar.ProgressBar(maxval=iterations,widgets=[progressbar.Bar('=','[',']'),' ', progressbar.Percentage()])
        bar.start()
        count=0
        for i in range(0,iterations):
            count +=1
            bar.update(i+1)
            #Reset parameters at each iteration.
            #And do more PDE until you can't do anymore.
            dW1 = np.zeros((self.inputLayerSize,self.hiddenLayerSize))
            dW2 = np.zeros((self.hiddenLayerSize,self.outputLayerSize))
            db1 = np.zeros((1,self.hiddenLayerSize))
            db2 = np.zeros((1,self.outputLayerSize))
            dJdW1,dJdW2,dJdb1,dJdb2 = self.costFunctionPrime(X,y)
            dW1 += dJdW1
            dW2 += dJdW2
            for j in range(0,(m-1)):
                db1 += dJdb1[j]
                db2 += dJdb2[j]
            #Altering the parameters
            self.W1 = self.W1 - alpha*(((1/m)*dW1)+lam*self.W1)
            self.W2 = self.W2 - alpha*(((1/m)*dW2)+lam*self.W2)
            self.bias1 = self.bias1 - alpha*((1/m)*db1)
            self.bias2 = self.bias2 - alpha*((1/m)*db2)
        bar.finish()
        print('Score:')
        print(self.costFunction(X,y,lam))
        return self.forward(X)

    #The following functions test for a single example
    def forward_stochastic(self,X,k):
    	#This is a feedforward function, where you estimate the output
    	#using the weigh matrix and activation function in each layer.
        self.z2 = np.dot(X[k],self.W1) + self.bias1
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.W2) + self.bias2
        yHat = self.sigmoid(self.z3)
        return yHat

    def costFunction_stochastic(self,X,y,k,lam):
        m = len(X[0])
        self.yHat_s = self.forward_stochastic(X,k)
        J = np.sum((y[k] - self.yHat_s)**2) + ((lam/2) * (np.sum(self.W2**2) + np.sum(self.W1**2)))
        return J

    def costFunctionPrime_stochastic(self, X, y,k):
        #To get a batch gradient descent, we need to do more derivatives.
        self.yHat = self.forward_stochastic(X,k)
        #We need to add lam to make sure we are not overfitting
        delta3 = np.multiply(-(y[k]-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3,self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X[k].T.reshape(self.inputLayerSize,1),delta2)
        dJdb1 = delta2
        dJdb2 = delta3

        return dJdW1, dJdW2, dJdb1, dJdb2

    def train_stochastic(self,X,y,iterations,alpha,lam,pos_len,neg_len):
        self.X = X
        self.y = y
        m = X.shape[0]
        pos, neg = 0,0
        bar = progressbar.ProgressBar(maxval=iterations,widgets=[progressbar.Bar('=','[',']'),' ', progressbar.Percentage()])
        bar.start()
        for i in range(0,iterations):
            bar.update(i+1)
            #Generate a random number to see the index which dataset to use.
            #Get the index of pos_vec if it's > 0.3.
            #If less, use that of negative.
        rand = random.uniform(0,1)
        if rand > 0.3:
            rand2 = random.randint(0,pos_len-1)
        else:
            rand2 = random.randint(pos_len,neg_len)

        new_costfunc = self.costFunction_stochastic(X,y,rand2,lam)
        new_alpha = (2*alpha * new_costfunc) + alpha #Use cost function value to alter alpha.
        # Set update matrices to zero each iteration
        dW1 = np.zeros((self.inputLayerSize,self.hiddenLayerSize))
        dW2 = np.zeros((self.hiddenLayerSize,self.outputLayerSize))
        db1 = np.zeros((1,self.hiddenLayerSize))
        db2 = np.zeros((1,self.outputLayerSize))

        # Find partial derivatives
        dJdW1,dJdW2,dJdb1,dJdb2 = self.costFunctionPrime_stochastic(X,y,rand2)
        # Update "change" matrices
        dW1 += dJdW1
        dW2 += dJdW2
        db1 += dJdb1
        db2 += dJdb2

        # And finally, update the weight and bias matrices.
        self.W1 = self.W1 - new_alpha*(((1/m) * dW1) + lam * self.W1)
        self.W2 = self.W2 - new_alpha*(((1/m) * dW2) + lam * self.W2)
        self.bias1 = self.bias1 - new_alpha*((1/m) * db1)
        self.bias2 = self.bias2 - new_alpha*((1/m) * db2)
        bar.finish()
        print('Score:')
        return self.forward(X)

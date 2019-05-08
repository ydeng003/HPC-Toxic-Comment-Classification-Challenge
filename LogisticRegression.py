import numpy as np


class LogisticRegression:
    
    
    def __init__(self, trainX, trainY, learningRate = 0.05, numSteps = 50000, multiTarget = 1):
        self.trainX = np.c_[np.ones(trainX.shape[0]), trainX]
        self.trainY = trainY
        self.learningRate = learningRate
        self.numSteps = numSteps
        self.multiTarget = int(self.power(2, multiTarget))
    #fit for first two strategies            
    def fit(self):
        self.betas = np.zeros(self.trainX.shape[1])
        for step in range(self.numSteps):
            bx = np.dot(self.trainX, self.betas)
            hx = np.exp(bx) / (1 + np.exp(bx))
            err = hx - self.trainY
            grad = np.dot(self.trainX.T, err)
            self.betas -= self.learningRate * grad
        return self.betas
    
    #fit for strategy 3
    def fitMulti(self):
        self.betasList = []
        new_targets = []
        for i in range(1, self.multiTarget + 1):
            new_target = [0.0]*(self.trainX.shape[0])
            for j in range(len(self.trainY)):
                if self.trainY[j] == i:
                    new_target[j] = 1
            new_targets.append(new_target) 
        for i in range(len(new_targets)):
            beta = [0.0] * (self.trainX.shape[1])
            for step in range(self.numSteps):
                bx = np.dot(self.trainX, beta)
                hx = np.exp(bx) / (1 + np.exp(bx))
                err = hx - new_targets[i]
                grad = np.dot(self.trainX.T, err)
                beta -= self.learningRate * grad
            self.betasList.append(beta)
        return self.betasList

    #predict for first two strategies
    def predict(self, tstX):
        testX = np.c_[np.ones(tstX.shape[0]), tstX]
        self.pred = np.zeros(tstX.shape[0])
        bx = np.dot(testX, self.betas)
        hx = list(np.exp(bx) / (1 + np.exp(bx)))
        for i in range(len(hx)):
            if(hx[i] >= 0.5):
               self.pred[i] =  1
        return self.pred
    #predict for strategy 3   
    def predictMulti(self, tstX):
        testX = np.c_[np.ones(tstX.shape[0]), tstX]
        self.predList = []
        for i in range(self.multiTarget):
            bx = np.dot(testX, self.betasList[i])
            pred = list(np.exp(bx) / (1 + np.exp(bx)))
            self.predList.append(pred)
        self.pred = [0.0]*(tstX.shape[0])
        for i in range(tstX.shape[0]):
            predMax = 0
            predInd = 0
            for j in range(len(self.predList)):
                if self.predList[j][i] > predMax:
                     predMax = self.predList[j][i]
                     predInd = j + 1
            self.pred[i] = predInd      
        return self.pred
    
    @staticmethod
    def power(m, n):
        a = m
        if n == 0:
            return 1
        else:
            for i in range(n-1):
                m = m * a
        return m
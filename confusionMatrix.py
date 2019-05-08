import numpy as np


class confusionMatrix:
        
    def __init__(self, pred, actual):
        self.pred = np.array(pred)
        self.actual = np.array(actual)
    
    def tfpn(self):
        diff = list(self.pred * 2 - self.actual)
        self.fn = diff.count(-1)
        self.fp = diff.count(2)
        self.tp = diff.count(1)
        self.tn = diff.count(0)        
            

    def confMatrix(self):
        self.tfpn()
        print("n = %d \t\t Actural Positive \t Actural Negative" % len(self.pred))
        print("Predicted True \t\t %d \t %d" % (self.tp, self.fp))
        print("Predicted False \t %d \t %d" % (self.fn, self.tn))

    
    def getPrecision(self):
        self.tfpn()
        if self.tp + self.fp ==0:
            return 1
        else: 
            self.precision = self.tp / float(self.tp + self.fp)
        return self.precision
        
    
    def getRecall(self):
        self.tfpn()
        if self.tp + self.fn == 0:
            return 1
        else: 
            self.recall = self.tp / float(self.tp + self.fn)
        return self.recall
    
    def getAccuracy(self):
        self.tfpn()
        self.accuracy = (self.tp + self.tn) / float(self.tp + self.tn + self.fp + self.fn)
        return self.accuracy

        

    
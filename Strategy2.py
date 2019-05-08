from pyspark import SparkContext, SparkConf
conf = (SparkConf().set("spark.driver.maxResultSize", "4g"))
sc = SparkContext(conf=conf,appName="Strategy2")
from LogisticRegression import LogisticRegression
from confusionMatrix import confusionMatrix
import numpy as np
from numpy import array
import sys
import time

start_time = time.time()
if len(sys.argv) != 4:
    print >> sys.stderr, "Usage: Strategy2.py <train file> <test file> <output directory>"
    exit(-1)
accuracy = []
precision = []
recall = []
targets_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train_cmts = sc.textFile(sys.argv[1])
train_cmts = train_cmts.map(lambda line: "".join(c for c in line if c not in ['[',']']))
train_cmts = train_cmts.flatMap(lambda line: [line.strip().split(', ')])
train_cmts = train_cmts.map(lambda comment: [ float(n) for n in comment ])
train_cmts = train_cmts.collect()
train_cmts = array(train_cmts)
length1 = len(train_cmts[0])

test_cmts = sc.textFile(sys.argv[2])
test_cmts = test_cmts.map(lambda line: "".join(c for c in line if c not in ['[',']']))
test_cmts = test_cmts.flatMap(lambda line: [line.strip().split(', ')])
test_cmts = test_cmts.map(lambda comment: [ float(n) for n in comment ])
test_cmts = test_cmts.collect()
test_cmts = array(test_cmts)
length2 = len(test_cmts[0])

trainX = train_cmts[:,[range(6,length1)]]
trainX = trainX.reshape(trainX.shape[0],trainX.shape[2])
trainY = train_cmts[:,[0]]
trainY = trainY.reshape(trainY.shape[0])


testX = test_cmts[:,[range(6,length2)]]
testX = testX.reshape(testX.shape[0],testX.shape[2])
testY = test_cmts[:,[0]]
testY = testY.reshape(testY.shape[0])

for i in range(6):
    clf = LogisticRegression(trainX, trainY)
    clf.fit()
    pred = clf.predict(testX)
    confM = confusionMatrix(pred, testY)
    print("Predict %s:" % targets_names[i])
    confM.confMatrix()
    accu = confM.getAccuracy()
    print("Accuracy:", accu)    
    accuracy.append(accu) 
    prec = confM.getPrecision()
    print("Precision:", prec)
    precision.append(prec) 
    rec = confM.getRecall()
    print("Recall:", rec)
    recall.append(rec)   
    sc.parallelize(pred,1).saveAsTextFile(sys.argv[3]+"/%s/pred/" % targets_names[i])
    sc.parallelize(testY,1).saveAsTextFile(sys.argv[3]+"/%s/target/"  % targets_names[i])
    if i < 5:
        x = train_cmts[:,[i]]
        trainX = np.concatenate((trainX, x), axis=1)
        trainY = train_cmts[:,[i + 1]]
        trainY = trainY.reshape(trainY.shape[0])
        y = array(pred)
        y = y.reshape(y.shape[0],1)
        testX = np.concatenate((testX, y), axis=1)
        testY = test_cmts[:,[i + 1]]
        testY = testY.reshape(testY.shape[0])
            
def listMean(List):
    Sum = 0
    for i in range(len(List)):
        Sum += List[i]
    return Sum/len(List)

print('Average accuracy is: ', listMean(accuracy))
print('Average precision is: ', listMean(precision))
print('Average recall is: ', listMean(recall))
print("--- %s seconds ---" % (time.time() - start_time))
sc.stop()
    
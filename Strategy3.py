from pyspark import SparkContext, SparkConf
conf = (SparkConf().set("spark.driver.maxResultSize", "4g"))
sc = SparkContext(conf=conf,appName="Strategy3")
from LogisticRegression import LogisticRegression
from confusionMatrix import confusionMatrix
from numpy import array
import sys
import time

start_time = time.time()
if len(sys.argv) != 4:
    print >> sys.stderr, "Usage: Strategy3.py <train file> <test file> <output directory>"
    exit(-1)
accuracy = []
precision = []
recall = []
targets_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def power(m, n):
    a = m
    if n == 0:
        return 1
    else:
        for i in range(n-1):
            m = m * a
    return m

def resetTarget(comment):
    targets = comment[:6]
    new_target = 1
    for i in range(len(targets)):
        if targets[i] == 1:
            new_target += power(2, 5-i)    
    comment.insert(0, new_target)
    return comment

def resetPred(preds):
    predList = []
    for i in range(len(preds)):
        preds[i] = preds[i] - 1
        pred = []
        for j in range(6):                       
            a = preds[i] % 2
            pred.insert(0,int(a))
            preds[i] = int(preds[i]/2)
        predList.append(pred)
    return predList

        
        
train_cmts = sc.textFile(sys.argv[1])
train_cmts = train_cmts.map(lambda line: "".join(c for c in line if c not in ['[',']']))
train_cmts = train_cmts.flatMap(lambda line: [line.strip().split(', ')])
train_cmts = train_cmts.map(lambda comment: [ float(n) for n in comment ])
train_cmts = train_cmts.map(resetTarget)
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

trainX = train_cmts[:,[range(7,length1)]]
trainX = trainX.reshape(trainX.shape[0],trainX.shape[2])
trainY = train_cmts[:,[0]]
trainY = trainY.reshape(trainY.shape[0])

testX = test_cmts[:,[range(6,length2)]]
testX = testX.reshape(testX.shape[0],testX.shape[2])
testY = test_cmts[:,[range(6)]]
testY = testY.reshape(testY.shape[0],testY.shape[2])

clf = LogisticRegression(trainX, trainY,numSteps = 50000, multiTarget = 6)
clf.fitMulti()
predY = clf.predictMulti(testX)
predList = resetPred(predY)
predList = array(predList)

for i in range(6):
    p = predList[:,[i]]
    a = testY[:,[i]]
    confM = confusionMatrix(p, a)
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
    sc.parallelize(p,1).saveAsTextFile(sys.argv[3]+"/%s/pred/" % targets_names[i])
    sc.parallelize(a,1).saveAsTextFile(sys.argv[3]+"/%s/target/" % targets_names[i])  
    
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
    
from pyspark import SparkContext
sc = SparkContext(appName="tfidf")
import sys
import math
import time

start_time = time.time()
if len(sys.argv) != 6:
    print >> sys.stderr, "Usage: tfidf.py <train file> <test file> <train output directory> <test output directory> <threshold>"
    exit(-1)

THRESHOLD = int(sys.argv[5])

# Transform each line of RDD to [[list of targets], [whole comment]]
def getValueTarget(line):
    cmt = line[1:-6]
    targets = line[-6:]
    for i in range(len(targets)):
        targets[i] = int(targets[i])
    comment = [targets]
    comment.append(cmt)      
    return comment
 
# Tokenize the each comment. Filter out all stop words and special characters.
def token(comment):
    stop_words = ['a','about','above','after','again','against','all','am','an','and','any','are',"aren\'t",'as','at','be','because',
 'been','before','being','below','between','both','but','by',"can\'t",'cannot','could',"couldn\'t",'did',"didn\'t",'do','does',"doesn\'t",'doing',"don\'t",
 'down','during','each','few','for','from','further','had',"hadn\'t",'has',"hasn\'t",'have',"haven\'t",'having','he',"he\'d","he\'ll","he\'s",'her','here',
 "here\'s",'hers','herself','him','himself','his','how',"how\'s",'i',"i\'d","i\'ll","i\'m","i\'ve",'if','in','into','is',"isn\'t",'it',"it\'s",'its','itself',
 "let\'s",'me','more','most',"mustn\'t",'my','myself','no','nor','not','of','off','on','once','only','or','other','ought','our','ours','ourselves',
 'out','over','own',"same","shan\'t",'she',"she\'d","she\'ll","she\'s",'should',"shouldn\'t",'so','some','such','than','that',"that\'s",'the','their','theirs',
 'them','themselves','then','there',"there\'s",'these','they',"they\'d","they\'ll","they\'re","they\'ve",'this','those','through','to','too','under','until','up',
 'very','was',"wasn\'t",'we',"we\'d","we\'ll","we\'re","we\'ve",'were',"weren\'t",'what',"what\'s",'when',"when\'s",'where',"where\'s",'which','while','who',
 "who\'s",'whom','why',"why\'s",'with',"won\'t",'would',"wouldn\'t",'you',"you\'d","you\'ll","you\'re","you\'ve",'your','yours','yourself','yourselves','']
    spec_char = ['!','.',',','?','(',')','-','_','=','[',']',';',':','\\','/','"','\n','@','|','#','$','%','^','&','*','+','{','}','<','>','~','`']
    targets = comment[0]
    cmt = comment[1]
    cmt = ' '.join(cmt)
    tokens = cmt.split(' ')
    tokens = map(lambda token: token.lower(), tokens)
    tokens = map(lambda token: "".join(c for c in token if c not in spec_char), tokens)
    tokens = filter(lambda token: token not in stop_words, tokens)
    tokens = map(lambda token: "".join(c for c in token if c not in ["'"]), tokens)
    tks = [targets]
    tks.append(tokens)
    return tks

#Calculate term frequency for each term in the corresponding comment.
def tf(comment):
    targets = comment[0]
    tokens = comment[1]
    commentTF = {}
    if tokens != []:
        for token in tokens:
            if token in commentTF:
                commentTF[token] += 1
            else:
                commentTF[token] = 1 
    if commentTF != {}:
        maxTF = commentTF[max(commentTF.keys(), key=(lambda k: commentTF[k]))]
    for token in commentTF.keys():
        commentTF[token] = commentTF[token] / float(maxTF)
    tks = [targets]
    tks.append(commentTF)
    return tks

#Create a list which include all tokens appears in all comment. 
def tokenList(comment):
    tokens = comment[1]
    tokens = list(tokens.keys())
    return tokens

#Create a dictionary which include inverse document frequency for each token. 
def idf(tfList, threshold):
    length = len(tfList)
    idfDict = {}
    for tokens in tfList:
        for token in tokens:
            if token in idfDict:
                idfDict[(token)] += 1
            else:
                idfDict[token] = 1
    for token in idfDict:
        idfDict[token] = math.log(length / float(idfDict[token]))
    idfDict = {k: idfDict[k] for k in list(idfDict.keys())[:threshold]}
    return idfDict

# Use idf dictionary to calculate tfidf for each term in the corresponding comment
def tfidf(comment, idfDict):
    targets = comment[0]
    tokentf = comment[1]
    commentTFIDF = {}
    for token in tokentf:
        if token in idfDict:
            commentTFIDF[token] = tokentf[token] * idfDict[token]
    tfidfs = [targets]
    tfidfs.append(commentTFIDF)
    return tfidfs

#The following two functions together create a vector for each comment. 
#The vector is a list of frequencies for each unique word in the dataset—the TF-IDF value 
#if the word is in the review, or 0.0 otherwise.
def cmtVector(comment,tokenDict):	
    targets = comment[0]
    tokentfidf = comment[1]
    cmtVector = []
    for i, token in enumerate(tokenDict):
        if token in tokentfidf:
            cmtVector.append((i, tokentfidf[token]))
    tokenVector = [targets]
    tokenVector.append(cmtVector)
    return tokenVector
     
def cmtVectorDense(comment, threshold):	
    targets = comment[0]
    tokens = comment[1]
    cmtVector = [0.0] * (threshold +1)
    for token in tokens:        
        cmtVector[token[0]] = token[1]
    tokenVectorDense = [targets]
    tokenVectorDense.append(cmtVector)
    return tokenVectorDense

    
# read train data
comments = sc.textFile(sys.argv[1])
comments = comments.map(lambda line : ([x for x in line.split(",")]))

#seperate comment and target for each line
comments = comments.map(getValueTarget)

# downsample data make target balance
cmt0 = comments.filter(lambda comment: comment[0] == [0,0,0,0,0,0])
cmt1 = comments.filter(lambda comment: comment[0] != [0,0,0,0,0,0])
cmt0 = cmt0.takeSample(True, int(cmt1.count()/2), 2)
cmt0 = sc.parallelize(cmt0,1)
cmts = cmt0.union(cmt1)

#get tfidf for each comment and create a dense matrix
cmt_tokens = cmts.map(token)
cmt_tfs = cmt_tokens.map(tf)
cmt_tfList = cmt_tfs.map(tokenList)
cmt_tfList = cmt_tfList.collect()
cmt_idfs = idf(cmt_tfList, THRESHOLD)
cmt_tfidfs = cmt_tfs.map(lambda comment: tfidf(comment, cmt_idfs))
tokenDict = sorted(cmt_idfs.keys())
cmt_vectors = cmt_tfidfs.map(lambda comment: cmtVector(comment, tokenDict))
cmt_vctDense = cmt_vectors.map(lambda comment: cmtVectorDense(comment, THRESHOLD))
cmt_vctDense.repartition(1).saveAsTextFile(sys.argv[3])

# read test data
test_cmts = sc.textFile(sys.argv[2])
test_cmts = test_cmts.map(lambda line : ([x for x in line.split(",")]))

#seperate comment and target for each line
test_cmts = test_cmts.map(getValueTarget)
#filter out all lines that has no label for the toxicity
test_cmts = test_cmts.filter(lambda comment: comment[0]!= [-1,-1,-1,-1,-1,-1])

# get tfidf for each comment with cmt_idfs get from train data
test_cmt_tokens = test_cmts.map(token)
test_cmt_tfs = test_cmt_tokens.map(tf)
test_cmt_tfidfs = test_cmt_tfs.map(lambda comment: tfidf(comment, cmt_idfs))
test_cmt_vectors = test_cmt_tfidfs.map(lambda comment: cmtVector(comment, tokenDict))

# generate a dense matrix for test data
test_cmt_vctDense = test_cmt_vectors.map(lambda comment: cmtVectorDense(comment, THRESHOLD))
test_cmt_vctDense.saveAsTextFile(sys.argv[4])
print("--- %s seconds ---" % (time.time() - start_time))
sc.stop()
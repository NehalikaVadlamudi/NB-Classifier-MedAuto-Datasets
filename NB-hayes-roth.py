import csv
import math
import random
import urllib

#Handle data
def loadCsv(filename):
  lines = csv.reader(open(filename, "r"))
  next(lines, None) 
  dataset = list(lines)
  for i in range(len(dataset)):
    dataset[i] = [float(x) for x in dataset[i]]
  return dataset

#Test handling data
""" 
filename = 'path/hayes-roth.csv'
SomeDataset = loadCsv(filename)
print("Loaded data file {0:s} with {1:5d} rows".format(filename,len(SomeDataset)))
"""

#Separate by Class
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

#Test separating by class
"""
dataset = [[1,20,1],[2,21,0],[3,22,1]]
separated = separateByClass(dataset)
print('Separated instances: {0}'.format(separated))
"""

#Calculate Mean
def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

#Test stdev & mean calculation
"""
numbers = [1,2,3,4,5]
print('Summary of {0}: mean={1}, stdev={2}'.format(numbers, mean(numbers), stdev(numbers)))
"""

#Summarize Dataset
def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

#Test summarizing data
"""
dataset = [[1,20,0], [2,21,1], [3,22,0]]
summary = summarize(dataset)
print('Attribute summaries: {0}'.format(summary))
"""

#Summarize attributes by class
def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries

#Test summarizing attributes
"""
dataset = [[1,20,1], [2,21,0], [3,22,1], [4,22,0]]
summary = summarizeByClass(dataset)
print('Summary by class value: {0}'.format(summary))
"""

#Calculate Gaussian Probability Density Function
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1/(math.sqrt(2*math.pi)*stdev))*exponent

#Testing Gaussing PDF
"""
x = 71.5
mean = 73
stdev = 6.2
probability = calculateProbability(x,mean,stdev)
print('Probability of belonging to this class: {0}'.format(probability))
"""

#Calculate Class Probabilities
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
		return probabilities

#Testing Class Probability calculation
"""
summaries = {0:[(1, 0.5)], 1:[(20, 5.0)]}
inputVector = [1.1, '?']
probabilities = calculateClassProbabilities(summaries, inputVector)
print('Probabilities for each class: {0}'.format(probabilities))
"""

#Make a prediction
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

#Test prediction
"""
summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
inputVector = [1.1, '?']
result = predict(summaries, inputVector)
print('Prediction: {0}'.format(result))
"""

#Get predictions

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

#Test predictions
"""
summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
testSet = [[1.1,'?'], [19.1,'?']]
predictions = getPredictions(summaries, testSet)
print('Predictions: {0}',format(predictions))
"""

#Get Accuracy
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet)))*100.0

#Test accuracy
"""
testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
predictions = ['a', 'a', 'a']
accuracy = getAccuracy(testSet, predictions)
print('Accuracy: {0}'.format(accuracy))
"""

#implementing cross validation split
def crossValSplit(dataset, numFolds):
  '''
  Description:
      Function to split the data into number of folds specified
  Input:
      dataset: data that is to be split
      numFolds: integer - number of folds into which the data is to be split
  Output:
      split data
  '''
  dataSplit = list()
  dataCopy = list(dataset)
  foldSize = int(len(dataset) / numFolds)
  for _ in range(numFolds):
      fold = list()
      while len(fold) < foldSize:
          index = random.randrange(len(dataCopy))
          fold.append(dataCopy.pop(index))
      dataSplit.append(fold)
  return dataSplit


filename = '/content/hayes-roth.csv'
dataset = loadCsv(filename)

#create crosss validation fold
numFolds = 2
folds = crossValSplit(dataset, numFolds)

scores = list()
for fold in folds:
  trainSet = list(folds)
  trainSet.remove(fold)
  trainSet = sum(trainSet, [])
  testSet = list()
  for row in fold:
    rowCopy = list(row)
    testSet.append(rowCopy)           
  #prepare model
  summaries = summarizeByClass(trainSet)
  #test model
  predictions = getPredictions(summaries, testSet)
  accuracy = getAccuracy(testSet, predictions)   
  scores.append(accuracy)
print('Scores: %s' % scores)      
print('Maximum Accuracy: %3f%%' % max(scores))
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
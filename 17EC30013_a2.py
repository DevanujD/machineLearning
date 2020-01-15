#17EC30013    
#Devanuj Deka
#Assignment 2 (Naive Bayes Classifier)

import numpy as np
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("data2_19.csv")
test_set = pd.read_csv("test2_19.csv")
dataset = dataset["D,X1,X2,X3,X4,X5,X6"].str.split(',', expand = True)      #given dataset is not in usable form, since all values are written in one column, separated by commas.
test_set = test_set["D,X1,X2,X3,X4,X5,X6"].str.split(',', expand = True)    #I used dataset["D,X1,X2,X3,X4,X5,X6"].str.split to split this column into different columns, by setting the separators as the commas (',').

dataset.columns = ['D', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6']         #reassigning column names from index to the specified names.
test_set.columns = ['D', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6']

dataset = dataset.astype(int)                       #given dataset is in string type, i used dataset.astype(int) to typecast the values to int data type.
test_set = test_set.astype(int)

X_test = test_set.iloc[:, 1:]               #selecting features into X_test
Y_test = test_set.iloc[:, 0]                #Selecting the result column into Y_test

X_train = dataset.iloc[:, 1:]
Y_train = dataset.iloc[:, 0]

def prob(dataset, result, column, value):    #parameters denoting the value of the given attribute, whose individual conditional probability is to be calculated given value in "D" is 'result'
    d = 0
    n = 0
    col = dataset.loc[:, column]
    Y = dataset.loc[:, "D"]
    for i in range(len(Y)):
        if Y[i] == result:
            d = d + 1           #counting number of times in the dataset this result occurs
            if col[i] == value:
                n = n + 1       #among the cases with the given result, counting the number of times the given attribute value occurs
    n = n + 1                   #These two tweaks are done
    d = d + 5                   #for Laplacian smoothing
    return n/d

Probabilities = {0:{}, 1:{}}    #records conditional probabilities of all possible values in any given column given all possible values in result
values = [1, 2, 3, 4, 5]        #list of possible values in each of the attributes


for i in X_train.columns:
    Probabilities[0][i] = {}    #records all possible conditional probabilities for result = 0, column = i
    Probabilities[1][i] = {}    #records all possible conditional probabilities for result = 1, column = i
    for value in values:
        Probabilities[0][i][value] = prob(dataset, 0, i, value)   #records conditional probability of column = i, value = value, given result = 0
        Probabilities[1][i][value] = prob(dataset, 1, i, value)   #records conditional probability of column = i, value = value, given result = 1
    
      
Y_pred = []

for j in range(0, len(X_test)):
    prob_0 = 1
    prob_1 = 1
    for column in X_test.columns:
        prob_0 = prob_0*Probabilities[0][column][X_test[column].iloc[j]]     #probability of given attributes given that result = 0 
        prob_1 = prob_1*Probabilities[1][column][X_test[column].iloc[j]]     #probability of given attributes given that result = 1
        
    if(prob_0 > prob_1):
        Y_pred.append(0)            #recording the predicted values; when
    else:                           #probability of result=0 > result=1, the
        Y_pred.append(1)            #model predicts 0, otherwise predicts 1
   
count = 0     
for i in range(0, len(Y_pred)):
    if Y_pred[i] == Y_test[i]:
        count += 1

print("Accuracy of prediction for this Bayesian Classifier: ", count/len(Y_pred))
print("=", 100*count/len(Y_pred), "%")


#sum = 0
#for i in range(len(Y_train)):               #counting number of times the result is zero in the dataset
#    if (Y_train[i] == 0):
#        sum = sum + 1

#no_of_zeroes = sum
#no_of_ones = len(Y_train) - sum

#To write the confusion matrix, we count the number of true positives, false positives, false negatives and true negatives
#tp, fp, fn, tn = 0, 0, 0, 0
#for i in range(0, len(Y_pred)):
#    if Y_pred[i] == 0:
#        if Y_test.iloc[i] == 0:
#            tp += 1
#        else:
#            fp += 1
#    elif Y_pred[i] == 1:
#        if Y_test.iloc[i] == 1:
#            tn += 1
#        else:
#            fn += 1
#"""
    
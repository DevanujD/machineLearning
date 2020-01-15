# 17EC30013     # Devanuj Deka     # Assignment 1 (Decision Tree), Machine Learning - 60050

# Importing necessary libraries
import numpy as np
import pandas as pd              #I tried this code by importing the dataset using numpy, but couldn't fing relevant syntax for accessing data elements in time
import pprint

delta  = np.finfo(float).eps #it is an infinitesimal number to avoid divide by zero error or log(0) error

# Importing Dataset
dataset = pd.read_csv('data1_19.csv')

#Calculates the entropy of a column of a table
#entropy_of_column(dataset,pclass) will calculate the entropy of pclass column
def entropy_of_column(table, column):     #table.loc takes the header name. table.iloc takes the header index.
    current = table.loc[:, column]
    values = list(current.unique())
    total_entropy = 0
    Y_or_N = list(table.iloc[:, -1].unique())
    for V in values:
        entropy = 0
        for Y_N in Y_or_N:
            num = len(table[column][table[column] == V][table.iloc[:, -1] == Y_N]) #No. of examples with current feature value, which have the current result value (Y or N)
            denom = len(table[column][table[column] == V])  #No. of examples with the current feature value
            fraction =  num / (denom + delta)
            entropy += -fraction * np.log2(fraction + delta)
        fract2 = denom / len(table)
        total_entropy += -fract2 * entropy
    
    return abs(total_entropy)    #Returns absolute value of total_entropy


#Calculates the entropy of an entire table
def entropy_of_table(table):
    p_plus = 0
    p_minus = 0
    n_plus = (table.iloc[:,-1] == "yes").sum()
    n_minus = (table.iloc[:,-1] == "no").sum()
    total = n_plus + n_minus
    p_plus = n_plus / (total + delta)
    p_minus = n_minus / (total + delta)
    entropy = -1*(p_plus*(np.log2(p_plus)) + p_minus*(np.log2(p_minus)))
    return entropy

#Returns the best feature attribute to split
#Use Information Gain i.e entropy of table - entropy after splitting of a column
def get_best(table):
    Infogain = []
    for heads in table.keys()[:-1]:
        Infogain.append(entropy_of_table(table) - entropy_of_column(table,heads))
        
    best = np.argmax(Infogain)   #stores the index of the column (attribute) for the best split
    return table.keys()[:-1][best]  #returns the column index for the best split

#after splitting on a particular attribute, make a new table again
#and then find the best attribute of these table
def delete_table(table, node, value):
    return (table[table[node] == value].reset_index(drop = True)).drop(node,axis = 1)

#Building a tree as a dictonary
def build_tree(table, tree = None):
   names = table.columns
   if len(names) == 1:      #when we reach a node which has only one example, we stop branching, and return this to the tree
        return tree
    
   node = get_best(table)
   att_values = np.unique(table[node])
   newtable = table
   if tree is None:
        tree = {}
        tree[node] = {}
        
   for value in att_values:
        #print(value)
       subtable = delete_table(newtable, node, value)            #input subtable which is a result of the split at the previous node
       clValue, counts = np.unique(subtable['survived'], return_counts = True)
       if len(counts) == 1:
           tree[node][value] = clValue[0]
       elif counts[0]/counts[1] > 100 or (len(names) == 2 and counts[0]/counts[1] > 1):
           tree[node][value] = clValue[0]
       elif counts[1]/counts[0] > 100 or (len(names) == 2 and counts[1]/counts[0] > 1): 
           tree[node][value] = clValue[1]
       else:
           tree[node][value] = build_tree(subtable)  #Split further on the attribute with the best split (node)
    
   return tree      #when it reaches a node which need not be further split (leaf), return this leaf to the previous node

#building the tree on the dataset
Tree = build_tree(dataset)
pprint.pprint(Tree, depth = 6, width = 50)        #depth argument chosen based on elbow method

#Create the prediction array
def Predict(data, Tree):
    for splits in Tree.keys():
        value = data[splits]
        if value not in Tree[splits]:
            return 'No'
        
        Tree = Tree[splits][value]
        prediction = 0
        if type(Tree) is dict:
            prediction = Predict(data,Tree)
        else:
            prediction = Tree
            break;
    
    return prediction

#Prediction array
our_pred = []
for i in range(len(dataset)):
    data = dataset.iloc[i]
    our_pred.append(Predict(data, Tree))
    
#Count the number of correct predictions based on the dataset
count = 0
for i in range(len(our_pred)):
    if our_pred[i] == dataset.survived[i]:
        count += 1
        
#Finding the accuracy (percentage) of our predictions        
accuracy_of_tree = (count / len(our_pred))

    
    
    
    
    
    
    


    
    

    



    


    
    
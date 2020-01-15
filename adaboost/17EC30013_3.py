#17EC30013     #Devanuj Deka     #Assignment 3 Adaboost     #Due to random sampling for the new dataset in every iteration, the errors change, altering the importance of the classifier and subsequent sampled datasets, changing the overall final classifier and its accuracy.
#So, if accuracy comes out to be too low, just restart kernel and run it again. I have obtained various results for the accuracy, incuding extreme values such as approx 7.6%, and even as high as 81.54%. But most commonly the program gives an accuracy of about 56.9% or around 46.2%.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
import pprint

delta=np.finfo(float).eps

# Importing the dataset
dataset = pd.read_csv('data3_19.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values
test = pd.read_csv('test3_19.csv')
test.columns = ['pclass','age','gender','survived']
X_train = pd.DataFrame(X)
Y_train = pd.DataFrame(Y)

##Decision Tree code

#calculates the entropy of a column of a table
#cal_entropy(dataset,pclass) will calculate the entropy of pclass column
def cal_entropy(table,column):
    current=table.loc[:,column]
    values = list(current.unique())
    tentropy=0
    Y_N = list(table.iloc[:,-1].unique())
    for V in values:
        entropy=0
        for YN in Y_N:
            Numerator = len(table[column][table[column]==V][table.iloc[:,-1] ==YN])
            Denominator = len(table[column][table[column]==V])
            Fract=Numerator/(Denominator+delta)
            entropy += -Fract*np.log2(Fract+delta)
        Fract2=Denominator/len(table)
        tentropy += -Fract2*entropy
    
    return abs(tentropy)


#calculates the entropy of an entire table
def cal_entropy_init (table):
    pp = 0
    pn = 0
    pp=(table.iloc[:,-1]=="yes").sum()
    pn=(table.iloc[:,-1]=="no").sum()
    total = pp + pn
    pp = pp/(total+delta)
    pn = pn/(total+delta)
    entropy = -1*(pp*(np.log2(pp))+pn*(np.log2(pn)))
    return entropy

#gives us the best attribute to which we will split
#we use our infogain here i.e entropy of table - entropy after splitting of a column
def get_best(table):
    Infogain=[]
    for heads in table.keys()[:-1]:
        Infogain.append(cal_entropy_init(table)-cal_entropy(table,heads))
        
    best=np.argmax(Infogain)
    return table.keys()[:-1][best]

#after the data has been split on a particular attribute we need to make a new table again
#the find the best attribute of these table
def delete_table(table,node,value):
    return (table[table[node]==value].reset_index(drop=True)).drop(node,axis=1)

#these funtion builds a tree as a dictonary
def build_tree(table,tree=None):
   names=table.columns
   if len(names)==1:
        return tree
   node=get_best(table)
   att_values=np.unique(table[node])
   newtable=table
   if tree is None:
        tree={}
        tree[node]={}
        
   for value in att_values:
        #print(value)
       subtable=delete_table(newtable,node,value)
       clValue,counts=np.unique(subtable['survived'],return_counts=True)
       if len(counts)==1:
           tree[node][value]=clValue[0]
       elif counts[0]/counts[1]>100 or (len(names)==2 and counts[0]/counts[1]>1):
           tree[node][value]=clValue[0]
            
       elif counts[1]/counts[0]>100 or (len(names)==2 and counts[1]/counts[0]>1): 
            tree[node][value]=clValue[1]
       else:
           tree[node][value]=build_tree(subtable)
    
   return tree

#Create the prediction array
def Predict(data,Tree):
    for splits in Tree.keys():
        value=data[splits]
        
        if value not in Tree[splits]:
            return 'No'
        
        Tree=Tree[splits][value]
        prediction = 0
        
        if type(Tree) is dict:
            prediction = Predict(data,Tree)
            
        else:
            prediction= Tree
            break;
    
    return prediction

#Boosting Part
Tree=[]
count=[]
count = np.array(count)
ind=[]
error=[]
error=np.ones(4,dtype=float)*(1e-10)
error[2] = 2e-8
error[3] = 1e-9
imp = np.zeros(4,dtype=float)
Z=np.zeros(4,dtype=float)
wt = np.zeros((5,2150),dtype=float)
wt[0] = (1/2150)*np.ones(2150)
train_data = dataset.iloc[:, :]

#Building a tree for the modified dataset in each iteration
for i in range(0,4):
    if i==0:
        ind2=list(np.arange(2150))
        ind.append(ind2)
    else:
        ind2=list(np.random.choice(np.arange(2150),2150,True,wt[i]))
        ind.append(ind2)

    train_data = dataset.iloc[ind[-1] , :].values
    train_data = pd.DataFrame(train_data)
    train_data.columns = ['pclass','age','gender','survived']


    #building the tree on this dataset
    Tree.append(build_tree(train_data))
    pprint.pprint(Tree[i], depth=6, width=50)


    #Prediction array
    our_pred = []

    #Predicting values on this datset using the constructed tree in this iteration
    for j in range(len(train_data)):
        data=train_data.iloc[j]
        our_pred.append(Predict(data,Tree[i]))
    

    count=np.append(count,0)
    #Calculating the error of this tree, and its importance (voting power)
    for j in range(len(our_pred)):
        if our_pred[j]==train_data.survived[j]:
            count[i]=count[i]+1
        else:
            error[i]= error[i] + (1/2150)*wt[i][ind[i][j]]
                
        imp[i] = np.log((1-error[i])/(error[i] + delta))*0.5

    #Updating Weights for the next iteration
    for j in range(len(our_pred)):
        if our_pred[j] == train_data.survived[j]:
            wt[i+1][ind[i][j]] = wt[i][ind[i][j]]*np.exp(-imp[i])
        else:
            wt[i+1][ind[i][j]] = wt[i][ind[i][j]]*np.exp(imp[i])

    #Normalizing the weights
    Z[i]=wt[i+1].sum()
    wt[i+1] = wt[i+1]/Z[i]
    
    #end of loop iteration
    




#Finding Importance * Base-classifier in each iteration, sum of which is the final classifier
pred = np.zeros((4,65))

for i in range(0, 4):
    for j in range(len(test)):
        data=test.iloc[j]
        if(Predict(data,Tree[i]) == 'yes'):
            pred[i][j]=1*imp[i]
        else:
            pred[i][j]=-1*imp[i]
    
  
        
final=np.ones(65)
#Boosted Classifier
final = pred[0] + pred[1] + pred[2]


cnt=0
#Predicting Test set results
for i in range(len(test)):
    if ((final[i]>0 and test.iloc[i,3]=='yes')or(final[i]<0 and test.iloc[i,3]=='no')):
        cnt+=1
    
#Accuracy
accuracy = (cnt/len(test))*100
print(accuracy, '%')

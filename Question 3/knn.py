#-------------------------------------------------------------------------
# AUTHOR: Hoang Quan Dinh
# FILENAME: knn.py
# SPECIFICATION: knn
# FOR: CS 4210- Assignment #2
# TIME SPENT: 45min
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

correct = 0
total = 0
#Reading the data in a csv file using pandas
db = []
df = pd.read_csv('email_classification.csv')
for _, row in df.iterrows():
    db.append(row.tolist())

#Loop your data to allow each instance to be your test set
for i in db:
    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    X = []
    for j in db:
        if j != i:
            newJ = [float(b) for b in j[:-1]]
            X.append(newJ)

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    Y = []
    for k in db:
        if k != i:
            if k[-1] == 'ham': Y.append(float(1)) #1 = ham, 2 = spam
            else: Y.append(float(2))

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = i[:-1]

    #Fitting the knn to the data using k = 1 and Euclidean distance (L2 norm)
    #--> add your Python code here
    clf = KNeighborsClassifier(n_neighbors=1, p = 2)
    clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    class_prediction = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    testExpected = i[-1]
    if testExpected == 'ham': testExpected = 1
    else: testExpected = 2

    if testExpected == class_prediction: correct += 1

    total += 1
#Print the error rate
#--> add your Python code here
errorRate = (total - correct) / total
print('The error rate is: ' + str(errorRate))







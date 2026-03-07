#-------------------------------------------------------------------------
# AUTHOR: Hoang Quan Dinh
# FILENAME: naive_bayes.py
# SPECIFICATION: Bayes
# FOR: CS 4210- Assignment #2
# TIME SPENT:
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

dbTraining = []
dbTest = []
X = []
Y = []
encoder = OrdinalEncoder()

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
X = encoder.fit_transform([row[:-1] for row in dbTraining])

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
for row in dbTraining:
    if row[-1] == 'Yes': Y.append(1)
    else: Y.append(2)

#Fitting the naive bayes to the data using smoothing
#--> add your Python code here
clf = GaussianNB()
clf.fit(X, Y)

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header os the solution
#--> add your Python code here
print('Day'.ljust(15) + 'Outlook'.ljust(15) + 'Temperature'.ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + 'Confidence'.ljust(15))
#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
XTest = encoder.fit_transform([row[:-1] for row in dbTest])
for i in range(len(dbTest)):
    predictedConfidence = clf.predict_proba([XTest[i]])[0]
    for j in range(len(predictedConfidence)):
        if predictedConfidence[j] >= 0.75:
            for k in dbTest[i][:-1]:
                print(k.ljust(15), end='')
            if j == 0: print('Yes'.ljust(15), end='')
            else: print('No'.ljust(15), end='')
            print(predictedConfidence[j])